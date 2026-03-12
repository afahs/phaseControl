#!/usr/bin/env python3
"""Async-buffered variant of the beacon VCXO PLL controller.

This keeps the beacon controller's state machine, logging, and control logic,
but replaces per-update `capture_data(...)` polling with a background
`device.stream(...)` reader that fills an asyncio queue.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass

import numpy as np

import beacon_vcxo_pll_controller as base


@dataclass
class QueuedBlock:
    unix_time_s: float
    iq_block: np.ndarray


@dataclass
class StreamSentinel:
    error: BaseException | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = base.build_arg_parser()
    parser.add_argument(
        "--async_queue_blocks",
        type=int,
        default=None,
        help="Maximum buffered SDR blocks in the async queue. Defaults to 4x blocks_per_update.",
    )
    return parser


def bytes_to_iq_block(raw_bytes: bytes, nsamples: int) -> np.ndarray:
    expected = 2 * int(nsamples)
    if len(raw_bytes) != expected:
        raise RuntimeError(
            "Expected {} bytes for one IQ block, received {}.".format(
                expected, len(raw_bytes)
            )
        )
    raw = (np.frombuffer(raw_bytes, dtype=np.uint8) - 128).view(np.int8)
    raw = raw.reshape(int(nsamples), 2).astype(np.float32, copy=False)
    return (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)


def default_async_queue_blocks(blocks_per_update: int) -> int:
    blocks_per_update = max(1, int(blocks_per_update))
    return max(8, 4 * blocks_per_update)


def signal_stream_end(
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
    error: BaseException | None = None,
) -> None:
    sentinel = StreamSentinel(error=error)
    while True:
        try:
            queue.put_nowait(sentinel)
            return
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return


async def stream_iq_blocks(
    device,
    nsamples: int,
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
) -> None:
    try:
        async for raw in device.stream(
            num_samples_or_bytes=2 * int(nsamples),
            format="bytes",
        ):
            iq_block = bytes_to_iq_block(raw, nsamples)
            await queue.put(
                QueuedBlock(
                    unix_time_s=float(time.time()),
                    iq_block=iq_block,
                )
            )
    except asyncio.CancelledError:
        signal_stream_end(queue)
        raise
    except Exception as exc:
        signal_stream_end(queue, error=exc)
        raise
    else:
        signal_stream_end(queue)
    finally:
        try:
            await device.stop()
        except Exception:
            pass


async def read_update_blocks(
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
    blocks_per_update: int,
) -> tuple[list[np.ndarray], float | None]:
    blocks: list[np.ndarray] = []
    t_meas: float | None = None

    while len(blocks) < int(blocks_per_update):
        item = await queue.get()
        if isinstance(item, StreamSentinel):
            if item.error is not None:
                raise RuntimeError("Async SDR stream terminated unexpectedly.") from item.error
            break
        blocks.append(item.iq_block)
        t_meas = float(item.unix_time_s)

    return blocks, t_meas


async def stop_stream_task(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def run_controller(args: argparse.Namespace) -> int:
    params = base.params_from_args(args)
    log_csv = base.make_log_path(args.log_csv)
    logger = base.CsvLogger(log_csv)
    print("Logging to: {}".format(log_csv), flush=True)

    mcfg = base.MeasureConfig(
        fs=params.sample_rate,
        beacon_hz=params.beacon_freq_hz,
        center_hz=params.center_freq_hz,
        fft_search_span_hz=params.fft_search_span_hz,
    )

    state = base.ControlState(
        mode=(
            base.MODE_OPEN
            if params.open_loop
            else (
                base.MODE_ADJUSTED
                if params.adjusted_loop
                else (
                    base.MODE_QUALITY_SELF
                    if params.quality_self
                    else (
                        base.MODE_QUALITY_BEACON
                        if params.quality_beacon
                        else base.MODE_ACQUIRE
                    )
                )
            )
        ),
        clock0_cmd_hz=float(params.clock0_nom_hz),
        dac_code_cmd=int(params.dac_code_start),
        t_start_s=base.now_unix(),
        phase_setpoint_rad=float(params.phase_setpoint_rad),
    )
    phase_hist = base.PhaseHistory()
    cal_state = base.SetpointCalState(
        enabled=bool(params.calibrate_phase_setpoint),
        target_samples=int(params.setpoint_cal_samples),
        done=(not bool(params.calibrate_phase_setpoint)),
    )

    stopper = base.GracefulStop()
    sdr = None
    clock_obj = None
    stream_task: asyncio.Task[None] | None = None

    try:
        sdr = base.sdr_open(params.sample_rate, params.center_freq_hz, params.gain)

        if not (
            params.open_loop
            or params.adjusted_loop
            or params.quality_self
            or params.quality_beacon
        ):
            if base.cvh is None:
                raise RuntimeError(
                    "closed_loop selected but clock_vcxo_helper.py import failed"
                )
            clock_obj = base.clock_create()
            base.clock_open(clock_obj)
            base.clock_init_28p8(clock_obj, apr_ppm=params.apr_ppm)
            state.clock0_cmd_hz = base.clock_set_coarse_hz(
                clock_obj,
                params.clock0_nom_hz,
                settle_s=params.coarse_settle_s,
            )
            state.dac_code_rb = base.clock_set_dac_code(
                clock_obj,
                state.dac_code_cmd,
                settle_s=params.fine_settle_s,
            )
            state.dac_volts = base.clock_dac_code_to_volts(
                clock_obj,
                state.dac_code_cmd,
            )
        else:
            state.clock0_cmd_hz = float(params.clock0_nom_hz)
            state.dac_volts = float("nan")

        t_block = float(params.nsamp_block) / float(params.sample_rate)
        t_upd_nom = t_block * float(params.blocks_per_update)
        if params.verbose:
            print(
                "Mode={} | Fs={:.1f} | block={} ({:.3f}s) | blocks/update={} (~{:.3f}s) | center={:.6f} MHz | beacon={:.6f} MHz".format(
                    state.mode,
                    params.sample_rate,
                    params.nsamp_block,
                    t_block,
                    params.blocks_per_update,
                    t_upd_nom,
                    params.center_freq_hz / 1e6,
                    params.beacon_freq_hz / 1e6,
                ),
                flush=True,
            )

        queue_depth = args.async_queue_blocks
        if queue_depth is None:
            queue_depth = default_async_queue_blocks(params.blocks_per_update)
        queue_depth = max(int(queue_depth), int(params.blocks_per_update))
        block_queue: asyncio.Queue[QueuedBlock | StreamSentinel] = asyncio.Queue(
            maxsize=queue_depth
        )
        stream_task = asyncio.create_task(
            stream_iq_blocks(sdr, params.nsamp_block, block_queue)
        )

        while not stopper.stop:
            if params.run_seconds is not None and (
                base.now_unix() - state.t_start_s
            ) >= float(params.run_seconds):
                break
            if params.max_updates is not None and state.update_index >= int(
                params.max_updates
            ):
                break

            iq_blocks, t_meas = await read_update_blocks(
                block_queue, params.blocks_per_update
            )
            if len(iq_blocks) < int(params.blocks_per_update):
                break
            assert t_meas is not None

            state.setpoint_cal_event = 0
            dac_step_result = None
            quality_seeded = False

            if params.quality_self:
                if not state.nco_initialized:
                    if params.quality_self_tone_hz is not None:
                        state.nco_freq_bb_hz = float(params.quality_self_tone_hz)
                    else:
                        state.nco_freq_bb_hz = base.estimate_dominant_tone_offset_hz(
                            iq_blocks,
                            sample_rate_hz=params.sample_rate,
                            dc_exclusion_hz=params.quality_self_dc_exclusion_hz,
                        )
                    state.nco_initialized = True
                    state.quality_sample_start = 0
                    phase_hist.reset()
                    quality_seeded = True

                meas, next_sample_start = (
                    base.build_update_measurement_from_quality_self(
                        iq_blocks,
                        mcfg,
                        phase_hist,
                        t_meas,
                        tone_offset_hz=state.nco_freq_bb_hz,
                        sample_start=state.quality_sample_start,
                    )
                )
                state.quality_sample_start = int(next_sample_start)

            elif params.quality_beacon:
                if state.nco_initialized:
                    meas, nco_end = base.build_update_measurement_from_track(
                        iq_blocks,
                        mcfg,
                        phase_hist,
                        t_meas,
                        state.nco_freq_bb_hz,
                        state.nco_phase_rad,
                    )
                    state.nco_phase_rad = float(nco_end)
                else:
                    meas = base.build_update_measurement_from_acquire(
                        iq_blocks, mcfg, phase_hist, t_meas
                    )

            else:
                use_persistent = False
                if params.adjusted_loop:
                    use_persistent = state.nco_initialized
                elif params.open_loop and params.open_loop_persistent_ref:
                    use_persistent = state.nco_initialized
                elif not params.open_loop:
                    use_persistent = (
                        state.mode == base.MODE_TRACK and state.nco_initialized
                    )

                if use_persistent:
                    meas, nco_end = base.build_update_measurement_from_track(
                        iq_blocks,
                        mcfg,
                        phase_hist,
                        t_meas,
                        state.nco_freq_bb_hz,
                        state.nco_phase_rad,
                    )
                    state.nco_phase_rad = float(nco_end)
                else:
                    meas = base.build_update_measurement_from_acquire(
                        iq_blocks, mcfg, phase_hist, t_meas
                    )

            if state.nco_initialized:
                state.df_nco_peak_hz = float(
                    state.nco_freq_bb_hz - meas.peak_freq_bb_hz
                )
            else:
                state.df_nco_peak_hz = 0.0
            base.sync_state_with_phase_history(state, phase_hist)

            state.phase_ctrl_err_raw_rad = base.wrap_to_pi(
                meas.phase_meas_rad - state.phase_setpoint_rad
            )
            state.phase_ctrl_err_filt_rad = state.phase_ctrl_err_raw_rad

            if params.quality_self:
                state.mode = base.MODE_QUALITY_SELF
                state.action = "qself_seed_tone" if quality_seeded else "qself_monitor"
                state.phase_ctrl_err_raw_rad = float(meas.phase_meas_rad)
                state.phase_ctrl_err_filt_rad = float(meas.phase_meas_rad)

            elif params.quality_beacon:
                state.mode = base.MODE_QUALITY_BEACON
                if not state.nco_initialized:
                    if base._snr_ok_for_acq(params, meas):
                        state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                        state.nco_phase_rad = 0.0
                        state.nco_initialized = True
                        state.df_nco_peak_hz = 0.0
                        phase_hist.reset()
                        state.action = "qbeacon_seed_nco"
                    else:
                        state.action = "qbeacon_wait_seed"
                else:
                    state.action = "qbeacon_monitor"
                state.phase_ctrl_err_raw_rad = float(meas.phase_meas_rad)
                state.phase_ctrl_err_filt_rad = float(meas.phase_meas_rad)

            elif params.open_loop:
                state.mode = base.MODE_OPEN
                if params.open_loop_persistent_ref:
                    if not state.nco_initialized:
                        if base._snr_ok_for_acq(params, meas):
                            state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                            state.nco_phase_rad = 0.0
                            state.nco_initialized = True
                            state.df_nco_peak_hz = 0.0
                            phase_hist.reset()
                            state.action = "open_seed_nco"
                        else:
                            state.action = "open_wait_seed"
                    else:
                        state.action = "open_persistent_monitor"
                else:
                    state.action = "open_stream"
                    state.nco_initialized = False

            elif params.adjusted_loop:
                state.mode = base.MODE_ADJUSTED
                if not state.nco_initialized and base._snr_ok_for_acq(params, meas):
                    state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                    state.nco_phase_rad = 0.0
                    state.nco_initialized = True
                    phase_hist.reset()
                    state.action = "adjust_seed_nco"
                else:
                    state.action = "adjust_monitor"
                    if state.nco_initialized and params.track_use_fft_freq_assist:
                        alpha = base.clamp(params.track_nco_alpha, 0.0, 0.999999)
                        state.nco_freq_bb_hz = (
                            alpha * state.nco_freq_bb_hz
                            + (1.0 - alpha) * float(meas.peak_freq_bb_hz)
                        )
                        state.df_nco_peak_hz = float(
                            state.nco_freq_bb_hz - meas.peak_freq_bb_hz
                        )

                if state.nco_initialized:
                    state.phase_ctrl_err_raw_rad = base.wrap_to_pi(
                        meas.phase_meas_rad - state.phase_setpoint_rad
                    )
                    state.phase_ctrl_err_filt_rad = base.circular_phase_lpf_update(
                        state.phase_ctrl_err_raw_rad,
                        params.phase_filt_alpha,
                        state.phase_filt,
                    )

                if (
                    state.nco_initialized
                    and base._phase_quality_ok_for_ctrl(params, meas)
                    and (not cal_state.done)
                ):
                    done_now, setpoint = cal_state.push(meas.phase_meas_rad)
                    if done_now:
                        state.phase_setpoint_rad = float(setpoint)
                        state.phase_filt = base.CircularLpfState()
                        state.setpoint_cal_event = 1
                        state.action = "adjust_setpoint_cal"

            else:
                if state.mode == base.MODE_ACQUIRE:
                    snr_ok = base._snr_ok_for_acq(params, meas)
                    if not snr_ok:
                        state.lock_counter = 0
                        state.unlock_counter += 1
                        state.action = "acq_hold_low_snr"
                    else:
                        state.unlock_counter = 0
                        fft_freq_err = float(
                            meas.peak_freq_abs_hz - params.beacon_freq_hz
                        )
                        if params.fft_pullin_enable and (not state.fft_pullin_done):
                            changed = base.coarse_update_from_freq_error(
                                fft_freq_err, params, state, clock_obj
                            )
                            state.action = (
                                "acq_coarse_fft"
                                if changed
                                else "acq_coarse_fft_hold"
                            )
                            state.fft_pullin_done = True
                        elif params.fft_pullin_enable and (
                            not state.fft_dac_pretrim_done
                        ):
                            dac_step_result = base.dac_feedforward_from_freq_error(
                                fft_freq_err, params, state, clock_obj
                            )
                            state.action = (
                                "acq_dac_fft"
                                if dac_step_result.changed
                                else "acq_dac_fft_hold"
                            )
                            state.fft_dac_pretrim_done = True
                        else:
                            changed = base.coarse_update_from_freq_error(
                                meas.freq_err_block_hz, params, state, clock_obj
                            )
                            state.action = (
                                "acq_coarse_fll"
                                if changed
                                else "acq_coarse_fll_hold"
                            )

                        if (
                            abs(meas.freq_err_block_hz) < params.f_switch_hz
                            and meas.coh >= params.coh_min_trk
                            and meas.snr_db >= params.snr_min_trk_db
                        ):
                            state.lock_counter += 1
                        else:
                            state.lock_counter = 0

                        if state.lock_counter >= params.lock_count_required:
                            state.mode = base.MODE_TRACK
                            state.lock_counter = 0
                            state.unlock_counter = 0
                            state.I_term = 0.0
                            state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                            state.nco_phase_rad = 0.0
                            state.nco_initialized = True
                            state.df_nco_peak_hz = 0.0
                            state.phase_filt = base.CircularLpfState()
                            phase_hist.reset()
                            state.action = "switch_to_track"

                elif state.mode == base.MODE_TRACK:
                    if params.track_use_fft_freq_assist and state.nco_initialized:
                        alpha = base.clamp(params.track_nco_alpha, 0.0, 0.999999)
                        state.nco_freq_bb_hz = (
                            alpha * state.nco_freq_bb_hz
                            + (1.0 - alpha) * float(meas.peak_freq_bb_hz)
                        )
                        state.df_nco_peak_hz = float(
                            state.nco_freq_bb_hz - meas.peak_freq_bb_hz
                        )

                    stay_ok = base._phase_quality_ok_for_track(params, meas)
                    ctrl_ok = base._phase_quality_ok_for_ctrl(params, meas)
                    state.phase_ctrl_err_raw_rad = base.wrap_to_pi(
                        meas.phase_meas_rad - state.phase_setpoint_rad
                    )
                    state.phase_ctrl_err_filt_rad = base.circular_phase_lpf_update(
                        state.phase_ctrl_err_raw_rad,
                        params.phase_filt_alpha,
                        state.phase_filt,
                    )

                    if ctrl_ok and (not cal_state.done):
                        done_now, setpoint = cal_state.push(meas.phase_meas_rad)
                        if done_now:
                            state.phase_setpoint_rad = float(setpoint)
                            state.phase_filt = base.CircularLpfState()
                            state.I_term = 0.0
                            state.phase_ctrl_err_raw_rad = base.wrap_to_pi(
                                meas.phase_meas_rad - state.phase_setpoint_rad
                            )
                            state.phase_ctrl_err_filt_rad = (
                                state.phase_ctrl_err_raw_rad
                            )
                            state.setpoint_cal_event = 1
                            state.action = "track_setpoint_cal"

                    if not stay_ok:
                        state.unlock_counter += 1
                        base.holdover_update(params, state)
                        state.action = "trk_hold_lowq"
                        if state.unlock_counter >= params.unlock_count_required:
                            state.mode = base.MODE_HOLDOVER
                            state.action = "to_holdover"
                    elif base._track_stability_inconsistent(params, state, meas):
                        state.mode = base.MODE_ACQUIRE
                        state.lock_counter = 0
                        state.unlock_counter = 0
                        state.nco_initialized = False
                        state.phase_filt = base.CircularLpfState()
                        phase_hist.reset()
                        state.action = "trk_reacquire_freq"
                    else:
                        state.unlock_counter = 0
                        if ctrl_ok:
                            dac_step_result = base.pi_update_dac(
                                state.phase_ctrl_err_filt_rad,
                                params,
                                state,
                                clock_obj,
                                allow_integrate=True,
                            )
                            state.action = (
                                "dac_fine" if dac_step_result.changed else "dac_hold"
                            )
                        else:
                            base.pi_update_dac(
                                state.phase_ctrl_err_filt_rad,
                                params,
                                state,
                                clock_obj=None,
                                allow_integrate=False,
                            )
                            state.action = "trk_hold_ctrlq"

                elif state.mode == base.MODE_HOLDOVER:
                    base.holdover_update(params, state)
                    state.action = "holdover"
                    if base._snr_ok_for_acq(params, meas) and (
                        meas.coh >= params.coh_min_acq
                    ):
                        state.mode = base.MODE_ACQUIRE
                        state.unlock_counter = 0
                        state.lock_counter = 0
                        state.nco_initialized = False
                        phase_hist.reset()
                        state.action = "recover_to_acq"
                else:
                    state.mode = base.MODE_ACQUIRE
                    state.action = "reset_mode"

            if (
                (
                    not (
                        params.open_loop
                        or params.adjusted_loop
                        or params.quality_self
                        or params.quality_beacon
                    )
                )
                and clock_obj is not None
                and state.dac_volts is None
            ):
                try:
                    state.dac_volts = base.clock_dac_code_to_volts(
                        clock_obj, state.dac_code_cmd
                    )
                except Exception:
                    state.dac_volts = float("nan")

            dt_update_s = None
            if state.t_prev_meas_s is not None:
                dt_update_s = float(t_meas) - state.t_prev_meas_s
            state.t_prev_meas_s = float(t_meas)

            state.update_index += 1
            meas_quality_ok = base._phase_quality_ok_for_ctrl(params, meas)
            base.print_status(params, state, meas)
            base.log_row(
                logger,
                params,
                state,
                meas,
                args.notes,
                dt_update_s,
                dac_step_result,
                meas_quality_ok,
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Closing SDR/I2C handles (DAC left unchanged).",
            flush=True,
        )
    except Exception as exc:
        print("\nERROR: {}".format(exc), file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await stop_stream_task(stream_task)
        try:
            logger.close()
        except Exception:
            pass
        base.sdr_close(sdr)
        if clock_obj is not None:
            base.clock_close(clock_obj)

    return 0


async def run_async_helper_self_tests() -> None:
    q: asyncio.Queue[QueuedBlock | StreamSentinel] = asyncio.Queue(maxsize=2)
    b0 = np.array([1.0 + 2.0j], dtype=np.complex64)
    b1 = np.array([3.0 + 4.0j], dtype=np.complex64)
    await q.put(QueuedBlock(unix_time_s=1.0, iq_block=b0))
    await q.put(QueuedBlock(unix_time_s=2.0, iq_block=b1))
    blocks, t_meas = await read_update_blocks(q, 2)
    assert len(blocks) == 2
    assert blocks[0] is b0
    assert blocks[1] is b1
    assert t_meas == 2.0

    q2: asyncio.Queue[QueuedBlock | StreamSentinel] = asyncio.Queue()
    await q2.put(StreamSentinel())
    blocks, t_meas = await read_update_blocks(q2, 2)
    assert blocks == []
    assert t_meas is None


def run_self_tests() -> int:
    status = base.run_self_tests()
    if status != 0:
        return status

    raw = bytes([0, 255, 128, 128])
    iq = bytes_to_iq_block(raw, 2)
    assert iq.shape == (2,)
    assert iq[0] == np.complex64(-128.0 + 127.0j)
    assert iq[1] == np.complex64(0.0 + 0.0j)

    assert default_async_queue_blocks(1) == 8
    assert default_async_queue_blocks(5) == 20

    asyncio.run(run_async_helper_self_tests())
    print("Async stream self-tests passed.")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.run_self_tests:
        return run_self_tests()
    return asyncio.run(run_controller(args))


if __name__ == "__main__":
    sys.exit(main())
