import json
from pathlib import Path
import numpy as np
from .telemetry import read_run, VALID, GAP, SLIP, STALE


def clean_segments(records):
    """Never bridge record loss, time gaps, estimator resets, or invalid data."""
    if not len(records):
        return []
    flags = records["flags"]
    good = ((flags & VALID) != 0) & ((flags & (GAP | SLIP | STALE)) == 0)
    differences = np.diff(records["t"])
    positive = differences[differences > 0]
    cadence = float(np.median(positive)) if len(positive) else 1.
    breaks = np.flatnonzero((np.diff(records["record"]) != 1)
                           | (np.diff(records["segment"]) != 0)
                           | (differences <= 0) | (differences > cadence * 1.5)) + 1
    chunks = np.split(np.arange(len(records)), breaks)
    result = []
    for chunk in chunks:
        valid = good[chunk]
        edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= 3:
                result.append(chunk[a:b])
    return result


def analyze(path, plots=False):
    metadata, rows = read_run(path)
    segments = clean_segments(rows)
    pilot = metadata["config"]["pilot_hz"]
    result = {"records": len(rows), "continuous_valid_segments": len(segments),
              "telemetry": metadata["telemetry"], "summary": metadata.get("summary", {}),
              "scope": "Combined pilot/reference/path/receiver behavior; not absolute timing accuracy",
              "allan": [], "coherence": []}
    good = rows[(rows["flags"] & VALID) != 0]
    if len(good):
        result.update(frequency_mean_hz=float(np.mean(good["frequency"])),
                      frequency_std_hz=float(np.std(good["frequency"])),
                      dsp_p99_s=float(np.percentile(good["dsp_s"], 99)),
                      age_p99_s=float(np.percentile(good["age_s"], 99)))
    # Select truly uniform subseries without interpolation. 10 Hz scheduling over
    # 8 ms blocks alternates 96/104 ms, so every other point has a 200 ms cadence.
    # The first estimate after reset has a different FIR midpoint; trim it.
    uniform = []
    for indexes in segments:
        for stride in range(1, 17):
            s = rows[indexes[2::stride]]
            if len(s) < 3:
                break
            delta = np.diff(s["t"])
            dt = float(np.mean(delta))
            if np.max(abs(delta-dt)) <= max(1e-8, dt*1e-5):
                uniform.append((s, dt, stride))
                break
    result["allan_uniform_segments"] = len(uniform)
    result["allan_cadence_method"] = "uniform strided subseries; no interpolation or drift removal"
    result["phase_excursions"] = dict(
        segments=len(segments),
        longest_segment_s=max((float(rows[ix[-1]]["t"]-rows[ix[0]]["t"]) for ix in segments), default=0.),
        max_within_segment_peak_to_peak_rad=max((float(np.ptp(rows[ix]["phase"])) for ix in segments), default=0.))
    buckets = {}
    for factor in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        for s, dt, stride in uniform:
            if len(s) < 2 * factor + 1:
                continue
            tau = factor * dt
            x = s["phase"] / (2 * np.pi * pilot)
            dd = x[2*factor:] - 2*x[factor:-factor] + x[:-2*factor]
            # Never combine different averaging times into a fictitious mean tau.
            bucket = buckets.setdefault(round(tau, 8), dict(total=0., count=0, coherence=[]))
            bucket["total"] += float(np.sum(dd*dd / (2*tau*tau)))
            bucket["count"] += len(dd)
            for start in range(0, len(s) - factor, factor):
                phase = s["phase"][start:start+factor+1]
                # Integrate exp(i*phase) along linear phase intervals. Unlike an
                # average of sparse phasors this includes measured phase winding.
                phasor = np.exp(.5j*(phase[:-1]+phase[1:])) * np.sinc(np.diff(phase)/(2*np.pi))
                bucket["coherence"].append(float(abs(np.mean(phasor))))
    for tau, bucket in sorted(buckets.items()):
        result["allan"].append(dict(tau_s=tau, deviation=float(np.sqrt(bucket["total"]/bucket["count"])),
                                    terms=bucket["count"]))
        result["coherence"].append(dict(integration_s=tau,
                                        pilot_coherence=float(np.mean(bucket["coherence"])),
                                        model="piecewise-linear recorded phase; faster noise unresolved"))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
        for indexes in segments:
            s = rows[indexes]
            axes[0].plot(s["t"], s["phase"]-s["phase"][0],
                         linewidth=.6, color="tab:purple")
            axes[1].plot(s["t"], s["frequency"], linewidth=.6, color="tab:blue")
        fig.suptitle("Combined reference/path/receiver/clock measurements\nPhase origin reset per continuous segment", fontsize=11)
        axes[0].set_ylabel("Relative phase deviation (rad)")
        axes[1].set_ylabel("Pilot frequency error (Hz)")
        axes[1].set_xlabel("Sample time (s)")
        if result["allan"]:
            axes[2].loglog([p["tau_s"] for p in result["allan"]],
                           [p["deviation"] for p in result["allan"]], "o-")
        axes[2].set_xlabel("Averaging time (s)")
        axes[2].set_ylabel("Overlapping ADEV")
        fig.savefig(Path(path) / "analysis.png", dpi=140)
        plt.close(fig)
    (Path(path) / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
