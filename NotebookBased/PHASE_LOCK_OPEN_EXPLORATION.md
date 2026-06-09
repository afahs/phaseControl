# Phase Lock Open Exploration

This note is intentionally less prescriptive than [PHASE_LOCK_AGENT_NOTES.md](/home/rrrodriguez13/phaseControl/NotebookBased/PHASE_LOCK_AGENT_NOTES.md).

Its purpose is to help a future agent explore the design space without being overly anchored to the current controller structure.

The right mindset for this file is:

- preserve evidence
- avoid treating current defaults as final truth
- separate observations from interpretations
- keep architectural alternatives alive

## What Is Actually Known

These are the things that seem genuinely well supported by the experiments so far.

### 1. Simulation and live hardware do not behave the same

This is the biggest fact.

- In simulation, phase mode can work very well when started in a favorable region.
- On live hardware, raw phase measurements are much less clean and much less repeatable.

So any conclusion drawn from simulation alone should be treated as a hypothesis about live behavior, not proof.

### 2. The controller can affect the live system causally

This is supported by the sham-control test:

- real closed loop produced better phase-lock metrics than sham randomized control
- therefore the loop is not only “winning by filtering”

That does **not** prove the loop is fully solving the right problem, but it does show that actuation matters.

### 3. The best live results so far came after improving the measurement path

The biggest improvements came from things like:

- chunked phase estimation
- fit-quality metrics
- innovation gating
- using the `adaptive` path

This suggests the limiting issue may be measurement quality or measurement trust, not only loop gain.

### 4. Raw phase is still not convincingly centered on most live blocks

This remains true even in the stronger runs.

So the current best interpretation is not “problem solved,” but “some useful closed-loop behavior exists, while raw phase lock remains questionable or partial.”

## What Is Not Yet Proven

These are common interpretations that should **not** be treated as settled.

### 1. Pull-in is definitely required forever

The current evidence suggests pull-in helps a lot on live hardware, but this is not the same as proving that a different one-stage design could never work.

A true single-stage design might still succeed if:

- the phase detector is redesigned
- the phase controller has a larger capture range
- the loop uses a different measurement schedule
- the controller contains built-in frequency-centering behavior

So “pull-in is necessary” should be treated as:

- true for the current phase law
- not yet proven for all possible one-stage designs

### 2. Single-stage phase locking is a dead end

This is not proven.

What has been shown is narrower:

- the tested single-stage implementations did not perform well enough on live hardware
- they did perform better in simulation when starting conditions were favorable

That leaves open several possibilities:

- the one-stage idea is good, but the tested controller was too weak
- the one-stage idea is good, but the measurement path was not yet good enough
- one-stage could work if it blends coarse frequency correction and fine phase correction in one law

### 3. The current handoff logic is close to optimal

Not proven.

The current handoff logic is simply the most recent workable compromise.

Alternative handoff strategies are still valid to explore:

- purely residual-based handoff
- purely phase-state-based handoff
- time-based trust accumulation
- no explicit handoff at all

### 4. Current defaults are the “best” settings

Not proven.

The current defaults are just the latest reverted operating point after several experiments. They are a useful baseline, not a final answer.

## A Less Anchored Interpretation Of The Project

A future agent should think of the project as:

“Build a live SDR clock/phase control loop that can both acquire and hold phase alignment robustly under noisy, delayed, real-world measurements.”

That leaves open multiple architectures:

- staged pull-in then phase hold
- one-stage hybrid controller
- adaptive controller that smoothly changes behavior with confidence
- controller that changes effective bandwidth or block size during operation

## Alternative Hypotheses Still Worth Exploring

These are all still live possibilities.

### Hypothesis A: One-stage can work if the control law includes both phase and frequency terms

This means not “phase-only startup” in the naive sense, but one continuous controller that:

- acts like coarse frequency centering when far away
- acts like phase centering when close

In other words, one stage does not have to mean one linear law.

### Hypothesis B: One-stage failed mostly because the measurement was not ready yet

Earlier single-stage tests happened before some of the best measurement-path changes were in place.

So a future agent could reasonably retry one-stage with:

- chunk-fit measurement
- fit residual gating
- adaptive path
- post-DAC ignore logic

That would be a more fair test than simply replaying the earliest single-stage version.

### Hypothesis C: The main barrier is not controller structure but confidence scheduling

The system may need to adapt based on trust:

- when measurement confidence is low, act gently
- when confidence is high, act strongly

That idea could apply equally well to:

- one-stage control
- staged control
- hybrid control

### Hypothesis D: The main remaining problem is raw phase sensing, not phase control

This is one of the strongest current hypotheses, but it is still a hypothesis.

If true, then major future gains will come from:

- better detector design
- better trust metrics
- better use of accepted vs rejected measurements

If false, then a new controller law could still solve much of the problem even with the present detector.

## Architectures Worth Revisiting

These should not be considered closed.

### 1. True single-stage hybrid controller

A future agent could revisit this with a more deliberate design:

- use one controller from startup
- include residual frequency centering directly in the control law
- gradually increase phase authority as trust improves

This avoids brittle handoff logic entirely.

### 2. Single-stage with gain scheduling

Instead of separate modes, use one controller whose gains depend on:

- residual frequency magnitude
- fit RMS
- innovation size
- coherence or local consistency

This could achieve most of the benefit of a staged loop without a hard switch.

### 3. Variable `nsamp` or effective windowing

The system already showed that measurement windowing matters.

It may still be worth exploring:

- large windows for coarse acquisition
- smaller windows for near-lock
- or continuous chunk-based midpoint estimation without explicit mode switching

### 4. More aggressive phase authority under strong confidence

A future agent could explore:

- larger max DAC step in phase mode
- larger phase authority only when fit residual is low
- stronger centering when the accepted phase state is consistent

### 5. Alternative phase detectors

The current chunk-fit detector helped, but it should not be treated as the only viable option.

Still-open ideas:

- midpoint phase from chunk regression
- weighted chunk fusion
- local phase unwrap over chunks
- detectors that explicitly model intrablock slope and curvature

## Experimental Principles For Future Agents

To stay open-minded while still being rigorous:

### 1. A failed architecture test is not a proof of impossibility

If a one-stage test failed, record:

- which detector was used
- what startup region was tested
- what gains and step limits were used
- what trust logic existed

Then frame the conclusion narrowly:

- “this implementation failed”

not:

- “this entire architectural idea is impossible”

### 2. Separate “works in simulation” from “works live”

Simulation is useful for:

- logic checks
- pull-in range studies
- sensitivity exploration

But live runs remain the final arbiter.

### 3. Compare against strong baselines, not only against the current experiment

Good reference runs are:

- [20260406_001120_live_measurement_upgrade_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s)
- [20260406_004100_live_kd0p20_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s)

A new idea should be compared to those, not only to the immediately previous test.

### 4. Keep track of what metric is being improved

Different changes can improve different things:

- drift reduction
- filtered phase error
- raw accepted phase error
- dwell time
- phase-band fraction

Do not assume improvement in one means improvement in the others.

## A Good Workflow For Re-Exploration

If a future agent wants to reopen the design space, a good process would be:

1. Re-run a strong current baseline live.
2. Choose one architecture to reopen.
3. Define the narrow question.
4. Test in simulation first if useful.
5. Then test on live hardware quickly.
6. Compare against the strong live baseline.

Example narrow questions:

- Can a one-stage hybrid law beat staged control on filtered phase metrics?
- Can a one-stage law avoid handoff chatter while matching drift cleanup?
- Can a redesigned phase detector make one-stage viable?

## Suggested Future Questions

These are intentionally open-ended on architecture.

### Question 1

Can a one-stage hybrid controller outperform the current staged controller if it uses:

- adaptive measurement
- chunk-fit detector
- confidence-gated authority

### Question 2

Can the live detector be redesigned so that raw phase behaves more like the accepted filtered phase?

### Question 3

Can we make control authority conditional on confidence so that the loop is gentle when uncertain and strong when trustworthy?

### Question 4

Can variable measurement windows or chunk-based midpoint estimators remove the need for a hard pull-in to phase handoff?

## If You Are A Future Agent

Do not read the current wrapper defaults as a commandment.

Read them as:

- the latest stable-ish baseline
- one operating point that has shown useful live behavior

You are allowed to challenge:

- the staged architecture
- the handoff logic
- the meaning of “phase mode”
- whether one-stage should be revisited
- whether the detector should be redesigned first

But when you challenge those, do it with:

- a clear hypothesis
- a clear baseline
- a clean comparison

The goal is not to preserve the current design. The goal is to find the simplest design that works well on live hardware.
