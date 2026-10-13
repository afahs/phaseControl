# Project instructions

Read `PI_HANDOFF.md` before working with the Raspberry Pi or deployment.

- Develop and commit changes in this local repository, then deploy with Git.
- Use SSH alias `project-pi` and the existing dedicated key.
- Keep password-based SSH login enabled on the Pi: the owner explicitly requested fallback access.
- Keep private keys, passwords, and runtime secrets out of Git.
- Application: Python package `beacon_clock`; see README.md and docs/design.md.
- Keep acquisition/control free of disk I/O. Save bounded binary telemetry, never CSV.
- Treat the Pi's legacy phaseControl experiments as unverified evidence, not a trusted control design.
- Keep monitor receive-only. Feedback requires a compatible measured calibration and verified clock configuration.
- Do not enable a startup service until interactive hardware validation has passed.
