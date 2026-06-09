`NotebookBased/runs/` is the home for new experiment sessions.

Recommended workflow:

```bash
python3 NotebookBased/run_reduced_nsamp_sweep.py
```

Each session is written to:

```text
NotebookBased/runs/<timestamp>_<label>/
  monitor/nsamp_<N>/
  pid/nsamp_<N>/
  comparison/
```

This keeps new SDR reruns out of the top-level `NotebookBased/` folder while preserving older flat artifacts in place.
