Inside the `project/code/` directory, create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, your shell prompt should usually show `(.venv)`.

## Install dependencies

With the virtual environment activated, install all required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the simulation

Run the script from `code/` directory:

```bash
python school_abm.py
```

By default, the script runs the configured number of repeated simulations and writes outputs into the `code/` directory:

- `code/results.csv`: per-run time-series counts for each misinformation state.
- `code/results_summary.csv`: aggregated summary statistics across runs.
- `code/results.png`: line plot of state counts over time.
- `code/network_spread.png`: network snapshot plot showing misinformation spread over time.

## Useful command-line options

You can override the defaults with command-line options:

```bash
python code/school_abm.py \
  --runs 10 \
  --base-seed 42 \
  --results-path outputs/results.csv \
  --summary-path outputs/results_summary.csv \
  --plot-path outputs/results.png \
  --network-plot-path outputs/network_spread.png \
  --network-snapshots 5
```

Common options:

- `--runs`: number of independent simulation runs.
- `--base-seed`: base random seed; each run uses `base_seed + run_id`.
- `--no-seed`: disable deterministic seeding.
- `--results-path`: where to write per-run CSV output.
- `--summary-path`: where to write aggregated CSV output.
- `--plot-path`: where to write the time-series plot.
- `--network-plot-path`: where to write the network snapshot plot.
- `--network-snapshots`: number of uniformly spaced network snapshots to plot; must be between 2 and 5.
- `--show`: display plot windows in addition to saving plot files.

If you write outputs to a new directory such as `outputs/`, create it first:

```bash
mkdir -p outputs
```

On Windows PowerShell, use:

```powershell
New-Item -ItemType Directory -Force outputs
```

## Deactivate the virtual environment

When finished, deactivate the virtual environment:

```bash
deactivate
```
