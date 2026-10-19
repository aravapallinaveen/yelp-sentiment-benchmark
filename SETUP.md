# Setup and running experiments

## Environment

Use a dedicated Python environment and a PyTorch build compatible with your hardware. `requirements.in` lists dependency candidates; it is not a tested lockfile. The recorded run used an NVIDIA GeForce RTX 4090. Training times and memory measurements depend on the hardware and software versions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.in
```

Installation has not been verified in a clean environment. Choose the appropriate PyTorch build for your CPU/CUDA configuration before training. After a successful setup, save resolved versions with `python -m pip freeze > requirements.lock.txt`.

## Data and notebooks

Obtain Yelp Polarity from its authorized provider. Datasets and checkpoint weights are not included. Keep datasets outside version control.

```powershell
cd notebooks
python -m jupyterlab
```

Open `benchmark_sentiment.ipynb`. Review setup and data paths before executing cells. The notebooks were adapted to the standalone layout and their old outputs were cleared; the edited copies have not been rerun end-to-end. Saved experiment metrics remain in `reports/metrics.csv`.

Use a fresh working copy or new output directories for experiments. Several cells write fixed filenames, so rerunning them can overwrite previous artifacts. Some cells require CUDA despite a CPU fallback elsewhere. First verify a small batch, finite loss, an optimizer step, and checkpoint save/reload before a full run.

## Project paths

- `notebooks/`: experiment notebooks.
- `outputs/`: generated samples, evaluation tables, and training histories.
- `reports/`: saved metrics and analysis.
- `data/` and `data_processed/`: local data created or read by preprocessing.
- `checkpoints/`: locally stored model weights.

## Current limitations

Training and evaluation were not rerun after the directory changes. The recorded metrics describe the existing experiments, not a new execution of these edited notebooks.

The stacked BiGRU full raw training log was overwritten during an accidental rerun. Its complete eight-epoch history survives; see [training log note](reports/training_log_note.md). The saved configs in configs/ describe the recorded run; notebook preprocessing writes new configs under data_processed/.
