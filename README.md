# Federated and Cluster-Aware Wi-Fi Fingerprinting for Indoor Localisation

This repository contains the experimental notebooks and generated outputs used to compare **centralised**, **distributed/federated**, and **cluster-aware** learning strategies for Wi-Fi fingerprinting-based indoor localisation.

The experiments cover four public datasets and three model families:

- **Datasets:** TUT, TUJI1, UJIIndoorLoc, and SODIndoorLoc.
- **Models:** K-Nearest Neighbours (KNN), Multi-Layer Perceptron (MLP), and eXtreme Gradient Boosting (XGBoost).
- **Strategies:** centralised global, federated global, centralised cluster-aware, and federated cluster-aware.
- **Tasks:** two-dimensional coordinate estimation and, when the selected dataset contains more than one floor, floor classification.

The federated experiments are **simulated on a single server**. Clients are logical partitions defined mainly from device or phone identifiers; the repository does not implement communication over a physical federated network.

## Repository structure

```text
federated-cluster-aware-indoor-localisation/
├── TUT/
│   ├── 00_TUT_preparar_y_clusterizar.ipynb
│   ├── TUT_KNN.ipynb
│   ├── TUT_MLP.ipynb
│   ├── TUT_XGBoost.ipynb
│   └── source CSV files
├── TUJI1/
│   ├── 00_TUJI1_preparar_y_clusterizar.ipynb
│   ├── TUJI1_KNN.ipynb
│   ├── TUJI1_MLP.ipynb
│   ├── TUJI1_XGBoost.ipynb
│   └── source CSV files
├── UJIIndoor/
│   ├── 00_UJIIndoor_preparar_y_clusterizar.ipynb
│   ├── UJIIndoor_KNN.ipynb
│   ├── UJIIndoor_MLP.ipynb
│   ├── UJIIndoor_XGBoost.ipynb
│   └── source CSV files
├── SOD/
│   ├── 00_SOD_preparar_y_clusterizar.ipynb
│   ├── SOD_KNN.ipynb
│   ├── SOD_MLP.ipynb
│   ├── SOD_XGBoost.ipynb
│   └── source CSV files
├── prepared/
│   ├── TUT/
│   ├── TUJI1/
│   ├── UJIIndoor/
│   └── SOD/
├── results/
│   ├── TUT/
│   ├── TUJI1/
│   ├── UJIIndoor/
│   └── SOD/
└── README.md
```

Each dataset folder contains one preparation notebook and three model notebooks. The `prepared/` directory stores the generated training, validation, test, routing, and diagnostic files. The `results/` directory stores the final metrics, tuning tables, and selected hyperparameters.

## Dataset provenance

The repository snapshot already contains the CSV files required by the notebooks. The links below identify the original public sources and should be used when citing or downloading the datasets independently.

| Dataset | Original source | Files used in this repository | Experimental subset |
|---|---|---|---|
| **TUT** | [Zenodo dataset](https://doi.org/10.5281/zenodo.1001662) · [Data paper](https://doi.org/10.3390/data2040032) | `TUT/Training_dataset.csv`, `TUT/Testing_dataset.csv` | Original files are merged and restricted to the five most represented devices. Samples are partitioned by physical position. |
| **TUJI1** | [Zenodo dataset](https://doi.org/10.5281/zenodo.7641701) · [Data paper](https://doi.org/10.1016/j.dib.2024.110356) | `TUJI1/training_data_complete.csv`, `TUJI1/testing_data_complete.csv`, `TUJI1/DEVICE_LABEL.csv` | The official test partition is preserved. Validation is extracted only from the original training data. |
| **UJIIndoorLoc** | [UCI Machine Learning Repository](https://doi.org/10.24432/C5MS59) | `UJIIndoor/TrainingData.csv`, `UJIIndoor/ValidationData.csv` | Building 1, floors 0 and 1, and phone identifiers 6, 7, 13, 14, and 17. The current notebook filters the external test data to devices observed during development. |
| **SODIndoorLoc** | [Official GitHub repository](https://github.com/bijingxue/SODIndoorLoc) · [Data paper](https://doi.org/10.1186/s43020-022-00086-y) | HCXY files under `SOD/` | Only the HCXY building is used. Final model notebooks evaluate the RAW representation containing all repeated RSSI acquisitions. |

The datasets remain subject to the terms and attribution requirements of their original publishers. This repository does not relicense third-party data.

## Current prepared partitions

Running the preparation notebooks with the configuration included in this repository produces the following partitions:

| Dataset configuration | Training | Validation | Test | Main task(s) |
|---|---:|---:|---:|---|
| TUT top-five devices | 1,868 | 409 | 400 | Coordinates and floor |
| TUJI1 official protocol | 5,740 | 1,012 | 2,147 | Coordinates |
| UJIIndoorLoc Building 1, floors 0--1 | 2,489 | 363 | 67 | Coordinates and floor |
| SODIndoorLoc HCXY RAW | 9,660 | 1,710 | 860 | Coordinates |

For SODIndoorLoc, the partition also corresponds to 322 training positions, 57 validation positions, and 86 external test positions. Repeated acquisitions at each position are retained in the RAW representation.

> **Important:** the current UJIIndoorLoc notebooks and result files correspond to the 67-sample external test subset produced after filtering to known selected phone identifiers. Any thesis table based on a different UJIIndoorLoc test size must be synchronised with the repository before publication.

## Requirements

A recent Python 3 environment is recommended. The notebooks use the following main packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `pyproj`
- `xgboost`
- `torch`
- `jupyterlab`
- `ipython`

Create and activate a virtual environment from the repository root:

### Linux, macOS, or WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas scikit-learn pyproj xgboost torch ipython
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas scikit-learn pyproj xgboost torch ipython
```

For an NVIDIA GPU, install the PyTorch build compatible with the local CUDA version by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Computational resources

- **MLP:** configured to use CUDA GPU acceleration when CUDA is available, with an automatic CPU fallback.
- **KNN:** executed on the CPU and uses parallel neighbour searches where supported.
- **XGBoost:** executed on the CPU with the histogram tree method.

The presence of federated strategies in the notebooks does not imply a physically distributed deployment. All clients and coordination procedures are simulated in the same execution environment.

## How to run the experiments

### 1. Start Jupyter from the repository root

```bash
cd federated-cluster-aware-indoor-localisation
jupyter lab
```

Starting Jupyter from the root directory is recommended because the notebooks automatically search the current directory and its parent directories for the dataset folders. If automatic detection fails, set `PROJECT_ROOT` in the first configuration cell of the notebook:

```python
from pathlib import Path
PROJECT_ROOT = Path("/absolute/path/to/federated-cluster-aware-indoor-localisation")
```

### 2. Prepare a dataset

Run the corresponding `00_*_preparar_y_clusterizar.ipynb` notebook from top to bottom:

| Dataset | Preparation notebook | Generated prefix |
|---|---|---|
| TUT | `TUT/00_TUT_preparar_y_clusterizar.ipynb` | `tut_top5` |
| TUJI1 | `TUJI1/00_TUJI1_preparar_y_clusterizar.ipynb` | `tuji1_official` |
| UJIIndoorLoc | `UJIIndoor/00_UJIIndoor_preparar_y_clusterizar.ipynb` | `uji_b1_f01` |
| SODIndoorLoc | `SOD/00_SOD_preparar_y_clusterizar.ipynb` | `sod_raw` and `sod_avg` |

The preparation notebooks perform the following operations:

1. Load and normalise the source CSV files.
2. Apply the dataset-specific subset and partitioning rules.
3. Prevent the same physical position from appearing in multiple development partitions.
4. Fit K-means spatial communities using training coordinates.
5. Train an RSSI-based router for every candidate number of communities.
6. Save the prepared partitions, routes, diagnostics, and preparation report under `prepared/<DATASET>/`.

The prepared files are already included in the repository, so this stage can be skipped when only reproducing the model execution from the existing partitions. Run it when reproducing the complete data-preparation process or after replacing the source data.

### 3. Run the model notebooks

Once the prepared files exist, run any model notebook from top to bottom:

| Dataset | KNN | MLP | XGBoost |
|---|---|---|---|
| TUT | `TUT/TUT_KNN.ipynb` | `TUT/TUT_MLP.ipynb` | `TUT/TUT_XGBoost.ipynb` |
| TUJI1 | `TUJI1/TUJI1_KNN.ipynb` | `TUJI1/TUJI1_MLP.ipynb` | `TUJI1/TUJI1_XGBoost.ipynb` |
| UJIIndoorLoc | `UJIIndoor/UJIIndoor_KNN.ipynb` | `UJIIndoor/UJIIndoor_MLP.ipynb` | `UJIIndoor/UJIIndoor_XGBoost.ipynb` |
| SODIndoorLoc | `SOD/SOD_KNN.ipynb` | `SOD/SOD_MLP.ipynb` | `SOD/SOD_XGBoost.ipynb` |

Each model notebook:

1. Loads the prepared training, validation, test, and routing files.
2. Audits the partition and available clients.
3. Selects global hyperparameters using validation data.
4. Selects the number of spatial communities using validation data.
5. Trains and evaluates the four strategies.
6. Displays validation and test metrics.
7. Writes the final outputs to `results/<DATASET>/`.

The test partition is evaluated only after configuration selection. Model notebooks may overwrite result files with the same names, so preserve a copy before running modified configurations.

### Optional command-line execution

A notebook can also be executed non-interactively from the repository root:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=-1 \
  TUT/00_TUT_preparar_y_clusterizar.ipynb
```

Then execute a model notebook in the same way:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=-1 \
  TUT/TUT_MLP.ipynb
```

Replace the path with the required dataset and model notebook.

## Prepared files

The preparation stage generates files following this pattern:

```text
prepared/<DATASET>/
├── <prefix>_train.csv
├── <prefix>_val.csv
├── <prefix>_test.csv
├── <prefix>_routes.csv
├── <prefix>_routing_diagnostics.csv
└── <prefix>_preparation_report.json
```

The main contents are:

- `*_train.csv`, `*_val.csv`, and `*_test.csv`: prepared RSSI features, targets, client identifiers, and metadata.
- `*_routes.csv`: predicted community assignments and routing information for every evaluated value of `K`.
- `*_routing_diagnostics.csv`: validation and diagnostic test performance of the RSSI router.
- `*_preparation_report.json`: partition sizes, position overlap checks, clients, and preparation settings.

## Result files

The model notebooks create files following these conventions:

```text
results/<DATASET>/
├── *_corrected.csv
├── *_corrected_tuning.csv
├── *_corrected_hyperparameter_tuning.csv
├── *_corrected_cluster_tuning.csv
└── *_corrected_selected_hyperparameters.json
```

Depending on the model, not every tuning suffix is used. In general:

- `*_corrected.csv`: final validation and test metrics for the selected configurations.
- `*_tuning.csv` or `*_hyperparameter_tuning.csv`: validation results for global model candidates.
- `*_cluster_tuning.csv`: validation results used to select the number of communities.
- `*_selected_hyperparameters.json`: final global and cluster-aware selections.

For TUT and UJIIndoorLoc, separate result files are generated for coordinate estimation and floor classification. TUJI1 and the selected SODIndoorLoc subset contain one floor, so only coordinate estimation is evaluated.

## Experimental design

### Centralised global

One model is trained using all training fingerprints from the selected dataset subset.

### Federated global

Training data remain divided into simulated clients. MLP uses Federated Averaging, XGBoost uses cyclic boosting across clients, and KNN performs an exact distributed neighbour search by merging local candidate neighbours.

### Centralised cluster-aware

Training positions are divided into K-means spatial communities. An RSSI-based router assigns validation and test fingerprints to a community, and the corresponding specialised centralised model performs localisation.

### Federated cluster-aware

A separate distributed model is trained for each spatial community using the clients represented in that community. The same RSSI-based routing mechanism selects the specialised model during inference.

## Reproducibility notes

- The main random seed is `42`.
- Hyperparameters and the number of communities are selected using validation data.
- Spatial communities are fitted using training coordinates only.
- True test coordinates are not used to choose a community during inference.
- Router test accuracy is retained for diagnostic reporting and is not used for model selection.
- UJIIndoorLoc coordinates are transformed to a local metric representation during preparation.
- Final SODIndoorLoc model notebooks use the RAW representation. AVG files are retained only as additional preparation outputs and are not part of the final reported comparison.
- Small numerical differences may occur across library versions, hardware, thread scheduling, or GPU implementations.

## Dataset citations

When using this repository, cite the original datasets in addition to the thesis or publication associated with this code.

### TUT

> E.-S. Lohan, J. Torres-Sospedra, H. Leppäkoski, P. Richter, Z. Peng, and J. Huerta, “Wi-Fi Crowdsourced Fingerprinting Dataset for Indoor Positioning,” *Data*, vol. 2, no. 4, 2017. https://doi.org/10.3390/data2040032

### TUJI1

> L. Klus, R. Klus, E. S. Lohan, J. Nurmi, C. Granell, M. Valkama, J. Talvitie, S. Casteleyn, and J. Torres-Sospedra, “TUJI1 Dataset: Multi-device dataset for indoor localization with high measurement density,” *Data in Brief*, vol. 54, 110356, 2024. https://doi.org/10.1016/j.dib.2024.110356

### UJIIndoorLoc

> J. Torres-Sospedra, R. Montoliu, A. Martínez-Usó, T. Arnau, and J. Avariento, “UJIIndoorLoc,” UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5MS59

### SODIndoorLoc

> J. Bi, Y. Wang, B. Yu, et al., “Supplementary open dataset for WiFi indoor localization based on received signal strength,” *Satellite Navigation*, vol. 3, article 25, 2022. https://doi.org/10.1186/s43020-022-00086-y
