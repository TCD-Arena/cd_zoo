# Causal Discovery Zoo
This is a collection of Causal Discovery methods for time-series it is part of the TCD-Arena package.
While this repo can be used in a standalone manner, if you want to work in the TCD-arena framework, please refer to the main repository.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Methods Overview](#-causal-discovery-methods-overview)
- - [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Supported Data Formats](#-supported-data-formats)
- [Quick Start](#-quick-start)
- [General Workflow](#-general-workflow)
- [Survey: Installable Amortized Methods](#-survey-installable-amortized-causal-discovery-methods)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Detailed Method Descriptions](#-detailed-method-descriptions)
- [Bridge Scripts for Prior and Benchmark Datasets](#-bridge-scripts-for-prior-and-benchmark-datasets)

---

## 📋 Overview

This repository provides a unified framework for evaluating causal discovery algorithms. It focuses on **time-series** methods but also includes several **i.i.d.** (non-time-series) methods. It includes:

1. **Unified Benchmarking Framework**: Standardized evaluation pipeline for causal discovery methods
2. **29 Causal Discovery Methods**: Implementations and wrappers arround state-of-the-art algorithms to provide a consistent interface
3. **Flexible Data Formats**: Support for multiple input/output formats (split, joint, TCD-Arena)
4. **Hydra Configuration Management**: Easy experiment configuration and hyperparameter tuning
5. **Comprehensive Evaluation Tools**: Multiple scoring metrics (F1, precision, recall, SHD, ROC-AUC)
6. **Batch Processing**: Scripts for large-scale parallel experiments

**Key Features:**
- ✅ Supports both classical (VAR, VARLiNGAM, PC, LiNGAM, GES, DAGMA, LPCMCI, oCSE) and modern deep learning methods (AVICI, Causal Pretraining)
- ✅ Covers both **time-series** (VAR, PCMCI+, LPCMCI, DYNOTEARS, LCM, SDCI, oCSE, …) and **i.i.d.** (LiNGAM, PC, GES, DAGMA, AVICI, CDFM, Arrow, TabCausal) causal discovery
- ✅ Splits lagged and instantaneous causal relationships during evaluation
- ✅ Per-sample and aggregate evaluation metrics
- ✅ Easy integration of new methods via standardized interface
- ✅ Compatible with TCD-Arena benchmarking framework


### If you run this repo as a drop in replacement in TCD arena run: 

```bash
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam # Beware this runs ALL datasets. 
# You can restrict the data by either
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs)
#Or
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs) maximum_number_of_datasets
```

Additional functionality can be found below.


## 🎯 Current Causal Discovery Methods Overview

| Method | Script Method Name | Type | Deep Learning | Instantaneous | Window Graph | Summary Graph | Publication |
|:-------|:-------------------|:-----|:--------------|:--------------|:-------------|:-------------|:------------|
| **Direct Cross-Correlation** | direct_crosscorr | Heuristic | ❌ | ❌ | ✅ | ✅ | Simple Correlation |
| **VAR** | var | Granger | ❌ | ❌ | ✅ | ✅ | [Stock & Watson (2001)](https://www.princeton.edu/~mwatson/papers/Stock_Watson_JEP_2001.pdf) |
| **VAR-LINGAM** | varlingam | Granger + Noise | ❌ | ✅ | ✅ | ✅ | [Hyvärinen et al. (2010)](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1468-0084.2012.00710) |
| **DYNOTEARS** | dynotears | Score | ❌ | ✅ | ✅ | ✅ | [Pamfil et al. (2020)](https://arxiv.org/abs/2002.00498) |
| **PCMCI** | pcmci | Constraint | ❌ | ❌ | ✅ | ✅ | [Runge et al. (2019)](https://www.science.org/doi/10.1126/sciadv.aau4996) |
| **PCMCI+** | pcmciplus | Constraint | ❌ | ✅ | ✅ | ✅ | [Runge (2020)](https://arxiv.org/abs/2003.03685) |
| **FPCMCI**  | fpcmci | Constraint | ❌ | ✅  | ✅ | ✅ | [Castri et al.(2023)](https://proceedings.mlr.press/v213/castri23a.html) |
| **Causal Pretraining** | cp | Ammortized | ✅ | ❌ | ✅ | ✅ | [Stein et al. (2024)](https://arxiv.org/abs/2402.09305) |
| **LCM** | lcm | Ammortized | ✅ | ❌ | ✅ | ✅ | [Kougioulis et al. (2026)](https://arxiv.org/abs/2602.18662) |
| **SDCI** | sdci | Ammortized | ✅ | ❌ | ❌ | ✅ | [Balsells-Rodas et al. (2025)](https://arxiv.org/abs/2110.06257) |
| **NTS-NOTEARS** | ntsnotears | Score | ✅ | ✅ | ✅ | ✅ | [Sun et al. (2021)](https://arxiv.org/abs/2109.04286) |
| **SVAR-RFCI** | svarrfci | Constraint | ❌ | ✅ | ✅ | ✅ | [Colombo et al. (2012)](https://www.jmlr.org/papers/v13/colombo12a.html) |
| **LPCMCI** | lpcmci | Constraint | ❌ | ✅ | ✅ | ✅ | [Gerhardus & Runge (2020)](https://proceedings.neurips.cc/paper/2020/hash/d8c8e63e31935c9118141b1e4c7eff3b-Abstract.html) |
| **VARMA-LiNGAM** | varma_lingam | FCM | ✅ | ❌ | ❌ | ✅ | [Hyvärinen et al. (2010)](https://www.jmlr.org/papers/v11/hyvarinen10a.html) |
| **oCSE** | ocse | Info-theoretic | ❌ | ❌ | ❌ | ✅ | [Slote et al. (2025)](https://arxiv.org/abs/2501.13365) |
| **Cross-Correlation Peak** | crosscorr_peak | Heuristic | ❌ | ❌ | ❌ | ✅ | [Stein et al. (2025)](https://arxiv.org/abs/2503.17452) |
| **Physical Principle** | physical | Heuristic | ❌ | ❌ | ❌ | ✅ |  [Stein et al. (2025)](https://arxiv.org/abs/2503.17452) |
| **CrossCorr + Physical** | crosscorr_physical | Heuristic | ❌ | ❌ | ❌ | ✅ |  [Stein et al. (2025)](https://arxiv.org/abs/2503.17452) |
| **CDMI** | cdmi | Granger | ✅ | ❌ | ❌ | ✅ | [Ahmad et al. (2022)](https://arxiv.org/abs/2207.04055) |
| **SVAR-FCI** | svarfci | Constraint | ❌ | ✅ | ✅ | ✅ | [Malinsky & Spirtes (2018)](http://proceedings.mlr.press/v92/malinsky18a) |
| **DirectLiNGAM** *(i.i.d.)* | lingam | FCM (Noise) | ❌ | ❌ | ❌ | ✅ | [Shimizu et al. (2011)](https://www.jmlr.org/papers/v12/shimizu11a.html) |
| **PC** *(i.i.d.)* | pc | Constraint | ❌ | ❌ | ❌ | ✅ | [Spirtes et al. (2000)](http://www.mitpressjournals.org/doi/book/10.7551/mitpress/1754.001.0001) |
| **AVICI** *(i.i.d.)* | avici | Amortized | ✅ | ❌ | ❌ | ✅ | [Lorch et al. (2022)](https://arxiv.org/abs/2205.12934) |
| **CDFM** *(i.i.d.)* | cdfm | Amortized | ✅ | ❌ | ❌ | ✅ | [Qiao et al. (2026)](https://arxiv.org/abs/2607.11508) |
| **Arrow** *(i.i.d.)* | arrow | Amortized | ✅ | ❌ | ❌ | ✅ | [Thompson et al. (2026)](https://arxiv.org/abs/2605.07204) |
| **TabCausal** *(i.i.d.)* | tabcausal | Amortized | ✅ | ❌ | ❌ | ✅ | [Li et al. (2026)](https://huggingface.co/LAMDA-Tabular/TabCausal) |
| **GES** *(i.i.d.)* | ges | Score-based | ✅ | ❌ | ❌ | ✅ | [Chickering (2002)](https://www.jmlr.org/papers/v3/chickering02a.html) |
| **ICA-LiNGAM** *(i.i.d.)* | ica_lingam | FCM | ✅ | ❌ | ❌ | ✅ | [Shimizu et al. (2006)](https://www.jmlr.org/papers/v7/shimizu06a.html) |
| **DAGMA** *(i.i.d.)* | dagma | Continuous-opt | ✅ | ❌ | ❌ | ✅ | [Bello et al. (2022)](https://arxiv.org/abs/2209.08037) |

> **Note on i.i.d. methods.** `lingam`, `pc`, `ges`, `ica_lingam`, `dagma`, `avici`, `cdfm`, `arrow`, and `tabcausal` operate on **i.i.d.** (non-time-series) data: each sample is an `(n_vars × n_obs)` matrix and they return a single **summary graph** `[effect, cause]` (no lag dimension, no separate instantaneous channel). Unlike the time-series methods above, they are meaningfully applicable to i.i.d. benchmarks. **DirectLiNGAM** and **ICA-LiNGAM** require non-Gaussian noise for full DAG identifiability; **PC** and **GES** return a CPDAG (undirected edges where orientation is not identifiable); **DAGMA** is a continuous-optimization method (NOTEARS successor); **AVICI**, **CDFM**, **Arrow**, and **TabCausal** are pretrained amortized foundation models that run a single forward pass.
>
> **Note on amortized time-series methods (LCM, SDCI).** These are pretrained neural networks that map a multivariate time series directly to a causal graph in a single forward pass (no per-dataset fitting). **LCM** outputs a lagged adjacency `pred[effect, cause, lag]` (supports up to 12 variables, 3 lags, 500 timesteps); **SDCI** outputs a summary graph only `[effect, cause]` (the released springs *region* checkpoint was trained on 2-D particle simulations with 4 features per node, so 1-D benchmark series are expanded to 4 features via discrete derivatives — an out-of-distribution setting where absolute accuracy is expected to be modest).




## 📁 Repository Structure

```
full_cd_zoo/
├── benchmark.py                      # Main benchmarking script for TCD-Arena style evaluation
├── example_run.py                    # Example usage script for simple data formats
├── run_methods_on_causal_rivers.py   # Script for CausalRivers dataset (⚠️ Data not available.)
├── run_methods_on_causal_benchmarks.py # Script for the `causal-benchmarks` package (TS datasets, in-process)
├── run_var_sweep.py                  # Sweep VAR over all `causal-benchmarks` TS entries
├── run_method_on_prior.py            # Run any method on the `linear_new` prior grid (91 cells)
├── run_method_on_linear_test.py      # Run any method on the `linear_test` prior scenarios (6 cells)
├── simple_usage.ipynb                # Trivial tutorial notebook
├── methods/                          # Causal discovery method implementations
│   ├── var.py         # Classical time series methods
│   ├──...
├── config/                           # Hydra configuration files
│   ├── benchmark.yaml                # Main benchmark configuration
│   ├── example_run.yaml              # Simple run configuration
│   ├── method/                       # Method-specific configurations
│   ├── ci_test/                      # Conditional independence test configurations
│   └── data_preprocess/              # Data preprocessing configurations
├── tools/                            # Utility functions and scoring tools
│   ├── tools.py                      # Core utilities
│   ├── scoring_tools.py              # Evaluation metrics
│   ├── method_loader.py              # Dynamic method loading
│   └── helpers.py                    # Helper functions
├── sample_datasets/                  # Example datasets in different formats
│   ├── split_format/                 # Per-sample file format
│   └── joint_format/                 # Batch processing format
│   └── no_violation_small/                 # TCD-Arena format
├── envs/                             # Conda environment configuration files
│   ├── cd_zoo.yml                    # Main environment
│   ├── pcmci.yml, cdmi.yml, deep.yml # Method-specific environments
│   └── base.yml, dyno.yml            # Additional environments
├── scripts/                          # Scripts used throughout the TCD-Arena experiments
│   ├── execute_all_methods.sh        # Run all benchmark.py with all available methods on specified TCD-Arena Folder
│   ├── execute_all_for_causal_rivers.sh # Run methods on CausalRivers dataset
│   ├── execute_method_on_all_folders.sh # Apply single method across folders
│   └── execute_nl_ci_tests.sh        # Test nonlinear CI tests
```

## 🔧 Installation & Setup

### Basic Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TCD-Arena/cd_zoo
   cd cd_zoo
   ```

2. **Install the main environment**
   ```bash
   
   conda env create -f envs/cd_zoo_basic.yml
   conda activate cd_zoo_basic
   ```

3. **Install the torch environment (optional)**
   ```bash
   conda env create -f envs/cd_zoo_torch.yml
   conda activate cd_zoo_torch
   ```
### Method-Specific Environments
For FPCMCI, please follow the instructions and install on top of the base environment: https://github.com/lcastri/causalflow
For CDMI, we have a custom environment in envs/cd_zoo_cdmi.yaml
   

## 📊 Supported Data Formats

The framework supports three input formats for maximum flexibility:

### Format 1: Split Files (Per-Sample Processing)
**Structure**: Individual files for each sample
- **Time Series File**: CSV/numpy file containing time series data (time × variables)
- **Labels File**: NetworkX pickle or adjacency matrix with ground truth causal graph
- **Use Case**: Individual sample analysis, debugging, small-scale experiments
- **Example**: `sample_datasets/split_format/`

**Directory structure:**
```
split_format/
├── 0_data.csv
├── 0_label.csv
├── 1_data.csv
└── 1_label.csv
```

### Format 2: Joint Files (Batch Processing)
**Structure**: Combined files for multiple samples
- **Time Series CSV**: Contains data for all samples with sample identifier column
- **Labels Pickle**: List of NetworkX graph objects (one per sample)
- **Use Case**: Multi-sample processing, efficient batch evaluation
- **Example**: `sample_datasets/joint_format/`

**Files:**
```
joint_format/
├── data.csv      # All samples combined
└── labels.p   # List of NetworkX graphs
```

### Format 3: TCD-Arena Structure (Large-Scale Benchmarking)
**Structure**: Hierarchical folder organization for systematic evaluation
- **Directory hierarchy**: `main_folder/violation_type/data_regime_and_level/files`
- **Data structure**: Numpy arrays for efficient batch processing + config
- **Use Case**: Large-scale benchmarking across multiple data regimes and violation types
- **Example**: Used in `benchmark.py` script

**Directory structure:**
```
tcd_arena_format/
├── no_violation_small/           # violation+ size: e.g. Small graph structure, no violations
│   ├── 0/                        # Dataset instance 0 (a certain violation level + specific data regime)
│   │   ├── time_series.npy       # Shape: (n_samples, n_variables, n_timesteps)
│   │   ├── lagged_graphs.npy     # Shape: (n_samples, n_variables, n_variables, max_lag)
│   │   └── instant_graphs.npy    # Shape: (n_samples, n_variables, n_variables)
│   │   └── instant_graphs.npy    # Shape: (n_samples, n_variables, n_variables)
│   │   └── resample_statistics.npy    # See Synth-TS generator
│   │   └── config.yaml    # Parameters used to generate the sample (See Synth-TS-generator)


│   ├── 1/                        # Dataset instance 1
│   │   ├── time_series.npy
│   │   ├── lagged_graphs.npy
│   │   └── instant_graphs.npy
│   └── ...                       # Additional dataset instances
│
├── other_violation_size/           # Baseline: Large data regime, no violations
│   ├── 0/
│   ├── 1/
│   └── ...
```


## 🚀 Quick Start

### Three Ways to Get Started

#### 1. Trivial Tutorial
```bash
jupyter notebook simple_usage.ipynb
```
You can explore basic usage with guided examples in the notebook.

#### 2. Simple Example Run
```bash
# Run with custom method and dataset
python example_run.py \
    loading_mode="joint" \
    data_path="sample_datasets/joint_format/" \
    method=var \
    data_preprocess.normalize=False
```

Supported loading modes:
- `loading_mode="single"`: Load individual sample files
- `loading_mode="joint"`: Load batch files (CSV + pickle)

#### 3. TCD-Arena Style Benchmarking
```bash
# Run full benchmark on a dataset folder
python benchmark.py \
    method=var \
    data_path="/path/to/dataset/folder" \
    which_dataset=0
```

#### 4. Batch Processing with Scripts
```bash

# If you run this as a drop in replacement in TCD arena run: 
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam # Beware this runs ALL datasets. 
# You can restrict the data by either
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs)
#Or
./cd_zoo/scripts/run_all_methods_on_data_release.sh --method varlingam string_that_should_be_contained_in_violation_name (e.g. obs) maximum_number_of_datasets


# Run all methods on a dataset
./scripts/execute_all_methods.sh /path/to/dataset

# Run specific method on all folders
./scripts/execute_method_on_all_folders.sh var /path/to/datasets
```


## 📂 General Workflow

The benchmarking framework follows a systematic process:

### 1. **Data Loading & Preprocessing**
- Loads time series data in various formats (split, joint, or TCD-Arena structure)
- Applies optional preprocessing: normalization, standardization, detrending
- Validates data shape and handles missing values

### 2. **Method Selection & Configuration** 
- Dynamically imports the specified causal discovery method via Hydra config
- Loads method-specific hyperparameters from YAML configuration files
- Supports 29 methods ranging from classical (VAR, VARLiNGAM, GES, DAGMA, LPCMCI, oCSE) to amortized deep learning (LCM, SDCI, Causal Pretraining, AVICI, CDFM, Arrow, TabCausal)

### 3. **Causal Discovery Execution**
- Each method implements a unified interface: `run_method(data, config)`
- Input: Multivariate time series (variables × time)
- Output: 
  - Lagged causal graph (effect × cause × lag)
  - Instantaneous causal graph (effect × cause) *if method supports it*
- Tracks execution time for performance analysis
- Supports parallel processing across datasets

### 4. **Comprehensive Evaluation**
- Computes multiple metrics: F1-score, precision, recall, accuracy, SHD, ROC-AUC
- Evaluates at three granularities:
  - **Summary graph**: Aggregated causal relationships
  - **Window graph**: Lag-specific relationships
  - **Instantaneous graph**: Same-time-step relationships
- Optional: Removes autoregressive relationships for cleaner evaluation
- Supports both per-sample and aggregate metrics

### 5. **Results Export & Analysis**
- Saves predictions, ground truth, and metrics in a structured format
- Outputs runtime statistics for performance profiling
- Compatible with downstream analysis and visualization tools
- Results organized by timestamp and experiment configuration


### 6. CausalRivers Integration

The suite includes complete integration with the CausalRivers benchmark:

```bash

# Clone causalrivers
git clone https://github.com/CausalRivers/causalrivers

# Install CausalRivers environment
./install.sh
conda activate causalrivers

# Generate benchmark datasets
cd causalrivers
python 0_generate_datasets.py

# Transfer samples to TCD-Format by running: 
cd ..
python prepare_causal_rivers.py

# Execute methods: 
python run_methods_on_causal_rivery.py


# Run benchmarking
python benchmark.py method=var \
    data_path=product/rivers_ts_east_germany.csv \
    method.max_lag=5
```


### 7. Causal-Benchmarks Integration

The suite integrates with the [`causal-benchmarks`](https://github.com/) package
(a sibling thesis project exposing ~190 causal-discovery benchmark datasets with
ground-truth graphs through a single `CausalDataset` interface). Because
`causal-benchmarks` is pip-installable and returns ready-to-use objects
(`.data` DataFrame + `.true_graph` adjacency), **no preparation/export step is
needed** — data is loaded in-process and fed straight into `benchmarking()` /
`score()`:

```bash
# install the benchmark package (once)
pip install -e ../causal-benchmarks

# run one method on one benchmark entry
cd cd_zoo
python run_methods_on_causal_benchmarks.py dataset=var_sl method=var

# sweep methods × datasets with Hydra multirun
python run_methods_on_causal_benchmarks.py --multirun \
    dataset=netsim,var_sl,kuramoto,lorenz96 method=var,direct_crosscorr
```

Notes:
- cd_zoo methods are **time-series** methods; point the script at time-series
  entries (`causal_benchmarks.list_datasets()` with `domain == "timeseries"`).
- The script groups datasets by node count, so collections with variable *N*
  (e.g. `netsim`, `tcs`, `causal_audit`) are scored per group automatically.
- **Orientation**: `causal-benchmarks` stores `adj[i,j]=1` ⟺ *i causes j*
  (`[cause, effect]`); cd_zoo scores `[effect, cause]`, so the script transposes
  each ground truth before scoring (verified: VAR on `var_sl` → AUROC 1.0).


---

## §8. Survey: Installable Amortized Causal Discovery Methods

This section surveys the **amortized / end-to-end causal-discovery** literature
(as reviewed in the accompanying thesis, Ch. 02-06) and records, for each
method, whether it ships **publicly-installable code *and pretrained model
weights*** so that zero-shot inference can be run without retraining. This is
the relevant criterion for adding a method to a discovery *zoo*: training-only
repositories or papers-without-code cannot be evaluated out of the box.

A method is marked ✅ **installable** only if a user can run pretrained
inference in a single forward pass. 🟡 means pretrained weights exist but are
not pip-installable / non-Python / unconfirmed. ❌ means training-only or no
public code. Methods already integrated into cd_zoo are tagged
**(in cd_zoo)**.

### A. Ready-to-run (pip / git-install + pretrained weights)

| Method | Year | Task | Data | Install | Repo / weights |
|:-------|:----|:-----|:-----|:--------|:----------------|
| **AVICI** | 2022 | Structure | i.i.d. | `pip install avici` | [HF](https://huggingface.co/larslorch/avici) **(in cd_zoo)** |
| **Causal Pretraining** | 2024 | Structure | TS | repo `Gideon-Stein/CausalPretraining` | weights in-repo **(in cd_zoo as `cp`)** |
| **Arrow** | 2026 | Structure | i.i.d. | `pip install git+https://github.com/ryan-thompson/arrowfm` | [repo](https://github.com/ryan-thompson/arrowfm) (HF auto-fetch) **(in cd_zoo)** |
| **CDFM** | 2026 | Structure | i.i.d. | `pip install cdfm-base` | [repo](https://github.com/DMIRLAB-Group/CDFM) · [HF](https://huggingface.co/DMIRLAB/CDFM) **(in cd_zoo)** |
| **DCD-PFN** *(StablePFN)* | 2026 | Structure | i.i.d. | repo `GuanZhengkang/StablePFN` | `checkpoints/` + download |
| **FoundCause** | 2026 | Structure | i.i.d. | repo `amazon-science/foundcause` | pretrained weights (archived) |
| **TabCausal** | 2026 | Structure | i.i.d. | repo `LAMDA-Tabular/TabCausal` | `checkpoints/` · [HF](https://huggingface.co/LAMDA-Tabular/TabCausal) **(in cd_zoo)** |
| **TCD-DL** | 2025 | Structure (ancestors) | i.i.d. | repo `snu-mllab/Targeted-Cause-Discovery` | `ckpt/` + `inference.py` |
| **CauScale** | 2026 | Structure (≤1000 nodes) | i.i.d. | repo `OpenCausaLab/CauScale` | [HF](https://huggingface.co/OpenCausaLab) weights |
| **LCM** | 2026 | Structure | TS | repo `kougioulis/LCM` | pre-trained weights + `pip install` **(in cd_zoo)** |
| **SDCI** | 2025 | Structure | TS | repo `charlio23/SDCI` | model + scripts (ICML'25) **(in cd_zoo)** |
| **CausalPFN** | 2025 | **Effect** | i.i.d. | `pip install causalpfn` | [repo](https://github.com/vdblm/CausalPFN) · [HF](https://huggingface.co/vdblm/causalpfn) |
| **CC-PFN** | 2026 | **Effect** (continuous tx) | i.i.d. | repo `layer6ai-labs/CCPFN-inference` | weights + HF + pip |

> **Note on AVICI's data regime.** AVICI (Lorch et al., 2022) is an **i.i.d.**
> cross-sectional structure-discovery method (each dataset is an `(n × d)` sample
> matrix → DAG), *not* a time-series method, despite some secondary sources
> mislabeling it. It is already integrated in cd_zoo as the `avici` method.

### B. Pretrained weights exist, but not cleanly pip-installable

| Method | Year | Task | Data | Repo | Caveat |
|:-------|:----|:-----|:-----|:-----|:-------|
| **DAG-EQ** (Li) | 2020 | Structure | i.i.d. | `lihebi/DAG-EQ` | pre-trained `.bson` for d=10..400, but **Julia/Flux** (no pip) |
| **SEA** (Wu) | 2024 | Structure | i.i.d. | `rmwu/sea-reproduce` | 3 `.ckpt` + `inference.py`, but pinned `torch==1.13`/`lightning==1.9` (own env); shell/argparse API; runs classical algos on subsets |
| **SLdisco** (Petersen) | 2023 | Structure | i.i.d. | `annennenne/sldisco` | 21 `.h5` weights, but **no inference code** (weights + pre-computed GES outputs only); all models fixed **N=6** nodes; TF/Keras `.h5` |
| **do-PFN** | 2025 | Effect | i.i.d. | `jr2021/Do-PFN` | code + `.pkl` training data; pretrained weights unconfirmed |
| **Javurek CSA-PFN** | 2026 | Effect (sensitivity) | i.i.d. | HF `arunkatika/csa-pfn-msm` | weights on HF, no PyPI package |
| **CausalTimePrior** | 2026 | — (data generator) | TS | `thummd/CausalTimePrior` | prior/data sampler, not a pretrained model |

### C. Code released, but training-only (no pretrained weights)

| Method | Year | Task | Data | Repo | Note |
|:-------|:----|:-----|:-----|:-----|:-----|
| **ACD** (Löwe) | 2022 | Structure | TS | `loeweX/AmortizedCausalDiscovery` | training + data generators, no weights |
| **NRI** (Kipf) | 2018 | Structure | TS | `ethanfetaya/NRI` | training-only; needs PyTorch 0.2 / Py2.7 |
| **Lopez-Paz** | 2015 | Structure (pairwise) | i.i.d. | `lopezpaz/causation_learning_theory` | training code + LaTeX, no weights |
| **CausalFM** (Ma) | 2025 | Effect | i.i.d. | `yccm/CausalFM-toolkit` | `from_pretrained` expects local `.pth`; none distributed |

### D. No public code / code not found

The following reviewed methods had **no installable public release** at the time
of writing: **CSIvA** (Ke 2022), **Dhir 2025** (meta-learning Bayesian), **SiCL**
(Zhang 2025), **DAG-FM** (Chen 2026), **Sypniewski 2025**, **TabOrder** (Xu
2026), **CInA** (Zhang 2023), **Mahajan 2026**, **TabPFN-CFM** (Zhu 2026, GSK
industry), **Bellot 2026**, and **PTCD** (Ouyang 2026). **PTCD** is confirmed to
have **no public code or weights** — its only code link is an anonymous
peer-review repo (`anonymous.4open.science/r/PTCD-D4E3`); the paper was
"Submitted to" (not accepted at) NeurIPS 2026, and the prominent
`yiming-l21/PTCD` / `zjutangk/PTCD` GitHub repos are unrelated namesake
collisions. A few further candidates remain with *unverified* weight status
(Reuter `ArikReuter/Graphs4CausalFoundationModels`; MACE-TNP `ACEsuit/mace`;
Thumm PFN `thummd/continuous-time-causal-pfn`; ACTIVA/CIR-ACTIVA
`sa-and/cir-activa`).

### Takeaways

- **13** of the ~40 reviewed amortized methods are ready to run out of the box
  with pretrained weights; of these, **7 are already in cd_zoo** (AVICI, Causal
  Pretraining, LCM, SDCI, CDFM, Arrow, TabCausal) and **4 more are structure-discovery methods** directly relevant
  to the zoo (DCD-PFN, FoundCause, TCD-DL, CauScale).
- The recent (2025–2026) PFN wave has markedly improved code/weight releases
  compared with the early methods (Lopez-Paz 2015, NRI 2018, ACD 2022), which
  shipped training code only.
- The i.i.d. structure-discovery ecosystem is far richer than the time-series
  one: the installable **time-series** structure methods LCM and SDCI are now in
  cd_zoo (alongside the in-zoo Causal Pretraining). The time-series amortized
  field is effectively exhausted: every TS structure-discovery method with
  pretrained weights (CP, LCM, SDCI) is already integrated; the only remaining
  candidate, **PTCD** (Ouyang 2026, scales to 40 nodes, 0.85 AUROC on
  CausalRivers), has **no released code or weights** (anonymous peer-review repo
  only), and **ACD** (Löwe 2022) is training-only. A handful of i.i.d.
  structure methods with weights remain non-integrable for hard reasons: **DAG-EQ**
  (Julia/Flux), **SLdisco** (no inference code, fixed N=6), **SEA** (torch 1.13
  pin, own env).
- Effect-estimation methods (CausalPFN, CC-PFN) are pip-installable but
  target **treatment-effect** estimation, not graph structure, and so are not
  direct cd_zoo candidates.

---

## 🤝 Contributing

We welcome contributions to the Causal Discovery Zoo! Here's how you can help:

### 🔧 Development Status
- Full Release 1.0
- We are committed to adding more methods in the same manner in the future. 

### 🔧 Adding a New Method

1. Create method implementation in `methods/your_method.py`
2. Implement the standard interface: `def run_your_method(data_sample, cfg)`
3. Add a configuration file in `config/method/your_method.yaml`
4. Update method loader in `tools/method_loader.py.`
5. Add to the test script and validate good performance on the linear systems
6. Extend the documentation


## 📄 Citation

When you use specific methods, please also cite the original papers listed in the methods table above.
If you use the Causal Discovery Zoo in your research, please cite the TCD-Arena paper:


```bibtex
@inproceedings{
stein2026tcdarena,
title={{TCD}-Arena: Assessing Robustness of Time Series Causal Discovery Methods Against Assumption Violations},
author={Gideon Stein and Niklas Penzel and Tristan Piater and Joachim Denzler},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=MtdrOCLAGY}
}
```

---

## 📜 License

This project is part of the TCD-Arena release. Please refer to the main repository for licensing information.

---


# 📖 Detailed Method Descriptions

*The following sections provide detailed mathematical descriptions and implementation details for each causal discovery method.*


## Cross-correlation-Peak
Naive strategy that directs edges by looking for the lag that maximizes the cross-correlation between two time series:

$$  \underset{k \in {-k, \dots, k}}{\text{argmax}}  \hat{\rho}{X,Y}(k) = \frac{\frac{1}{n}\sum{t=1}^{n-|k|} (X_t - \bar{X})(Y_{t+k} - \bar{Y})}{s_X s_Y} ] $$

if $k$ is positive $X \rightarrow Y$ is infered. If k is negative $X \leftarrow Y$ is inferred.

Naive strategy. Has no guarantees to work.


**HPS**: 

    - max_lag: maximum lag that is evaluated in both directions.
    - filter_mode: As the method only directs each edge,an additional filter can be applied that, for each parent, selects the strongest correlation as the only child 


-----

## Physical principle
A simple var model is fitted to the time series process:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{A}_i \mathbf{Y}_{t-i} + \boldsymbol{\varepsilon}_t$$

Naive strategy that simply directs all edges via the mean of the time series: 

$ x_1 \rightarrow x_2 \text{ if } \bar{x}_1 > \bar{x}_2 \text{ else } x_1 \leftarrow x_2 $


**HPS**: 

    - reverse_physical: Principle can be reversed.
    - filter_mode: : As the method only directs each edge,an additional filter can be applied that, for each parent, selects either the largest mean or the next closest mean.

-----

## Cross Correlation Peak + Reverse Physical
Combines both naive principles and only considers the union over the links

$$ \text{CrossCorr } \cup   \text{ Phsyical}$$

**HPS**: 

    - max_lag: maximum lag that is evaluated in both directions.
    - reverse_physical: Principle can be reversed.
    - filter_mode: Applies one of the three filters after the union operator.

-----

## Direct Cross Correlation

Attempts to recover the window causal graph by simply thresholding the lagged-cross-correlation Tensor.


$$ \text{Causal Strength}_{X \rightarrow Y, k}  = \frac{\frac{1}{n}\sum{t=1}^{n-|k|} (X_t - \bar{X})(Y_{t-k} - \bar{Y})}{s_X s_Y} $$



**HPS**: 

    - max_lag: maximum lag that is evaluated in both directions.

-----

## VAR
A simple var model is fitted to the time series process:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{A}_i \mathbf{Y}_{t-i} + \boldsymbol{\varepsilon}_t$$

Now we either use the coefficients directly ($\mathbf{A}$) or rely on p-values of the fitted model.


**HPS**: 

    - max_lag: the order of the model
    - base_on: Whether to base decisions on 'coefficients' or 'pvalues'
    - absolute_coefficients: Whether or not to consider the absolute values of the coefficients.
-----


## Varlingam

VAR model for lagged effects + Lingam for instantanous effects.  

$$x(t) = \sum_{\tau=0}^{k} B_\tau x(t - \tau) + e(t)$$


**Key assumptions**:

- linearity
- non gaussian error
- acyclicity


**HPS**: 

    - max_lag: the order of the model (can be disabled by setting criterion)
    - criterion: Model selection criterion ('aic', 'fpe', 'hqic', 'bic', or None). If set, automatically searches for best lag.
    - prune: Whether to prune weak edges from the causal graph

-----

## Dynotears

Reformulates the structure as an optimization problem that can be solved via standard solvers by adding the aciclicity constraint via augmented lagrangian.


$$\min_{W, A} \, f(W, A) \quad \text{s.t.} \quad W \text{ is acyclic}, \tag{5}$$
$$f(W, A) = \ell(W, A) + \lambda_W \| W \|_1 + \lambda_A \| A \|_1.$$
$$\ell(W, A) = \frac{1}{2n} \| X - XW - YA \|_F^2.$$

$$F(W, A) = f(W, A) + \frac{\lambda}{2} h(W)^2 + \alpha h(W) \tag{7}$$

$$\begin{equation}
h(W) = \text{tr}\left( e^{W W^\top} \right) - d \quad \text{s.t.} \quad h(W) = 0 \iff W \text{ is acyclic}. \tag{8}
\end{equation}$$




**HPS**: 

    - max_lag: the order of the model
    - lambda_w: l1 loss for instantaneous coefficients
    - lambda_a: l1 loss for lagged coefficients
    - max_iter: number of dual ascent steps
    - h_tol: breaks earlier if acyclicity constraint is satisfied (tolerance for h(W))




-----

## PCMCI
PCMCI combines the PC algorithm with momentary conditional independence (MCI) testing.

**Algorithm**:
1. **Condition-selection step**: For each variable, identify potential parents $\widehat{\mathcal{P}}$ using iterative conditional independence tests (variant of PC1).
2. **MCI step**: Test conditional independence between variables given their selected conditions to remove false positives caused by autocorrelation.

For variables $X_t$ and $Y_{t-\tau}$, test:
$$X_t \perp\!\!\!\perp Y_{t-\tau} \mid \widehat{\mathcal{P}}(X_t) \setminus \{Y_{t-\tau}\}, \widehat{\mathcal{P}}(Y_{t-\tau})$$

**Key innovation**: Restores well-calibrated false positive rates and improves statistical power in highly autocorrelated time series.


**HPS**: 

    - max_lag: the order of the model
    - ci_test: Conditional independence test that is used.

----
## PCMCI+
Extends PCMCI to discover both lagged and contemporaneous (instantaneous) causal relationships.

**Algorithm**:
1. **Lagged discovery**: Apply PCMCI to identify lagged parents $\mathcal{P}^-(X_t)$
2. **Contemporaneous discovery**: For same-time variables, test:

$$X_t \perp\!\!\!\perp Y_t \mid \mathcal{P}^-(X_t), \mathcal{P}^-(Y_t), \mathcal{S}$$

where $\mathcal{S}$ is a separating set of contemporaneous variables.

3. **Orientation**: Orient contemporaneous links using conditional independence tests and orientation rules to ensure acyclicity within each time slice.

**Key innovation**: Handles both lagged and instantaneous causal effects while maintaining computational efficiency.

**HPS**: 

    - max_lag: the order of the model
    - ci_test: Conditional independence test that is used
    - reset_lagged_links: Whether to reset lagged links during orientation
    - contemp_collider_rule: Rule for resolving contemporaneous colliders ('majority', 'conservative', 'contemporary')
-----

## FPCMCI
Filtered PCMCI. Uses Transfer Entropy to filter out variables before applying PCMCI


**HPS**: 

    - max_lag: the order of the model
    - ci_test: Conditional independence test that is used (e.g., 'parcorr', 'robust_parcorr')

-----


## Causal Discovery using Model Invariance (CDMI)

CDMI uses Deep-Ar for Granger causality testing with knockoff-based variable selection.

**Algorithm**:
1. **Fit predictive model**: Train neural network $\hat{f}$ to predict $X_t$ from lagged variables:

$$\hat{X}_t = \hat{f}(X_{t-1}, X_{t-2}, \ldots, X_{t-p})$$

2. **Generate knockoffs**: Create knockoff variables $\tilde{X}$ that preserve the conditional distribution structure

3. **Test for causality**: For each variable $Y$, compute test statistic based on prediction improvement:

$$T_Y = \text{Importance}(Y) - \text{Importance}(\tilde{Y})$$

4. **Threshold**: Variables with $T_Y > \lambda$ are identified as causes

**Key innovation**: Combines deep learning's expressiveness with rigorous statistical guarantees via knockoff framework.


**HPS**: 

    - batch_size: Batch size for training
    - freq: Frequency parameter (required by gluonts)
    - device: Device to run on ('cpu' or 'cuda')
    - learning_rate: Learning rate for training
    - epochs: Number of training epochs
    - num_layers: Number of LSTM layers
    - num_cells: Number of LSTM cells per layer
    - dropout_rate: Dropout rate
    - context_length: Length of context window
    - num_samples: Number of samples for prediction
    - intervention_type: Type of intervention ('knockoff')
    - error_metric: Error metric for evaluation ('mae')
    - significance_test: Statistical test ('kolmo')
    - mean_std: Mean standard deviation threshold
    - step_size: Step size for windowing
    - prediction_length: Length of prediction horizon
    - normalize_effect_strength: Whether to normalize effect strengths
    - select_automatic_data_split: Whether to automatically split data
    - training_length: Length of training data
    - num_windows: Number of windows for training
    - use_cached_model: Whether to use cached model
    - save_intermediate: Whether to save intermediate results

-----
## Causal Pretraining
A Direct mapping from Time series to Window causal graph that is learned in a supervised manner from synthetic data samples. It is parameterized by an arbitrary neural network.

A direct mapping from Time Series (TS) to a Causal Graph learned via supervised learning on massive synthetic datasets. Parameterized by a neural network (e.g., Transformer).
$$ f(\mathbf{X}_{t-p:t}) \rightarrow \mathbf{A} \in [0,1]^{d \times d} $$


**HPS**: 

    - batch_size: Batch size (only relevant for very long time series)
    - architecture: 'transformer' or unidirectional architecture
    - weight_path: Path to pretrained model weights
    - batch_aggregation: How to aggregate causal estimations over batch dimension (e.g., 'mean', only relevant if time series is longer than 600)



-----


-----
## NTSNOTEARS

Nonlinear Time Series NOTEARS extends DYNOTEARS by replacing linear relationships with neural networks to capture nonlinear temporal dependencies.

**Model**: For each variable $X_t^i$:

$$X_t^i = \sum_{j=1}^d f_{ij}(X_t^j) + \sum_{j=1}^d \sum_{\tau=1}^p g_{ij}^\tau(X_{t-\tau}^j) + \varepsilon_t^i$$

where:
- $f_{ij}$: Neural network modeling instantaneous effects (subject to acyclicity)
- $g_{ij}^\tau$: 1D CNN modeling lagged effects at lag $\tau$

**Optimization**: Same acyclicity constraint as DYNOTEARS, but applied **only** to the contemporaneous network weights $W_f$:
$$\min_{f, g} \, \mathcal{L}(f, g) + \lambda h(W_f) \quad \text{s.t.} \quad h(W_f) = 0$$
where $W_f$ is a matrix representing the $L_2$ norms of the first layers of the contemporaneous neural networks $f_{ij}$. Lagged networks $g$ are unconstrained since time prevents cyclic dependencies.

**Key innovation**: Captures nonlinear relationships while maintaining differentiability for efficient optimization.



**HPS**: 

    - max_lag: the order of the model
    - seed: Random seed for reproducibility
    - h_tol: Tolerance for acyclicity constraint
    - rho_max: Maximum value for augmented Lagrangian penalty parameter
    - lambda1: L1 regularization weight for instantaneous effects
    - lambda2: L1 regularization weight for lagged effects

-----

## SVAR-FCI

Structural VAR with Fast Causal Inference (SVAR-FCI) extends the FCI algorithm to time series data with potential latent confounders.

**Model**: Assumes a structural VAR with latent variables:

$$X_t = \sum_{\tau=1}^p B_\tau X_{t-\tau} + \Gamma L_t + \varepsilon_t$$

where $L_t$ represents unobserved confounders.

**Algorithm**:
1. **Adjacency search**: Build skeleton graph using conditional independence tests on time series
2. **Orientation**: Apply orientation rules considering possible latent confounders
3. **Output**: Returns Partial Ancestral Graph (PAG) with edges:
   - $\rightarrow$: Direct causal effect
   - $\leftrightarrow$: Latent confounder
   - $\circ$: Ambiguous orientation

**Key assumption**: Causal sufficiency not required—can handle hidden confounders.

**HPS**:

    - max_lag: Maximum time lag to consider
    - ci_test: Conditional independence test
    - alpha: Significance level for independence tests

-----

## SVAR-RFCI

Really Fast Causal Inference for Structural VAR (SVAR-RFCI) is a computationally efficient variant of SVAR-FCI.

**Key difference from SVAR-FCI**: Uses more aggressive conditional independence test ordering to reduce computational cost while maintaining correctness guarantees.

**Computational improvement**: 
- SVAR-FCI: Tests all possible separating sets up to maximum size
- SVAR-RFCI: Uses heuristics to prioritize likely separating sets

**Trade-off**: Faster runtime with same asymptotic correctness but may require more data for finite-sample accuracy.

**Output**: Same PAG structure as SVAR-FCI with uncertainty about edge orientations in presence of latent confounders.

**HPS**:

    - max_lag: Maximum time lag to consider
    - ci_test: Conditional independence test  
    - alpha: Significance level for independence tests

-----


## LPCMCI

Latent PCMCI (LPCMCI) is a constraint-based causal discovery method for time series that handles **latent/hidden confounders** (Gerhardus & Runge, NeurIPS 2020). Unlike PCMCI / PCMCI+, which assume causal sufficiency (no hidden common causes), LPCMCI relaxes this assumption and outputs a **partial ancestral graph (PAG)** that can represent: directed edges (`-->`), bidirected edges (`<->` indicating latent confounders), and circle marks (`o` indicating orientation uncertainty).

**Key innovation**: LPCMCI identifies that the root cause of poor recall of FCI-type methods on autocorrelated time series is low conditional-independence test effect size. It recovers recall by including causal parents in the conditioning sets early — during edge removal — combined with novel orientation rules that already operate during the skeleton phase.

**Implementation**: wraps `tigramite.lpcmci.LPCMCI` (tigramite is already a cd_zoo dependency). The API mirrors the PCMCI+ wrapper exactly: construct a tigramite DataFrame, instantiate `LPCMCI(dataframe, cond_ind_test)`, call `run_lpcmci(tau_max=...)`. The continuous edge-score matrix is derived from `pval_max_matrix` (max p-value across all CI tests per link), reversed as `1 - p` — the **same convention as PCMCI+ / SVAR-FCI**.

**When to use**: LPCMCI is the method of choice when hidden confounders are suspected and the causal-sufficiency assumption may be violated. If causal sufficiency holds, PCMCI+ is more powerful (fewer uncertainty marks). LPCMCI is the observational fallback for the interventional CAnDOIT method.

**HPS**:

    - max_lag: Maximum time lag to consider (default 3)
    - pc_alpha: Significance level for CI tests (default 0.05)
    - ci_test: Conditional independence test (Hydra group, e.g. ParCorr)

-----


## VARMA-LiNGAM

VARMA-LiNGAM is a structural vector autoregressive moving-average model estimated using non-Gaussianity (Hyvärinen, Zhang, Shimizu & Hoyer, JMLR 2010). It extends **VAR-LiNGAM** by adding a moving-average (MA) component, allowing it to model both autoregressive and moving-average noise structure.

**Model**: `x(t) = B·x(t) + Σᵢ Aᵢ·x(t-i) + e(t) + Σⱼ Mⱼ·e(t-j)` where B is the instantaneous effect matrix, Aᵢ are AR coefficients, and Mⱼ are MA coefficients (effects of past noise terms — not edges between observed variables).

**Implementation**: wraps `lingam.VARMALiNGAM` (the `lingam` package, already a cd_zoo dependency). The API mirrors the VAR-LiNGAM wrapper: `VARMALiNGAM(order=(p,q), criterion=..., prune=...).fit(X)` → `.adjacency_matrices_`. The property returns a **tuple of two** arrays `(psis, omegas)`: `psis[0]` is the instantaneous effect matrix B (lag 0), `psis[1:]` are the lagged AR coefficients; `omegas` are the MA coefficients (ignored for structure scoring since they represent noise-term effects, not observed-variable edges). Only `psis` is used, transposed to `[effect, cause, lag]` — the same convention as VAR-LiNGAM (verified empirically: AUROC 1.0 on a known VAR, no additional transpose needed).

**Note**: VARMA coefficient estimation (via statsmodels) is considerably slower than plain VAR. The noise must be non-Gaussian for full identifiability of the instantaneous effects.

**HPS**:

    - order: (AR order, MA order) tuple (default [3, 1])
    - criterion: Lag-order selection criterion ('aic'/'fpe'/'hqic'/'bic'/None)
    - prune: Whether to prune weak edges via LASSO (default True)

-----


## oCSE

Optimal Causation Entropy (oCSE) is an information-theoretic causal discovery method for time series (Slote, Fish & Bollt, 2025). It is **training-free** (classical — no neural network, no pretrained weights) and discovers causal networks using conditional mutual information (CMI).

**Algorithm**: Two-phase approach — (1) **forward selection**: iteratively add predictors that maximise conditional mutual information with the target; (2) **backward elimination**: remove predictors that lose significance when conditioned on the remaining set. The causal rule: variable X causes Y if knowing X(t) significantly improves prediction of Y(t+1), even when controlling for all other relevant variables. Supports Gaussian, kNN, KDE, geometric-kNN, and Poisson entropic estimators.

**Implementation**: wraps `causationentropy.discover_network` (`pip install causationentropy` — MIT, Python 3.12 + numpy <2 compatible, already installed). The function takes a `(T, n_vars)` DataFrame and returns a `networkx.MultiDiGraph` whose directed edges `(cause, effect)` carry `cmi` (conditional mutual information = edge strength), `p_value`, and `lag` attributes. The wrapper converts this into a continuous `[effect, cause, lag]` score matrix scored by CMI, split into lagged and instantaneous channels.

**Note**: The permutation testing phase (`n_shuffles` iterations per edge) can be computationally intensive; `n_shuffles` defaults to 50 (reduced from the package default) for speed.

**HPS**:

    - max_lag: Maximum time lag (default 3)
    - method: Discovery algorithm ('standard'/'alternative'/'information_lasso'/'lasso')
    - information: Entropic estimator ('gaussian'/'knn'/'kde'/'geometric_knn'/'poisson')
    - n_shuffles: Permutation-test iterations (default 50)
    - alpha_forward / alpha_backward: Significance thresholds (default 0.05)

-----

## DirectLiNGAM (i.i.d.)

DirectLiNGAM (Direct Linear Non-Gaussian Acyclic Model) estimates a linear structural equation model from **i.i.d.** observational data under the assumption of **non-Gaussian** noise [Shimizu et al. 2011]. Unlike constraint-based methods, LiNGAM is *fully identifiable*: given non-Gaussian noise it recovers the exact DAG (not just the Markov-equivalence class).

**Implementation**: wraps `lingam.DirectLiNGAM` (the same library used by `varlingam`, here without the VAR lag structure). Returns `|B|` where `B[effect, cause]` is the estimated structural coefficient matrix — matching cd_zoo's `[effect, cause]` convention directly.

**Identifiability**: with (near-)Gaussian noise the orientation of edges is **not identifiable** and some directions may be estimated arbitrarily; use this method when the data is believed to be non-Gaussian, or treat orientation results on Gaussian data with caution.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - random_state: Random seed for the estimation

-----

## PC (i.i.d.)

The PC algorithm [Spirtes, Glymour & Scheines 2000] is a **constraint-based** method that learns a causal graph by testing conditional independencies (CI) between variables and combining them into a CPDAG (Completed Partially Directed Acyclic Graph). Edges whose direction is not determined by colliders / background knowledge remain **undirected**.

**Implementation**: wraps `causallearn.search.ConstraintBased.PC.pc`. The learned CPDAG is converted to a continuous `[effect, cause]` score matrix: a directed edge `i → j` scores `1.0` at `[effect=j, cause=i]`, an undirected edge `i – j` scores `0.5` in both directions (genuine uncertainty), and an absent edge scores `0.0`.

**CI test**: the default `fisherz` (partial correlation) assumes **linear-Gaussian** relationships and is fast. For nonlinear data use e.g. `kci` or `rcit` (slower). PC's output is a CPDAG: edges that are not oriented by colliders remain undirected, which is the correct representation of the Markov-equivalence class.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - alpha: Significance level for the conditional-independence tests
    - indep_test: CI test name (e.g. `fisherz`, `kci`, `rcit`)

-----

## AVICI (i.i.d.)

AVICI (Amortized Variational Inference for Causal Inference) [Lorch et al. 2022] is a **deep-learning / amortized** method: a neural network is trained once on synthetic SCMs and, at inference time, maps a data matrix directly to a `[d, d]` matrix of per-edge posterior probabilities in a single forward pass (seconds on CPU). The default checkpoint `scm-v0` is a general-purpose model trained on a broad prior of linear & nonlinear SCMs with Gaussian / Laplace / Cauchy noise and up to 100 nodes.

**Implementation**: wraps `avici.load_pretrained(download=...)` + `model(x)`. The model is loaded once and **cached at module level** (the checkpoint download + JAX graph construction are expensive) and reused across samples. AVICI outputs `g_prob[i, j]` = P(edge `i → j`) in `[cause, effect]` orientation, so the prediction is **transposed** to cd_zoo's `[effect, cause]` convention before returning.

**Installation**: AVICI is JAX-based and not part of the default environment. Install with `pip install avici` (requires JAX). On Python 3.12 the pinned `pyarrow==10.0.1` dependency does not build; an inference-only install is possible with `pip install avici --no-deps` plus `jax`, `jaxlib`, `dm-haiku`, `optax`, `deepdiff`, `huggingface-hub` (the `tensorflow`/`pyarrow` dependencies are only needed for *training* / data simulation, not pretrained inference). The model checkpoints are downloaded from HuggingFace on first use (~60 MB, cached).

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - download: Pretrained checkpoint name (`scm-v0` general-purpose, `neurips-linear`, `neurips-rff`, `neurips-grn`)

-----

## LCM

LCM (Large Causal Models) [Kougioulis et al. 2026] is an **amortized** temporal-structure-discovery method: a transformer-style network pretrained on a diverse mixture of synthetic and realistic time-series generators, then applied zero-shot to new data in a single forward pass. It scales to larger variable counts than earlier amortized TS methods and is competitive with classical baselines, particularly in out-of-distribution settings.

**Implementation**: wraps `lcm_pytorch` (pip-installable from GitHub). The 2.5M-parameter checkpoint is downloaded from Google Drive on first use and **cached at module level**. The model ingests a standardized time series and a lagged cross-correlation feature, and emits a lagged adjacency `pred[effect, cause, lag]`. The lag axis is **flipped** (`np.flip(axis=2)`) because the model emits lags in descending order (index 0 = highest lag) whereas cd_zoo expects ascending order (index 0 = lag 1) — identical to the convention used by the `cp` method. Orientation matches cd_zoo directly (**no transpose**). Self-loops are zeroed per lag slice. Supports up to 12 variables, 3 lags and 500 timesteps.

**Installation**: `pip install git+https://github.com/kougioulis/LCM` (pulls `lcm_pytorch`). Requires `torch` + `lightning` + `gdown` (for the checkpoint download). The checkpoint (~40 MB) is cached under `cd_zoo/methods/lcm_weights/`.

**Output**: lagged adjacency `pred[effect, cause, lag]` (no separate instantaneous channel).

**HPS**:

    - checkpoint_size: Model size (`2.5M`; larger sizes may be released)
    - cache_dir: Directory for the downloaded checkpoint

-----

## SDCI

SDCI (Causal Discovery from Conditionally Stationary Time Series) [Balsells-Rodas et al. 2025, ICML] is an **amortized** temporal-structure-discovery method built on an NRI-style MLP edge encoder. The encoder maps a multivariate time series directly to a matrix of per-edge posterior probabilities over (hidden) regime states, yielding a **summary** causal graph in one forward pass.

**Implementation**: vendors the *minimal* model code (MLP encoder + `create_rel_rec_send` + `my_softmax`) needed for a single encoder forward pass — the state-encoder/decoder (only used for variational reconstruction during training) are not loaded, and the `torch_scatter` dependency (required only by the unused `att`/`rnn` edge encoders) is avoided entirely. The released springs *region* checkpoint is downloaded from Google Drive and **cached at module level**. The encoder weights are independent of the node count (per-node/per-edge MLPs), so the model generalises to any number of variables — only `rel_rec`/`rel_send` (rebuilt at runtime) depend on `num_atoms`.

  The checkpoint was trained on 2-D particle simulations with 4 features per node (`[x, y, vx, vy]`) over 80 timesteps. 1-D benchmark series are expanded to 4 features via discrete derivatives (position / velocity / acceleration / jerk), each normalised to `[-1, 1]`, and truncated to 80 steps — an **out-of-distribution** setting, so absolute accuracy on arbitrary benchmarks is expected to be modest. Edge probabilities are aggregated over regime states (max) to `[effect, cause]` (receiver = effect, sender = cause), matching cd_zoo's convention directly (**no transpose**).

**Installation**: no pip package; the model code is vendored in `methods/sdci.py`. Requires `torch` + `gdown`. The checkpoint (~4 MB) is cached under `cd_zoo/methods/sdci_weights/`.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - cache_dir: Directory for the downloaded checkpoint

-----

## CDFM (i.i.d.)

CDFM (Causal Discovery Foundation Model) [Qiao et al. 2026] is a pretrained foundation model for **zero-shot** causal discovery from observational tabular data. Given a data matrix `X (N, D)`, it predicts the causal graph `G (D, D)` in a single forward pass. It is pretrained on a massive, highly diverse space of synthetic structural causal models and generalizes zero-shot to any sample size `N` and variable count `D`, outperforming baselines across 15 mechanism families and on real-world benchmarks.

**Implementation**: wraps `cdfm-base` (pip-installable, `from cdfm import CDFM`). The model is loaded once via `CDFM.from_pretrained('DMIRLAB/CDFM')` and **cached at module level** (the checkpoint download from HuggingFace is expensive) and reused across samples. CDFM **auto-standardizes** inputs and handles missing values / categorical columns internally. Its `predict` returns `probabilities[i, j]` = P(edge `i -> j`) in `[cause, effect]` orientation, so the prediction is **transposed** to cd_zoo's `[effect, cause]` convention before returning — the same convention as AVICI. (Verified empirically: transposed gives AUROC 0.89 on a known linear-Laplace SCM with a collider, as-is gives ~0.21; 0.76 on the real Sachs benchmark.)

**Installation**: `pip install cdfm-base`. Requires `torch`, `numpy`, `scikit-learn`, `networkx`, `safetensors`, `huggingface_hub` (all already satisfied by the cd_zoo environment). The checkpoint is downloaded from HuggingFace on first use and cached in the default HF cache.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - repo_id: HuggingFace repo id (default `DMIRLAB/CDFM`)

-----

## Arrow (i.i.d.)

Arrow [Thompson et al. 2026] is a **zero-shot** foundation model for causal discovery on observational tabular data. It factorizes a DAG into an undirected skeleton and a topological order (guaranteeing acyclicity by construction), uses a transformer to contextualize variables within and across observations, then predicts skeleton edge probabilities and node-order scores that together define a graph. It is trained on diverse synthetic SCMs and matches or outperforms existing methods at substantially lower inference cost.

**Implementation**: wraps `arrowfm` (pip-installable from GitHub, `from arrowfm import ArrowPredictor`). The model is loaded once via `ArrowPredictor()` (which auto-fetches the checkpoint from HuggingFace) and **cached at module level**. Arrow **standardizes** inputs internally. The wrapper calls `predict_parameters(x)` to obtain the raw edge-probability tensor `p_hat` (rather than the decoded binary adjacency), which is what `score` needs for AUROC. `p_hat` is in `[cause, effect]` orientation — i.e. `p_hat[i, j]` is the probability of the directed edge `i -> j` — following directly from the architecture: `p_hat[i,j] = a[i,j] * m[i,j]` where `a` is the symmetric skeleton probability and `m[i,j] = sigmoid(s[i]-s[j])` is the probability node `i` precedes node `j` in topological order (the `decode_dag` decoder uses the same indexing). cd_zoo scores in `[effect, cause]`, so the prediction is **transposed** (`.T`) before scoring. (Verified empirically on the `iid_nongaussian` test set with known random DAGs: transposed gives AUROC 1.000 on every sample; as-is gives ~0.25 — worse than chance, the signature of a transposition error. On the real Sachs benchmark, `p_hat` `[cause, effect]` compared against causal-benchmarks' `[cause, effect]` GT directly gives AUROC 0.66.)

**Installation**: `pip install git+https://github.com/ryan-thompson/arrowfm.git`. Requires only `torch` and `huggingface-hub` (the lightest dependency footprint of any method in the zoo). The checkpoint is auto-fetched from HuggingFace on first use and cached in the default HF cache.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - (none; the model is loaded from the released checkpoint)

-----

## TabCausal (i.i.d.)

TabCausal [Li et al. 2026] is a **zero-shot** foundation model for causal discovery on observational tabular data. It uses an axial transformer that alternates attention over variables and over observations, predicts directed-edge logits via a bilinear head, and is trained on a large, diverse distribution of synthetic SCMs spanning many graph families, mechanisms, and sample sizes. It generalises zero-shot to unseen node counts, sample sizes, and SCM families, and supports an optional intervention-indicator channel (unused here, as cd_zoo feeds purely observational data).

**Implementation**: wraps the `tabcausal` package (`from tabcausal import TabCausalPredictor`, installed `--no-deps` from `LAMDA-Tabular/TabCausal`). The predictor is loaded once via `TabCausalPredictor(checkpoint_path)` (checkpoint auto-fetched from HuggingFace via `hf_hub_download`) and **cached at module level**. TabCausal **standardizes** each variable within-graph internally. The cd_zoo wrapper builds the two-channel input `[observations, variables, 2]` (value + zero intervention flag) from the `(n_obs × n_vars)` table, calls `predict_proba`, and transposes the result. TabCausal's `predict_proba` returns `probabilities[i, j]` = P(edge `i -> j`) in `[cause, effect]` orientation, so the prediction is **transposed** to `[effect, cause]` — the same convention as AVICI and CDFM. (Verified empirically: transposed gives AUROC 1.0 on a known linear collider-chain SCM, as-is gives ~0.52; 0.74 on the real Sachs benchmark.)

**Installation**: `pip install --no-deps .` from the `LAMDA-Tabular/TabCausal` repo. The core `tabcausal` package depends only on `numpy`, `torch`, `scikit-learn`, `tqdm`, `pandas`, and `matplotlib` (all already in the environment); `pyarrow` is listed but only lazily imported for parquet I/O, which the wrapper never triggers. The checkpoint is auto-fetched from HuggingFace on first use and cached in the default HF cache.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - `repo_id` (default `LAMDA-Tabular/TabCausal`)
    - `checkpoint` (default `checkpoints/tabcausal-base.pt`)
    - `device` (default auto)

-----

## GES (i.i.d.)

GES (Greedy Equivalence Search) [Chickering 2002] is a **score-based** causal discovery algorithm for i.i.d. observational data. Rather than performing conditional-independence tests (as PC does), GES searches over the space of CPDAGs (Markov equivalence classes of DAGs) by greedily adding edges in a forward phase and then deleting edges in a backward phase, each step maximising a decomposable score. With the default BIC score it assumes (near-)linear-Gaussian relationships; causal-learn also provides cross-validated and marginal-general score variants for non-Gaussian / nonlinear data. Like PC, GES returns a CPDAG: edges whose orientation is not determined by the score alone remain undirected.

**Implementation**: wraps `causallearn.search.ScoreBased.GES.ges` (causal-learn is already a dependency of cd_zoo, used by the PC method). The function returns a dict whose `'G'` entry is a causal-learn `GeneralGraph` whose `.graph` attribute encodes the learned CPDAG with the standard endpoint convention (`-1` tail, `1` arrowhead) — **exactly the same format PC produces**. The prediction is therefore derived with the **same `_cpdag_to_scores` helper** as PC (imported from `methods.pc` rather than duplicated): directed edges score `1.0`, undirected edges `0.5`, absent edges `0.0`, in `[effect, cause]` orientation (no transpose).

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - `score_func` (default `local_score_BIC`; alternatives: `local_score_CV_general`, `local_score_marginal_general`)
    - `maxP` (default `null` = no cap on number of parents)

-----

## ICA-LiNGAM (i.i.d.)

ICA-LiNGAM [Shimizu et al. 2006] is the original **LiNGAM** estimator for i.i.d. observational data: a linear non-Gaussian acyclic model estimated via Independent Component Analysis. It first separates the data into independent components using ICA and then permutes them into a causal ordering. Like DirectLiNGAM it is fully identifiable (recovers the exact DAG, not just the Markov-equivalence class) only when the noise is non-Gaussian, but it can be more brittle for large variable counts and is best suited to smaller graphs.

**Implementation**: wraps `lingam.ICALiNGAM` (the `lingam` package, already a dependency of cd_zoo via the DirectLiNGAM method). The API is **identical to DirectLiNGAM** — `ICALiNGAM().fit(X)` then `.adjacency_matrix_` — so the wrapper mirrors `methods.lingam` exactly. The absolute value `|adjacency_matrix_|` is returned as the per-edge score. `adjacency_matrix_` is stored in companion form `B[effect, cause]` (coefficient of the cause in the effect's structural equation), matching cd_zoo's `[effect, cause]` convention — **no transpose** is applied (same orientation as DirectLiNGAM; verified empirically: AUROC 1.0 as-is on a known linear non-Gaussian collider-chain SCM).

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - `random_state` (default `0`)

-----

## DAGMA (i.i.d.)

DAGMA (DAGs via M-matrices and a Log-Determinant acyclicity characterization) [Bello, Aragam, Ravikumar 2022] is a **continuous-optimization** causal discovery method for i.i.d. observational data. Like NOTEARS it replaces the combinatorial acyclicity constraint with a smooth, differentiable surrogate, but DAGMA uses a log-determinant formulation `h(W) = -log det(sI − W∘W) + d·log s` (over M-matrices) that has better-conditioned gradients, is about an order of magnitude faster than NOTEARS, and is better at detecting large cycles. It fits a weighted adjacency matrix `W` by a path-following sequence of unconstrained problems solved with accelerated gradient descent.

**Implementation**: wraps `dagma.linear.DagmaLinear` (`pip install dagma --no-deps`; all transitive deps — numpy, scipy, torch, tqdm, igraph — were already satisfied by the cd_zoo environment). The API is `DagmaLinear(loss_type).fit(X, lambda1)` → `.W_est`, a weighted weight matrix. DAGMA follows the NOTEARS convention where the reconstruction loss is `||X − X @ W||²`, i.e. `x_j = Σ_k W[k, j] · x_k`; thus `W[k, j]` is the coefficient of cause `x_k` in the equation of effect `x_j` — i.e. `W[cause, effect]`. cd_zoo scores in the `[effect, cause]` convention, so the prediction is **transposed** (`|W_est|.T`) before returning. (Note this is the opposite of DirectLiNGAM, whose `adjacency_matrix_` is already `[effect, cause]`. Verified empirically: transposed gives AUROC 1.0 on a known linear collider-chain SCM, as-is gives ~0.30.) Self-loops on the diagonal are zeroed, and the tqdm progress bar is silenced by temporarily patching DAGMA's module-level `tqdm` name.

**Installation**: `pip install --no-deps dagma` (version 1.1.1, Apache 2.0). All dependencies were already present in the environment; numpy remained at 1.26.4.

**Output**: single summary graph `[effect, cause]` (no lags, no separate instantaneous channel).

**HPS**:

    - `loss_type` (default `l2` for continuous data; use `logistic` for binary)
    - `lambda1` (default `0.02`, the L1 sparsity penalty)
    - `w_threshold` (default `0.0`; if >0, hard-threshold small weights before scoring)

-----


## §9. Bridge Scripts for Prior and Benchmark Datasets

Three standalone bridge scripts (no Hydra; plain `argparse`) live at the cd_zoo
repository root and run **any** registered cd_zoo method on three families of
time-series data, scoring predictions against ground-truth graphs through
cd_zoo's `benchmarking` / `score` pipeline. Because they live *inside* cd_zoo,
Python places the cd_zoo root on `sys.path[0]`, so `from tools.method_loader
import ...` binds cd_zoo's own package directly (no `sys.path` hacking needed
for cd_zoo itself).

| Script | Data source | Structure | Default |
|---|---|---|---|
| `run_method_on_prior.py` | `linear_new` prior grid | 91 cells `nvars{N}_lags{L}_ts512` (N∈2..16, L∈0..6), 5000 samples/cell | VAR |
| `run_method_on_linear_test.py` | `linear_test` prior scenarios | 6 named edge-case scenarios (see below) | VAR |
| `run_var_sweep.py` | `causal-benchmarks` registry | all 24 TS benchmark entries with a ground truth (~775 datasets) | VAR sweep |

### Ground-truth orientation

- **`linear_new` / `linear_test`** store GT in **companion form**:
  `lagged_links[effect, cause, lag]` and `instant_links[effect, cause]` (source
  `cause` acts on target `effect`). This is **exactly cd_zoo's `[effect, cause]`
  convention, so no transpose is needed** (verified: VAR on `only_lagged` →
  AUROC 0.99; on the `linear_new` grid → mean 0.915).
- **`causal-benchmarks`** stores `adj[i,j]=1` ⟺ i causes j (`[cause, effect]`),
  so `run_var_sweep.py` **transposes** each GT before scoring (verified: VAR on
  `var_sl` → AUROC 1.0 with transpose, ~0.44 without).

### The 6 `linear_test` scenarios

Deliberately-designed edge cases (generated with `cp/cp_2/prior`):

| Scenario | N | true lags | noise | lag density | inst density | design |
|---|---|---|---|---|---|---|
| `gauss_inst_unid` | 8 | 0 | gaussian | 0.00 | 0.04 | instantaneous-only, Gaussian (LiNGAM-unidentifiable) |
| `many_lags` | 5 | 8 | gaussian | 0.93 | 0.00 | high lag order (8) |
| `no_links` | 5 | 0 | gaussian | 0.00 | 0.00 | null model (no edges) |
| `nvars20` | 20 | 3 | uniform | 0.27 | 0.05 | larger graph, uniform noise (LiNGAM-identifiable) |
| `only_inst` | 5 | 0 | gaussian | 0.00 | 0.16 | instantaneous only |
| `only_lagged` | 5 | 3 | gaussian | 0.79 | 0.00 | lagged only (no instant edges) |

### Usage

```bash
# VAR on the linear_new grid (default: 200 samples/cell, all 91 cells)
python run_method_on_prior.py

# any method, more samples, restricted grid
python run_method_on_prior.py --method dynotears --max-lag 3 --max-samples 500
python run_method_on_prior.py --n-vars 5 10 --max-lags-range 0 4

# VAR on the 6 linear_test edge-case scenarios
python run_method_on_linear_test.py
python run_method_on_linear_test.py --method varlingam --max-lag 8 --max-samples 100
python run_method_on_linear_test.py --scenarios only_lagged many_lags

# sweep VAR over every causal-benchmarks TS entry
python run_var_sweep.py
```

### Reference results: VAR vs VARLiNGAM on `linear_test` (`max_lag=8`)

VAR is the fast linear baseline (lagged-only); VARLiNGAM additionally recovers
the instantaneous graph via non-Gaussian ICA. Run with `--max-lag 8` (the
highest true lag count, needed for `many_lags`).

| Scenario | N | true lags | noise | VAR lag AUROC | VLG lag AUROC | VLG inst AUROC | VAR time | VLG time |
|---|---|---|---|---|---|---|---|---|
| `gauss_inst_unid` | 8 | 0 | gaussian | NaN | NaN | 0.709 | 1.1s | 75.9s |
| `many_lags` | 5 | 8 | gaussian | 0.983 | 0.993 | NaN | 0.6s | 26.6s |
| `no_links` | 5 | 0 | gaussian | NaN | NaN | NaN | 0.7s | 37.1s |
| `nvars20` | 20 | 3 | uniform | 0.870 | 0.968 | 0.857 | 0.9s | 175.9s |
| `only_inst` | 5 | 0 | gaussian | NaN | NaN | 0.675 | 0.8s | 44.4s |
| `only_lagged` | 5 | 3 | gaussian | 0.958 | 0.985 | NaN | 0.7s | 39.0s |

Lagged-graph mean (3 datasets with lagged edges): VAR **0.937** vs VARLiNGAM
**0.982** (VARLiNGAM wins 3/3). Findings:

- **VARLiNGAM dominates on lagged graphs** — it exploits non-Gaussian noise for
  better identifiability, most strikingly on `nvars20` (uniform noise → 0.968 vs
  0.870).
- **VARLiNGAM scores the instant graph; VAR cannot** (VAR returns
  `instant=None`). On **uniform**-noise `nvars20` the instant graph is
  LiNGAM-identifiable → 0.857; on **Gaussian**-noise `gauss_inst_unid` /
  `only_inst` the instant orientation degrades to near-chance (0.71/0.68) — these
  are the "unidentifiable" scenarios by design.
- **NaN cases are structural, not failures**: lag AUROC is NaN on `lags=0`
  datasets (empty lagged graph → one class); inst AUROC is NaN on
  `inst_density=0` datasets (no instant edges). `no_links` (null model) is a
  no-op for both methods.

### Reference results: VAR sweep on `causal-benchmarks`

`run_var_sweep.py` (default, `max_lag=3`) over all 24 TS entries with a ground
truth: 775 datasets, all OK, **mean AUROC 0.617**. VAR-family synthetic entries
score 1.0; `tcs` 0.91; `causaldynamics` 0.30–0.41; `aerosol_cloud` 0.0 (1-edge
proxy GT). Full per-dataset results in
`/home/datasets4/stein/cd_benchmark_performances/`.

### Notes

- The prior dataloader (`cp/cp_2/components/dataloader.py`) is imported via
  `sys.path` from the sibling `cp` repo by `run_method_on_prior.py`. Because
  `cp/cp_2` ships a top-level `tools.py` that collides with cd_zoo's `tools/`
  package, that script imports cd_zoo's tools **first** so `sys.modules['tools']`
  binds correctly before the `cp/cp_2` path is added.
- For `max_lags == 0` cells the lagged (summary) graph is empty → AUROC is NaN
  (one class) for lagged methods — expected.
- For `many_lags` (true `max_lags=8`) pass `--max-lag 8` or the method is
  under-lagged (VAR at `--max-lag 3` scores only 0.758 there vs 0.983 at 8).


