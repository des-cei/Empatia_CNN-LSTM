# Empatia_CNN-LSTM

**Physiological emotion recognition with parallel CNN and LSTM branches.**

Empatia_CNN-LSTM contains research code for learning emotion-related patterns from physiological feature sequences. It combines handcrafted descriptors, feature-wise normalization, feature-map augmentation, and a parallel CNN–LSTM classifier, with optional principal component analysis (PCA) for feature compression. Dataset-specific scripts cover WEMAC and WESAD.

This repository accompanies:

> Junjiao Sun, Jorge Portilla, and Andres Otero. **Negative emotion recognition based on physiological signals using a CNN-LSTM model.** *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, pp. 3736–3741, 2024. [DOI: 10.1109/BIBM62325.2024.10822762](https://doi.org/10.1109/BIBM62325.2024.10822762).

The work relates to **Chapter 3** of Junjiao Sun's doctoral thesis, *Toward Practical and Uncertainty-Aware Affective Computing with Wearable Physiological Signals*. The companion [Empatia_DL](https://github.com/des-cei/Empatia_DL) repository covers the feature-map and CNN experiments.

> **Implementation status:** The model definitions are included, but the training scripts depend on dimension-aware preprocessing modules that are absent from this repository. Local paths and several experiment settings also require adaptation. Raw datasets, generated caches, normalization logs, and pretrained checkpoints are not included. The scripts should be treated as development-time research code rather than a ready-to-run reproduction of the thesis results.

## Method overview

The intended workflow is:

1. Extract window-level cardiovascular, electrodermal, and skin-temperature descriptors.
2. Apply feature-wise normalization (FWN).
3. Optionally compress the descriptors with PCA.
4. Construct feature sequences/maps and augment them using label-compatible frames.
5. Process each input through parallel convolutional and recurrent branches.
6. Concatenate the branch embeddings and classify the target emotion or affective category.

The feature extractors provide **123 descriptors**: 84 cardiovascular/BVP features, 34 GSR features, and 5 skin-temperature features. The training scripts also contain experiments using arousal, valence, and, for WEMAC, dominance targets. These targets are cast to integer class indices and optimized with cross-entropy; the supplied experiments are classification workflows.

### Implemented architecture

`ParallelCNNLSTMModel` is defined in both dataset-specific `Training/models/cnn_lstm.py` files. Both branches receive the same input tensor of shape **`(batch, sequence_length, feature_count)`**.

| Component | Implementation |
| --- | --- |
| CNN branch | Permute to `(batch, features, sequence_length)`; `Conv1d(F, 64, 3)` → ReLU → max pooling → `Conv1d(64, 128, 3)` → ReLU → max pooling → flatten → `LazyLinear(128)` → ReLU. |
| LSTM branch | Batch-first LSTM; take the final sequence output and project it to 128 dimensions. |
| Fusion | Concatenate the two 128-dimensional embeddings. |
| Classifier | Linear layer from 256 dimensions to the configured number of classes; returns logits. |

Both convolution layers use stride 1 and padding 1; pooling uses kernel and stride 2. Experiment defaults use an LSTM hidden size of 64 and two recurrent layers. A separate `SimpleLSTM` baseline is also included.

**Relationship to the thesis:** The checked-in convolutional branch uses **1D convolutions**. Chapter 3 describes a CNN–LSTM method with a 2D convolutional branch. This implementation detail must be recorded when comparing experiments with the chapter. The branches here operate in parallel; the CNN output is not fed into the LSTM.

### Feature-map strategies

| Strategy | Purpose |
| --- | --- |
| `AllFromOne` | Construct maps from a single participant's sequence, grouped by video or class. |
| `HalfAndHalf` | Combine a sequence with another label-compatible participant sequence. |
| `HalfAndRandom` | Combine a sequence with label-compatible sampled frames. |
| `All_concat` | Combine the outputs of the three strategies in the training scripts. |

The included basic generators use `numpy.resize` to produce square maps and stack them as `(F, F, N)`. This operation repeats or truncates values rather than interpolating images. These basic generators are not interchangeable with the missing dimension-aware generators used by training.

The training adapter converts maps from `(H, W, N)` to `(N, 1, H, W)` and then removes the singleton dimension. The model consequently interprets `H` as sequence length and `W` as feature count. Confirm the axis order when restoring the preprocessing pipeline: square maps can conceal an incorrect feature/time orientation.

## Repository guide

| Location | Purpose |
| --- | --- |
| `WEMAC/Pack_all_data.py` | Extract features from MATLAB recordings and associate spreadsheet labels. |
| `WEMAC/Data_extraction.py` | Read feature caches and prepare participant partitions. |
| `WEMAC/Data_normalization.py` | Search for feature-wise normalization functions. |
| `WEMAC/Create_feature_maps.py` | Basic WEMAC map generation and augmentation. |
| `WEMAC/Transfer_label.py` | Legacy IT06 label conversion. |
| `WESAD/read_data.py` | Read subject pickle files. |
| `WESAD/Feature_extraction_wesad.py` | Generate per-subject feature caches. |
| `WESAD/Abnormal_extraction_weasd.py` | Prepare cached WESAD features and labels. |
| `WESAD/Data_normalization_WESAD.py` | Search for WESAD normalization functions. |
| `WESAD/Create_feature_maps_wesad.py` | Basic WESAD map generation and augmentation. |
| Each dataset's `*_Signal_Features.py` files | Physiological feature implementations. |
| Each dataset's `FWN_main/` directory | Normalization and optimization routines. |
| Each dataset's `Training/models/cnn_lstm.py` | Parallel CNN–LSTM and standalone LSTM definitions. |

## Environment

The original README records an NVIDIA A30 GPU, CUDA environment 12.2, NVIDIA driver 535.154.05, and PyTorch `2.0.0+cu118`. The CUDA environment version and the PyTorch build tag are separate values. No pinned environment or Python version is provided.

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/des-cei/Empatia_CNN-LSTM.git
cd Empatia_CNN-LSTM
python -m venv .venv
source .venv/bin/activate
```

Install a compatible PyTorch/torchvision pair for your machine. Additional packages inferred from the source imports include:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib seaborn \
    mat73 biosppy neurokit2 distfit xlrd xlwt skfeature-chappers
```

The PCA training scripts additionally import RAPIDS `cudf` and `cuml`. Install versions compatible with your Python/CUDA environment if using those scripts. The package list above is not a tested dependency lockfile.

Training explicitly casts inputs to `torch.cuda.FloatTensor`, so the supplied loops require CUDA despite having a CPU device-selection branch. CPU support requires device-aware tensor conversion. The model class itself can operate on CPU.

### Model-only example

With PyTorch and torchvision installed, the model can be instantiated independently of the missing data pipeline. Run this from the repository root:

```python
import torch
from WEMAC.Training.models.cnn_lstm import ParallelCNNLSTMModel

model = ParallelCNNLSTMModel(
    input_size=123,
    hidden_size=64,
    num_layers=2,
    num_classes=2,
)
model.eval()
x = torch.randn(2, 30, 123)  # batch, sequence length, features
with torch.no_grad():
    logits = model(x)
print(logits.shape)  # torch.Size([2, 2])
```

The sequence length above is illustrative. Two pooling stages require at least four steps. Because the CNN branch flattens its output into a lazy linear layer, keep the pooled sequence length consistent after the first forward pass.

## Data preparation

Obtain WEMAC and WESAD separately under their respective access terms. The datasets are not bundled with this repository.

### WEMAC

The included extractor defaults to `BBDDLab_EH_CEI_VVG_IT07.mat` and `Labels_TabLab_IT07.xls`, using preprocessed BVP, GSR, and skin-temperature fields. BVP/GSR are sampled at 200 Hz and temperature at 10 Hz. Extraction uses 3,000-sample BVP/GSR windows with 2,500-sample overlap, equivalent to 15-second windows.

The intended preparation order is `Pack_all_data.py` → `Data_normalization.py` → feature-map generation. The cache is `Pack_all_data.json` and normalization produces a log of selected feature transformations.

The CNN–LSTM training filenames and hard-coded paths instead refer to **IT06** and a separate dimension-aware pipeline. Align the recording version, label schema, cache, and normalization log before connecting extraction to training.

### WESAD

The reader expects a dataset root containing subject files such as `S2/S2.pkl`. Configure the root in `Feature_extraction_wesad.py` and create its `json_files` output directory. The intended order is `Feature_extraction_wesad.py` → `Data_normalization_WESAD.py` → feature-map generation.

The checked-in extractor reads chest **ECG, EDA, and temperature at 700 Hz**, using 6,000-sample windows and 4,000-sample overlap. ECG is passed through the cardiovascular feature module under BVP-style names. This differs from the thesis's wrist BVP/GSR/SKT configuration. Reproducing that configuration requires adapting signal selection, sampling rates, and label alignment.

## Training experiments

These are experiment entry points **after** completing the integration work above:

| Script under `Training/` | Dataset | PCA | Epochs | Batch size | Default output classes |
| --- | --- | --- | --- | --- | --- |
| `WEMAC_cnn_lstm_it06.py` | WEMAC | No | 100 | 256 | 10 |
| `WEMAC_cnn_lstm_feature_fusion_it06.py` | WEMAC | Yes | 100 | 128 | 60 |
| `WESAD_cnn_lstm.py` | WESAD | No | 100 | 128 | 10 |
| `WESAD_cnn_lstm_feature_fusion.py` | WESAD | Yes | 50 | 128 | 60 |

All four scripts use cross-entropy loss, SGD with default learning rate `0.1`, momentum `0.9`, weight decay `5e-4`, and cosine scheduling with `T_max=200`. The default output sizes above are development settings, not the class counts of the thesis's binary and three-class benchmarks.

Configure targets, augmentation strategies, PCA dimensions, repetitions, and model parameters in each script's main block. Start with a single target, `AllFromOne`, and one fresh model/run. The default PCA loops sweep component counts from 122 down to 9, inclusive; check sample-count constraints before fitting PCA. Most scripts use ten repetitions; WESAD PCA uses five.

Scripts print loss, accuracy, weighted F1-score, timing, and aggregate statistics. Checkpoint saving is commented out, and `--resume` does not provide a complete save/resume workflow. Add explicit checkpoint handling if trained models are needed downstream.

## Evaluation and reproducibility

Several development-time choices need adjustment before using the scripts for independent benchmark evaluation:

- **PCA:** The fusion scripts fit separate PCA models to training and test data. Fit PCA only on training data and use the same fitted transformation for validation/test data, preserving a common coordinate system.
- **Partition overlap:** WEMAC fusion calls `set_reconstruct`, which adds the first third of the test maps and labels to training while retaining the full test set. Remove this overlap for held-out evaluation.
- **Independent runs:** Models are reused across repetitions, and in some scripts across targets or strategies. Instantiate a fresh model for each independent experiment. A loop named `K` does not itself establish K-fold or leave-one-subject-out evaluation; verify subject assignments when restoring the missing loaders.
- **Preprocessing and selection:** Fit normalization and dimensionality reduction within the training partition. Use an inner validation split for choosing preprocessing, PCA dimension, and checkpoints, and reserve test subjects for final evaluation.
- **Augmentation:** The scripts apply the selected augmentation to both partitions. Use original held-out maps when measuring performance on unaugmented observations.
- **Reported metrics:** Best accuracy and best F1 are selected across evaluation epochs and can come from different epochs. Report metrics from a checkpoint selected on validation data, with fixed seeds and documented subject splits.

These scripts alone do not establish reproduction of Chapter 3's final scores or subject-independent protocol. Consult the paper and thesis for the research results, and record the exact architecture, signals, labels, preprocessing, and evaluation settings used in each new experiment.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{sun2024negative,
  author    = {Sun, Junjiao and Portilla, Jorge and Otero, Andres},
  title     = {Negative emotion recognition based on physiological signals using a CNN-LSTM model},
  booktitle = {2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2024},
  pages     = {3736--3741},
  doi       = {10.1109/BIBM62325.2024.10822762}
}
```

Please also cite the original dataset publications when using WEMAC or WESAD data.

## License

No repository-level license file is included. Contact the maintainers to clarify reuse and redistribution terms. Dataset access and reuse remain subject to the respective providers' terms.
