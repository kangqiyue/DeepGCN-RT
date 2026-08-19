# DeepGCN-RT

[![Inference CI](https://github.com/kangqiyue/DeepGCN-RT/actions/workflows/inference-ci.yml/badge.svg)](https://github.com/kangqiyue/DeepGCN-RT/actions/workflows/inference-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.chroma.2023.464439-blue.svg)](https://doi.org/10.1016/j.chroma.2023.464439)

DeepGCN-RT is a graph neural network for predicting small-molecule retention
time. This repository accompanies the 2023 *Journal of Chromatography A*
article and provides the released source code, pretrained weights, datasets,
and transfer-learning results.

> Kang, Q.; Fang, P.; Zhang, S.; Qiu, H.; Lan, Z. Deep graph convolutional
> network for small-molecule retention time prediction. *Journal of
> Chromatography A* **1711** (2023), 464439.
> [https://doi.org/10.1016/j.chroma.2023.464439](https://doi.org/10.1016/j.chroma.2023.464439)

## Repository contents

- `inference.py`: supported command-line entry point for the released model.
- `model_path/best_model_weight.pth`: released 16-layer DeepGCN-RT checkpoint.
- `dataset.py` and `feature_ops.py`: molecular graph construction and features.
- `dataset/`: SMRT and transfer-learning datasets used by the project.
- `result/`: published transfer-learning result summaries.
- `train.py` and `transfer_learning.py`: original research training workflows.

## Quick start: pretrained inference

Python 3.9 or 3.10 is recommended. Create an isolated environment and install
the inference dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
```

Predict the retention time for ethanol on CPU:

```bash
python inference.py --smiles "CCO" --device cpu
```

Expected output with the released checkpoint:

```text
SMILES: CCO
Predicted retention time: 624.8956 s
```

`--SMILES` and `--model_path` remain accepted for compatibility with the
original command line. Run `python inference.py --help` for all options.

Retention time depends on the chromatographic system. The value returned here
is a model estimate in the domain represented by the training data, not a
universal retention time or an experimental measurement.

## Released checkpoint integrity

The bundled checkpoint is loaded in evaluation mode and is covered by a CPU
inference smoke test. Its SHA-256 checksum is:

```text
195135fe104a5da90189130e6cceeeee1aeb115be21d195d597ced2a0d766677
```

The current automated checks cover SMILES validation, molecular graph
construction, checkpoint integrity, and pretrained CPU inference. The original
training and transfer-learning scripts are retained for research provenance but
are not part of the inference CI workflow.

## Published model performance

The model was evaluated with mean absolute error (MAE), median absolute error
(MedAE), mean absolute percentage error (MAPE), mean squared error (MSE), and
R-squared. Values below reproduce the summary reported with the project.

| Model | Depth | MAE | MedAE | MAPE | R2 | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepGCN-RT | 3 | 27.97 | 14.01 | 0.035 | 0.892 | 3303 |
| DeepGCN-RT | 5 | 27.00 | 12.91 | 0.034 | 0.892 | 3288 |
| DeepGCN-RT | 8 | 26.61 | 12.44 | 0.034 | 0.892 | 3286 |
| DeepGCN-RT | 16 | **26.55** | **12.38** | **0.033** | **0.892** | 3299 |

For the full experimental setup, transfer-learning evaluation, and scientific
interpretation, refer to the article.

## Citation

GitHub can generate citation metadata from [`CITATION.cff`](CITATION.cff). The
corresponding BibTeX entry is:

```bibtex
@article{kang2023deepgcnrt,
  title   = {Deep graph convolutional network for small-molecule retention time prediction},
  author  = {Kang, Qiyue and Fang, Pengfei and Zhang, Shuai and Qiu, Huachuan and Lan, Zhenzhong},
  journal = {Journal of Chromatography A},
  volume  = {1711},
  pages   = {464439},
  year    = {2023},
  doi     = {10.1016/j.chroma.2023.464439}
}
```

## Contributing

Bug reports and focused improvements are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the information needed to reproduce an
inference problem and the local validation commands.

## License

DeepGCN-RT is distributed under the [Apache License 2.0](LICENSE.md).
