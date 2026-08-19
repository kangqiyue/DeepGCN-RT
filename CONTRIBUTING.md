# Contributing

Contributions that improve pretrained inference, reproducibility, documentation,
or compatibility are welcome.

## Before opening an issue

Please include:

- the operating system and Python version;
- the exact command that failed;
- the complete error message;
- a minimal SMILES example, when relevant; and
- whether inference ran on CPU or CUDA.

Do not include API keys, access tokens, private datasets, or other credentials.

## Local checks

Create an isolated Python environment, then run:

```bash
python -m pip install -r requirements-inference.txt pytest
python -m pytest -q
python inference.py --smiles "CCO" --device cpu
```

Keep changes focused. Changes to model architecture, feature construction, or
released weights should explain their scientific impact and include validation
against the published behavior.
