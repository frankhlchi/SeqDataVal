# Environment Setup

## Tested Configuration

This repository has been tested with the following versions:

| Package | Version |
|---------|---------|
| Python | 3.9.x |
| numpy | 1.25.x |
| pandas | 1.5.x |
| scikit-learn | 1.2+ |
| torch | 2.1.2 |
| torchvision | 0.16.2 |
| opendataval | 1.3.0 |
| transformers | 4.41.2 |

## Quick Setup

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate sequentialdataval
```

## Optional Dependencies

### For Utility Approximation Experiments

```bash
pip install cvxpy>=1.4.0
```

### For LLM Fine-tuning Experiments

```bash
pip install sentence-transformers>=2.2.0 transformers>=4.35.0
```

## Known Issues

### OpenML/OpenDataVal Compatibility

Some OpenML datasets have metadata that causes issues with OpenDataVal's loader in newer sklearn versions. This repository includes a runtime patch (`src/utils/opendataval_compat.py`) that is automatically applied when loading datasets.

If you encounter errors like:
```
ValueError: 'binaryClass' is not in list
```

The patch should handle this automatically. If not, manually apply:

```python
from src.utils.opendataval_compat import patch_opendataval_openml
patch_opendataval_openml()
```

### Transformers Version

Using transformers >= 4.42 may cause import issues with torchvision. We recommend sticking to transformers 4.41.x.
