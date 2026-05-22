from __future__ import annotations


def patch_opendataval_openml() -> None:
    """
    Patch OpenDataVal's OpenML loader for newer scikit-learn/OpenML metadata.

    Some OpenML datasets expose the target column name in `dataset["categories"]`,
    which breaks OpenDataVal's `load_openml()` (it assumes all category keys are
    feature names). Some datasets also return `object` dtype arrays that need a
    float cast after dropping categorical columns.

    This function monkey-patches `opendataval.dataloader.datasets.datasets.load_openml`
    to match the more robust implementation.
    """
    try:
        import numpy as np
        from sklearn.datasets import fetch_openml
        from opendataval.dataloader.datasets import datasets as dsmod
    except Exception:
        return

    current = getattr(dsmod, "load_openml", None)
    if current is None:
        return
    if getattr(current, "_sequentialdv_patched", False):
        return

    def load_openml(data_id: int, is_classification: bool = True):
        dataset = fetch_openml(data_id=data_id, as_frame=False)
        feature_names = list(dataset.get("feature_names") or [])
        categories = dataset.get("categories") or {}
        category_list = list(categories.keys())

        if len(category_list) > 0:
            category_indices = [
                feature_names.index(x) for x in category_list if x in feature_names
            ]
            noncategory_indices = [
                i for i in range(len(feature_names)) if i not in category_indices
            ]
            X, y = dataset["data"][:, noncategory_indices], dataset["target"]
        else:
            X, y = dataset["data"], dataset["target"]

        if is_classification is True:
            _, y = np.unique(y, return_inverse=True)
        else:
            y = (y - np.mean(y)) / (np.std(y.astype(float)) + 1e-8)

        if not np.issubdtype(X.dtype, np.number):
            X = X.astype(float)

        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        return X, y

    load_openml._sequentialdv_patched = True  # type: ignore[attr-defined]
    dsmod.load_openml = load_openml  # type: ignore[assignment]
