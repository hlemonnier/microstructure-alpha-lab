"""Keep a pretrained tabular transform's overflow fallback local to one row.

The upstream fallback clips the entire query batch if any row's power transform
overflows. Here unchanged rows retain their ordinary extrapolated values. Only
an individually failing row uses the already fitted training bounds.
"""

from __future__ import annotations

import numpy as np


class RowLocalPowerFallback:
    def __init__(self, original):
        if original.normalization_method != "power":
            raise ValueError("Only the fitted power-transform pipeline needs this adapter")
        self.original = original
        self.row_fallbacks = 0

    def __getattr__(self, name):
        original = self.__dict__.get("original")
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)

    def _normalize(self, matrix):
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                result = self.original.normalizer_.transform(matrix)
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite power transformation")
            return result
        except ValueError:
            if len(matrix) > 1:
                middle = len(matrix) // 2
                return np.concatenate([self._normalize(matrix[:middle]), self._normalize(matrix[middle:])])
            clipped = np.clip(matrix, self.original.X_min_, self.original.X_max_)
            result = self.original.normalizer_.transform(clipped)
            if not np.isfinite(result).all():
                raise ValueError("Training-bound fallback failed for one query row")
            self.row_fallbacks += 1
            return result

    def transform(self, values):
        matrix = np.array(values, copy=True)
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise ValueError("Nonempty finite query matrix required")
        if matrix.dtype == np.float16:
            matrix = matrix.astype(np.float32)
        scaled = self.original.standard_scaler_.transform(matrix)
        return self.original.outlier_remover_.transform(self._normalize(scaled))


def install_row_local_power_fallback(estimator):
    for method, preprocessor in list(estimator.ensemble_generator_.preprocessors_.items()):
        if method == "power" and not isinstance(preprocessor, RowLocalPowerFallback):
            estimator.ensemble_generator_.preprocessors_[method] = RowLocalPowerFallback(preprocessor)


def fallback_counts(estimator, *, reset=False):
    counts = {}
    for method, preprocessor in estimator.ensemble_generator_.preprocessors_.items():
        if isinstance(preprocessor, RowLocalPowerFallback):
            counts[method] = preprocessor.row_fallbacks
            if reset:
                preprocessor.row_fallbacks = 0
    return counts
