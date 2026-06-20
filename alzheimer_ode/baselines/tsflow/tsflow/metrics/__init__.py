try:
	from .linear_pred_score import linear_pred_score
except ImportError:
	linear_pred_score = None

from .wasserstein_distance import wasserstein

__all__ = ["wasserstein"] + (["linear_pred_score"] if linear_pred_score is not None else [])
