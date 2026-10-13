"""SHAP explanations for fitted surrogate models in an AL experiment."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np
from sqlalchemy.orm import Session

from backend.app.models.experiment import Experiment
from backend.app.services import experiment_service
from backend.ml.models.baseline import RandomForestSurrogate
from backend.ml.models.bayesian import GaussianProcessSurrogate

logger = logging.getLogger(__name__)

_EXPLAINER_CACHE: OrderedDict[tuple, object] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_MAX_CACHED_EXPLAINERS = 8
_DEFAULT_SAMPLE_SIZE = 128
_MAX_SAMPLE_SIZE = 256
_MAX_FEATURES = 20
_PERMUTATION_SAMPLE_SIZE = 24
_PERMUTATION_BACKGROUND_SIZE = 12


class ExplainabilityError(Exception):
    """Expected explanation failure safe to return to API clients."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _context(db: Session, experiment_id: str):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise ExplainabilityError("Experiment not found.", 404)

    state = (
        experiment_service._STATE_CACHE.get(experiment_id)
        or experiment_service._load_state(experiment_id)
    )
    if state is None or state.model is None or state.preprocessor is None:
        raise ExplainabilityError(
            "No trained model is available. Run an iteration before requesting explanations.",
            409,
        )
    supported = isinstance(state.model, (RandomForestSurrogate, GaussianProcessSurrogate))
    if not supported or state.model._model is None:
        raise ExplainabilityError(
            "SHAP explanations are currently available for fitted Random Forest and Gaussian Process experiments only.",
            422,
        )
    if state.preprocessor.pca is not None:
        raise ExplainabilityError(
            "This experiment uses PCA; original-feature SHAP attributions are unavailable "
            "because the model was trained on principal components.",
            422,
        )

    names = list(state.preprocessor.feature_columns)
    if not names or state.pool_X.empty or len(state.pool_ids) != len(state.pool_X):
        raise ExplainabilityError("Experiment feature data is unavailable.", 409)
    if state.model._model.n_features_in_ != len(names):
        raise ExplainabilityError("Model and feature dimensions do not match.", 422)
    return experiment, state, names


def _explainer(experiment_id: str, state):
    key = (experiment_id, state.current_iteration, id(state.model._model))
    with _CACHE_LOCK:
        cached = _EXPLAINER_CACHE.get(key)
        if cached is not None:
            _EXPLAINER_CACHE.move_to_end(key)
            return cached
        try:
            import shap

            if isinstance(state.model, RandomForestSurrogate):
                explainer = shap.TreeExplainer(
                    state.model._model,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                )
            else:
                labeled = sorted(set(state.labeled_indices))
                if not labeled:
                    raise ExplainabilityError("Training reference features are unavailable.", 409)
                positions = _sample_indices(len(labeled), _PERMUTATION_BACKGROUND_SIZE, 1729)
                raw_background = state.pool_X.iloc[[labeled[int(i)] for i in positions]]
                background = state.preprocessor.transform(raw_background)

                def predict(transformed):
                    return np.asarray(state.model.predict(np.asarray(transformed, dtype=float))[0]).reshape(-1)

                masker = shap.maskers.Independent(background, max_samples=len(background))
                explainer = shap.Explainer(predict, masker, algorithm="permutation")
                explainer._materials_background = background
        except ImportError as exc:
            logger.exception("SHAP dependency is unavailable")
            raise ExplainabilityError("SHAP explainability is unavailable on this server.", 503) from exc
        except Exception as exc:
            logger.exception("Could not create SHAP TreeExplainer")
            raise ExplainabilityError("The fitted model could not be explained.", 422) from exc

        _EXPLAINER_CACHE[key] = explainer
        while len(_EXPLAINER_CACHE) > _MAX_CACHED_EXPLAINERS:
            _EXPLAINER_CACHE.popitem(last=False)
        return explainer


def _shap_values(explainer, transformed: np.ndarray, model=None):
    try:
        if isinstance(model, GaussianProcessSurrogate):
            # Permutation SHAP is approximate; two permutations cap work for wide
            # materials descriptors while keeping global estimates reproducible.
            explanation = explainer(
                transformed,
                max_evals=2 * transformed.shape[1] + 1,
                batch_size=2 * transformed.shape[1] + 1,
            )
            values = explanation.values
            bases = np.asarray(explanation.base_values, dtype=float).reshape(-1)
        else:
            values = explainer.shap_values(transformed, check_additivity=True)
            bases = None
    except Exception as exc:
        logger.exception("SHAP computation failed")
        raise ExplainabilityError("SHAP values could not be computed for this model.", 422) from exc

    if isinstance(values, list):
        if len(values) != 1:
            raise ExplainabilityError("Only single-output regression models are supported.", 422)
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != transformed.shape or not np.isfinite(values).all():
        raise ExplainabilityError("SHAP returned incompatible feature values.", 422)
    return values, bases


def _sample_indices(length: int, sample_size: int, seed: int) -> np.ndarray:
    size = min(max(1, int(sample_size)), _MAX_SAMPLE_SIZE, length)
    if size == length:
        return np.arange(length)
    return np.sort(np.random.RandomState(seed).choice(length, size=size, replace=False))


def _baseline(explainer, model=None, bases=None) -> float:
    if isinstance(model, GaussianProcessSurrogate):
        value = np.asarray(bases if bases is not None else [], dtype=float).reshape(-1)
        if value.size and np.isfinite(value).all():
            return float(value.mean())
        return float(np.asarray(model.predict(explainer._materials_background)[0]).mean())
    value = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ExplainabilityError("The model has no finite scalar SHAP baseline.", 422)
    return float(value[0])


def get_global_explanation(
    db: Session, experiment_id: str, sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> dict:
    experiment, state, names = _context(db, experiment_id)
    if not 1 <= sample_size <= _MAX_SAMPLE_SIZE:
        raise ExplainabilityError(f"sample_size must be between 1 and {_MAX_SAMPLE_SIZE}.", 422)

    indices = _sample_indices(len(state.pool_X), sample_size, experiment.random_seed)
    raw_sample = state.pool_X.iloc[indices]
    transformed = state.preprocessor.transform(raw_sample)
    if transformed.shape[1] != len(names):
        raise ExplainabilityError("Preprocessing changed the feature dimensions.", 422)

    explainer = _explainer(experiment_id, state)
    if isinstance(state.model, GaussianProcessSurrogate):
        indices = indices[:_PERMUTATION_SAMPLE_SIZE]
        raw_sample = state.pool_X.iloc[indices]
        transformed = state.preprocessor.transform(raw_sample)
    values, bases = _shap_values(explainer, transformed, state.model)
    importance = np.abs(values).mean(axis=0)
    ranked = np.argsort(importance)[::-1]
    beeswarm_features = ranked[:_MAX_FEATURES]
    feature_rows = [
        {"feature": names[i], "mean_abs_shap": float(importance[i])}
        for i in ranked
    ]
    beeswarm = [
        {
            "candidate_id": state.pool_ids[int(row)],
            "feature": names[int(col)],
            "feature_value": float(raw_sample.iloc[row, col]),
            "shap_value": float(values[row, col]),
        }
        for col in beeswarm_features for row in range(len(indices))
    ]
    return {
        "experiment_id": experiment_id,
        "model_type": "Random Forest" if isinstance(state.model, RandomForestSurrogate) else "Gaussian Process",
        "model_iteration": state.current_iteration,
        "baseline": _baseline(explainer, state.model, bases),
        "sample_size": len(indices),
        "total_features": len(names),
        "feature_importance": feature_rows,
        "beeswarm_points": beeswarm,
        "interpretation": (
            "Positive SHAP values raise the model prediction above its baseline; "
            "negative SHAP values lower it. Attributions describe model behavior, "
            "not causal effects."
        ),
        "feature_value_source": "Active-learning pool features only; target values are not accessed.",
        "reference_method": (
            "TreeExplainer path-dependent baseline from fitted Random Forest training statistics."
            if isinstance(state.model, RandomForestSurrogate)
            else "Permutation SHAP with a deterministic background sample of labeled training features; approximation, not a calibration claim."
        ),
    }


def get_candidate_explanation(db: Session, experiment_id: str, candidate_id: str) -> dict:
    experiment, state, names = _context(db, experiment_id)
    try:
        pool_index = state.pool_ids.index(candidate_id)
    except ValueError as exc:
        raise ExplainabilityError("Candidate not found in this experiment's active-learning pool.", 404) from exc

    raw_row = state.pool_X.iloc[[pool_index]]
    transformed = state.preprocessor.transform(raw_row)
    explainer = _explainer(experiment_id, state)
    values, bases = _shap_values(explainer, transformed, state.model)
    values = values[0]
    baseline = _baseline(explainer, state.model, bases)
    predicted = float(state.model.predict(transformed)[0][0])
    contributions = [
        {
            "feature": names[i],
            "feature_value": float(raw_row.iloc[0, i]),
            "shap_value": float(values[i]),
        }
        for i in range(len(names))
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    positive = sorted(
        (item for item in contributions if item["shap_value"] > 0),
        key=lambda item: item["shap_value"], reverse=True,
    )[:5]
    negative = sorted(
        (item for item in contributions if item["shap_value"] < 0),
        key=lambda item: item["shap_value"],
    )[:5]
    interpretation = (
        f"The model predicts {predicted:.6g} from a baseline of {baseline:.6g}. "
    )
    if positive:
        interpretation += f"Largest upward contribution: {positive[0]['feature']}. "
    if negative:
        interpretation += f"Largest downward contribution: {negative[0]['feature']}. "
    interpretation += "These are model attributions, not causal effects."

    return {
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "model_iteration": state.current_iteration,
        "predicted_value": predicted,
        "baseline": baseline,
        "contributions": contributions,
        "largest_positive": positive,
        "largest_negative": negative,
        "interpretation": interpretation,
    }


def get_feature_dependence(
    db: Session, experiment_id: str, feature: str, sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> dict:
    experiment, state, names = _context(db, experiment_id)
    if feature not in names:
        raise ExplainabilityError("Feature not found in the model's input features.", 404)
    if not 1 <= sample_size <= _MAX_SAMPLE_SIZE:
        raise ExplainabilityError(f"sample_size must be between 1 and {_MAX_SAMPLE_SIZE}.", 422)

    indices = _sample_indices(len(state.pool_X), sample_size, experiment.random_seed)
    raw_sample = state.pool_X.iloc[indices]
    transformed = state.preprocessor.transform(raw_sample)
    values, _ = _shap_values(_explainer(experiment_id, state), transformed, state.model)
    feature_index = names.index(feature)
    points = [
        {
            "candidate_id": state.pool_ids[int(row)],
            "feature_value": float(raw_sample.iloc[row, feature_index]),
            "shap_value": float(values[row, feature_index]),
        }
        for row in range(len(indices))
    ]
    return {
        "experiment_id": experiment_id,
        "feature": feature,
        "sample_size": len(points),
        "points": points,
    }


def list_explainability_candidates(
    db: Session, experiment_id: str, query: str = "", limit: int = 50,
) -> dict:
    _, state, _ = _context(db, experiment_id)
    if not 1 <= limit <= 100:
        raise ExplainabilityError("limit must be between 1 and 100.", 422)
    q = query.casefold().strip()
    matches = [candidate_id for candidate_id in state.pool_ids if q in candidate_id.casefold()]
    return {
        "experiment_id": experiment_id,
        "candidates": matches[:limit],
        "total_matches": len(matches),
    }
