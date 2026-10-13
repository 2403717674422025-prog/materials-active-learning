import uuid

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.models.experiment import Experiment
from backend.app.services import experiment_service
from backend.app.services.explainability_service import (
    ExplainabilityError,
    get_candidate_explanation,
    get_feature_dependence,
    get_global_explanation,
    list_explainability_candidates,
)
from backend.database import Base
from backend.ml.active_learning.engine import ExperimentConfig, ExperimentState
from backend.ml.data.preprocessing import PreprocessingConfig, fit_preprocessor
from backend.ml.models.baseline import RandomForestSurrogate
from backend.ml.models.bayesian import GaussianProcessSurrogate


@pytest.fixture
def fitted_rf_experiment():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        dataset_id = str(uuid.uuid4())
        experiment_id = str(uuid.uuid4())
        db.add(Dataset(id=dataset_id, name="shap-test", status="ready"))
        db.flush()
        db.add(Experiment(
            id=experiment_id, dataset_id=dataset_id, model_type="rf", random_seed=7,
        ))
        db.commit()

        rng = np.random.RandomState(3)
        X = pd.DataFrame(rng.normal(size=(48, 3)), columns=[
            "MagpieData mean Number", "Stoichiometry p-norm", "Valence s electrons",
        ])
        y = 3 * X.iloc[:, 0] - 1.5 * X.iloc[:, 1] + 0.2 * X.iloc[:, 2]
        preprocessor = fit_preprocessor(
            X.iloc[:32], PreprocessingConfig(random_state=7), labeled_set_size=32,
        )
        model = RandomForestSurrogate(n_estimators=32, random_state=7)
        model.fit(preprocessor.transform(X.iloc[:32]), y.iloc[:32].to_numpy())

        state = ExperimentState(
            config=ExperimentConfig(experiment_id=experiment_id, model_type="rf", random_seed=7),
            pool_X=X,
            pool_y=y,
            pool_ids=[f"pool-{i}" for i in range(len(X))],
            test_X=pd.DataFrame([[100, 100, 100]], columns=X.columns),
            test_y=pd.Series([999.0]),
            test_ids=["held-out-only"],
            labeled_indices=list(range(32)),
            model=model,
            preprocessor=preprocessor,
            current_iteration=1,
        )
        experiment_service._STATE_CACHE[experiment_id] = state
        yield db, experiment_id, state
        experiment_service._STATE_CACHE.pop(experiment_id, None)
        db.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_global_shap_uses_real_feature_names_and_pool_only(fitted_rf_experiment):
    db, experiment_id, state = fitted_rf_experiment
    # Targets must not be touched while explanation uses pool features.
    state.pool_y = object()
    state.test_y = object()

    result = get_global_explanation(db, experiment_id, sample_size=12)

    assert result["sample_size"] == 12
    assert result["model_type"] == "Random Forest"
    assert {row["feature"] for row in result["feature_importance"]}.issubset(
        set(state.preprocessor.feature_columns)
    )
    assert len(result["beeswarm_points"]) == 12 * len(result["feature_importance"])
    assert "TreeExplainer path-dependent baseline" in result["reference_method"]
    assert all(point["candidate_id"] != "held-out-only" for point in result["beeswarm_points"])
    assert "target values are not accessed" in result["feature_value_source"]


def test_individual_shap_adds_to_prediction_and_rejects_heldout(fitted_rf_experiment):
    db, experiment_id, _ = fitted_rf_experiment

    result = get_candidate_explanation(db, experiment_id, "pool-4")

    assert result["candidate_id"] == "pool-4"
    assert sum(item["shap_value"] for item in result["contributions"]) == pytest.approx(
        result["predicted_value"] - result["baseline"], abs=1e-5,
    )
    with pytest.raises(ExplainabilityError, match="active-learning pool") as error:
        get_candidate_explanation(db, experiment_id, "held-out-only")
    assert error.value.status_code == 404


def test_dependence_and_candidate_search_use_pool_features_only(fitted_rf_experiment):
    db, experiment_id, _ = fitted_rf_experiment

    result = get_feature_dependence(db, experiment_id, "Stoichiometry p-norm", sample_size=9)
    matches = list_explainability_candidates(db, experiment_id, "pool-1", limit=10)

    assert result["feature"] == "Stoichiometry p-norm"
    assert len(result["points"]) == 9
    assert all(point["candidate_id"] != "held-out-only" for point in result["points"])
    assert matches["candidates"] == ["pool-1", "pool-10", "pool-11", "pool-12", "pool-13", "pool-14", "pool-15", "pool-16", "pool-17", "pool-18"]
    with pytest.raises(ExplainabilityError, match="Feature not found"):
        get_feature_dependence(db, experiment_id, "fabricated-feature")


def test_unavailable_model_and_pca_are_reported_without_retraining(fitted_rf_experiment):
    db, experiment_id, state = fitted_rf_experiment
    original_model = state.model
    state.model = None
    with pytest.raises(ExplainabilityError, match="No trained model") as missing:
        get_global_explanation(db, experiment_id)
    assert missing.value.status_code == 409

    state.model = original_model
    state.preprocessor.pca = object()
    with pytest.raises(ExplainabilityError, match="uses PCA") as pca_error:
        get_global_explanation(db, experiment_id)
    assert pca_error.value.status_code == 422


def test_fitted_gp_uses_permutation_shap_with_training_feature_background(fitted_rf_experiment):
    db, experiment_id, state = fitted_rf_experiment
    gp = GaussianProcessSurrogate(n_restarts_optimizer=0, random_state=7)
    gp.fit(state.preprocessor.transform(state.pool_X.iloc[:32]), state.pool_y.iloc[:32].to_numpy())
    state.model = gp
    state.pool_y = object()
    state.test_y = object()

    result = get_global_explanation(db, experiment_id, sample_size=4)
    local = get_candidate_explanation(db, experiment_id, "pool-40")

    assert result["model_type"] == "Gaussian Process"
    assert result["sample_size"] == 4
    assert "Permutation SHAP" in result["reference_method"]
    assert sum(item["shap_value"] for item in local["contributions"]) == pytest.approx(
        local["predicted_value"] - local["baseline"], abs=1e-5,
    )
