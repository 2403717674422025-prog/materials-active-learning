from types import SimpleNamespace

import pytest

from backend.app.services.experiment_service import _predictions_to_persist


def _prediction(index):
    return SimpleNamespace(
        candidate_id=f"candidate-{index}",
        pool_index=index,
        predicted_mean=float(index),
        predicted_std=float(index + 1),
    )


@pytest.mark.parametrize("selected_index", [499, 500, 501, 700])
def test_acquired_candidate_prediction_persists_beyond_pool_limit(selected_index):
    predictions = [_prediction(i) for i in range(750)]

    persisted = _predictions_to_persist(
        predictions, [f"candidate-{selected_index}"]
    )
    persisted_by_id = {prediction.candidate_id: prediction for prediction in persisted}

    assert f"candidate-{selected_index}" in persisted_by_id
    assert persisted_by_id[f"candidate-{selected_index}"].pool_index == selected_index
    assert persisted_by_id[f"candidate-{selected_index}"].predicted_mean == float(selected_index)
    assert persisted_by_id[f"candidate-{selected_index}"].predicted_std == float(selected_index + 1)
    expected_count = 500 if selected_index < 500 else 501
    assert len(persisted) == expected_count


def test_acquired_predictions_overlap_pool_sample_without_duplicates():
    predictions = [_prediction(i) for i in range(520)]

    persisted = _predictions_to_persist(
        predictions, ["candidate-10", "candidate-500", "candidate-519"]
    )

    ids = [prediction.candidate_id for prediction in persisted]
    assert len(ids) == len(set(ids)) == 502
    assert {"candidate-10", "candidate-500", "candidate-519"}.issubset(ids)


def test_selected_candidate_without_prediction_fails_instead_of_disappearing():
    with pytest.raises(ValueError, match="no pre-acquisition prediction"):
        _predictions_to_persist([_prediction(0)], ["not-in-pool"])
