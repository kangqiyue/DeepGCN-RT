import hashlib
import math

import pytest

from dataset import smiles2graph
from inference import DEFAULT_MODEL_PATH, predict_retention_time


EXPECTED_CHECKPOINT_SHA256 = (
    "195135fe104a5da90189130e6cceeeee1aeb115be21d195d597ced2a0d766677"
)


def test_released_checkpoint_checksum():
    digest = hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_CHECKPOINT_SHA256


def test_smiles_graph_has_bidirectional_bonds():
    graph = smiles2graph("CCO")
    assert graph["num_nodes"] == 3
    assert graph["edge_index"].shape == (2, 4)
    assert graph["edge_feat"].shape[0] == 4


@pytest.mark.parametrize("smiles", ["", "   ", "not-a-smiles", None])
def test_invalid_smiles_is_rejected(smiles):
    with pytest.raises(ValueError, match="SMILES|Invalid"):
        smiles2graph(smiles)


def test_released_checkpoint_cpu_inference():
    prediction = predict_retention_time("CCO", device="cpu")
    assert math.isfinite(prediction)
    assert prediction == pytest.approx(624.8956, abs=0.01)
