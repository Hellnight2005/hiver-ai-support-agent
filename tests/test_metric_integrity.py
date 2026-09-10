import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.config import settings
from src.api.main import app
from src.evaluation.metrics import EvaluationMetricsCalculator


def test_evaluation_artifact_exists_and_valid():
    eval_path = Path(settings.paths.evaluation_results)
    assert eval_path.exists(), f"Evaluation results artifact {eval_path} must exist"

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "Evaluation results must be a dictionary"
    required_systems = ["Majority", "TF-IDF", "Embedding Retrieval", "AI Agent"]
    for s in required_systems:
        assert s in data, f"System '{s}' must be present in evaluation results"
        assert "intent_metrics" in data[s]
        assert "escalation_metrics" in data[s]
        assert "records" in data[s]


def test_metric_value_ranges():
    eval_path = Path(settings.paths.evaluation_results)
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for s, res in data.items():
        im = res["intent_metrics"]
        em = res["escalation_metrics"]

        assert 0.0 <= im["accuracy"] <= 1.0
        assert 0.0 <= im["macro_f1"] <= 1.0
        assert 0.0 <= im["weighted_f1"] <= 1.0
        assert 0.0 <= em["accuracy"] <= 1.0
        assert 0.0 <= em["f1"] <= 1.0
        assert 0.0 <= em["unsafe_auto_handling_rate"] <= 1.0
        assert 0.0 <= em["unnecessary_escalation_rate"] <= 1.0


def test_safety_metric_is_derived_from_records():
    eval_path = Path(settings.paths.evaluation_results)
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agent_records = data["AI Agent"]["records"]
    assert len(agent_records) > 0, "AI Agent must have evaluation records"

    actual_unsafe = sum(
        1 for r in agent_records
        if r.get("gold_escalation", "").upper() == "ESCALATE"
        and r.get("pred_escalation", "").upper() == "AUTO_HANDLE"
    )
    reported_count = data["AI Agent"]["escalation_metrics"]["unsafe_auto_handling_count"]
    reported_rate = data["AI Agent"]["escalation_metrics"]["unsafe_auto_handling_rate"]

    assert actual_unsafe == reported_count
    assert round(actual_unsafe / len(agent_records), 4) == reported_rate


def test_report_consistency_with_artifact():
    eval_path = Path(settings.paths.evaluation_results)
    report_path = Path("reports/evaluation.md")

    if not report_path.exists():
        pytest.skip("reports/evaluation.md not yet generated")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    agent_acc = eval_data["AI Agent"]["intent_metrics"]["accuracy"]
    agent_acc_str = f"{agent_acc * 100:.1f}%"
    assert agent_acc_str in report_text, f"Report must contain actual agent accuracy {agent_acc_str}"


def test_api_evaluation_endpoint():
    client = TestClient(app)
    response = client.get("/v1/evaluation")
    assert response.status_code == 200

    data = response.json()
    assert "AI Agent" in data
    assert "intent_metrics" in data["AI Agent"]
    assert "escalation_metrics" in data["AI Agent"]


def test_deterministic_evaluation_fixture():
    # Verify metric calculator produces exact expected mathematical output
    g_intents = ["delivery_issue", "refund_request", "account_access", "delivery_issue"]
    p_intents = ["delivery_issue", "refund_request", "unknown_other", "delivery_issue"]

    metrics = EvaluationMetricsCalculator.calculate_intent_metrics(g_intents, p_intents)
    assert metrics["accuracy"] == 0.75  # 3/4
    assert metrics["unknown_rate"] == 0.25  # 1/4

    g_dec = ["AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE", "ESCALATE"]
    p_dec = ["AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"]

    em = EvaluationMetricsCalculator.calculate_escalation_metrics(g_dec, p_dec)
    assert em["accuracy"] == 0.50
    assert em["unsafe_auto_handling_count"] == 1  # idx 2: gold=ESCALATE, pred=AUTO_HANDLE
    assert em["unnecessary_escalation_count"] == 1  # idx 1: gold=AUTO_HANDLE, pred=ESCALATE
    assert em["unsafe_auto_handling_rate"] == 0.25
    assert em["unnecessary_escalation_rate"] == 0.25
