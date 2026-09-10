import pytest
from src.evaluation.metrics import EvaluationMetricsCalculator
from src.evaluation.human_agreement import HumanJudgeAgreementCalculator


def test_intent_metrics_calculation():
    golds = ["delivery_issue", "refund_request", "account_access"]
    preds = ["delivery_issue", "refund_request", "delivery_issue"]

    metrics = EvaluationMetricsCalculator.calculate_intent_metrics(golds, preds)
    assert metrics["accuracy"] == round(2.0 / 3.0, 4)
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics


def test_escalation_metrics_calculation():
    golds = ["AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"]
    preds = ["AUTO_HANDLE", "ESCALATE", "ESCALATE", "AUTO_HANDLE"]

    metrics = EvaluationMetricsCalculator.calculate_escalation_metrics(golds, preds)
    assert metrics["accuracy"] == 0.50
    assert metrics["unsafe_auto_handling_count"] == 1
    assert metrics["unnecessary_escalation_count"] == 1


def test_human_judge_agreement_calculation():
    human = [4.0, 5.0, 3.0, 2.0, 5.0]
    judge = [4.0, 4.8, 3.2, 2.0, 4.9]

    ag = HumanJudgeAgreementCalculator.calculate_agreement(human, judge)
    assert ag["sample_size"] == 5
    assert ag["pearson_corr"] > 0.90
    assert ag["mae"] < 0.20
    assert ag["within_1pt_agreement_pct"] == 100.0
