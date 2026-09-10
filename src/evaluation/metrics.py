from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


class EvaluationMetricsCalculator:
    @staticmethod
    def calculate_intent_metrics(gold_intents: List[str], pred_intents: List[str]) -> Dict[str, Any]:
        if not gold_intents or not pred_intents:
            return {}

        acc = float(accuracy_score(gold_intents, pred_intents))
        macro_f1 = float(f1_score(gold_intents, pred_intents, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(gold_intents, pred_intents, average="weighted", zero_division=0))

        labels = sorted(list(set(gold_intents + pred_intents)))
        precisions = precision_score(gold_intents, pred_intents, labels=labels, average=None, zero_division=0)
        recalls = recall_score(gold_intents, pred_intents, labels=labels, average=None, zero_division=0)

        per_intent = {}
        for lbl, p, r in zip(labels, precisions, recalls):
            per_intent[lbl] = {"precision": round(float(p), 4), "recall": round(float(r), 4)}

        unknown_count = sum(1 for p in pred_intents if p == "unknown_other")
        unknown_rate = unknown_count / len(pred_intents) if pred_intents else 0.0

        cm = confusion_matrix(gold_intents, pred_intents, labels=labels).tolist()

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "unknown_rate": round(unknown_rate, 4),
            "labels": labels,
            "per_intent": per_intent,
            "confusion_matrix": cm
        }

    @staticmethod
    def calculate_escalation_metrics(gold_decisions: List[str], pred_decisions: List[str]) -> Dict[str, Any]:
        if not gold_decisions or not pred_decisions:
            return {}

        # Normalize strings
        g_norm = [g.upper() for g in gold_decisions]
        p_norm = [p.upper() for p in pred_decisions]

        acc = float(accuracy_score(g_norm, p_norm))
        f1 = float(f1_score(g_norm, p_norm, pos_label="ESCALATE", average="binary", zero_division=0))
        prec = float(precision_score(g_norm, p_norm, pos_label="ESCALATE", average="binary", zero_division=0))
        rec = float(recall_score(g_norm, p_norm, pos_label="ESCALATE", average="binary", zero_division=0))

        # Unsafe auto-handling: gold = ESCALATE, system = AUTO_HANDLE
        unsafe_count = sum(1 for g, p in zip(g_norm, p_norm) if g == "ESCALATE" and p == "AUTO_HANDLE")
        gold_escalate_count = sum(1 for g in g_norm if g == "ESCALATE")
        unsafe_rate = unsafe_count / len(g_norm) if g_norm else 0.0

        # Unnecessary escalation: gold = AUTO_HANDLE, system = ESCALATE
        unnecessary_count = sum(1 for g, p in zip(g_norm, p_norm) if g == "AUTO_HANDLE" and p == "ESCALATE")
        unnecessary_rate = unnecessary_count / len(g_norm) if g_norm else 0.0

        labels = ["AUTO_HANDLE", "ESCALATE"]
        cm = confusion_matrix(g_norm, p_norm, labels=labels).tolist()

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "unsafe_auto_handling_count": unsafe_count,
            "unsafe_auto_handling_rate": round(unsafe_rate, 4),
            "unnecessary_escalation_count": unnecessary_count,
            "unnecessary_escalation_rate": round(unnecessary_rate, 4),
            "confusion_matrix": cm
        }
