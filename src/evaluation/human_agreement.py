from typing import List, Dict, Any
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


class HumanJudgeAgreementCalculator:
    @staticmethod
    def calculate_agreement(human_scores: List[float], judge_scores: List[float]) -> Dict[str, Any]:
        if not human_scores or not judge_scores or len(human_scores) != len(judge_scores):
            return {
                "sample_size": 0,
                "pearson_corr": 0.0,
                "spearman_corr": 0.0,
                "mae": 0.0,
                "exact_agreement_pct": 0.0,
                "within_1pt_agreement_pct": 0.0,
                "cohen_kappa": 0.0
            }

        h_arr = np.array(human_scores, dtype=float)
        j_arr = np.array(judge_scores, dtype=float)

        # MAE
        mae = float(np.mean(np.abs(h_arr - j_arr)))

        # Handle zero variance constant arrays to avoid NaN in correlation
        h_std = np.std(h_arr)
        j_std = np.std(j_arr)

        if h_std == 0 or j_std == 0:
            # If scores match closely, correlation is effectively perfect
            pearson_val = 1.0 if mae < 0.5 else 0.0
            spearman_val = 1.0 if mae < 0.5 else 0.0
        else:
            pearson_val, _ = pearsonr(h_arr, j_arr)
            spearman_val, _ = spearmanr(h_arr, j_arr)

        # Exact and Within +/- 1 point agreement
        exact = float(np.mean(np.round(h_arr) == np.round(j_arr))) * 100.0
        within_1pt = float(np.mean(np.abs(h_arr - j_arr) <= 1.0)) * 100.0

        # Cohen's Kappa on binary pass/fail (score >= 4.0 is pass)
        h_pass = [1 if s >= 4.0 else 0 for s in h_arr]
        j_pass = [1 if s >= 4.0 else 0 for s in j_arr]
        
        try:
            if len(set(h_pass + j_pass)) <= 1:
                kappa = 1.0
            else:
                kappa = float(cohen_kappa_score(h_pass, j_pass))
        except Exception:
            kappa = 0.0

        return {
            "sample_size": len(human_scores),
            "pearson_corr": round(float(pearson_val), 4),
            "spearman_corr": round(float(spearman_val), 4),
            "mae": round(mae, 4),
            "exact_agreement_pct": round(exact, 2),
            "within_1pt_agreement_pct": round(within_1pt, 2),
            "cohen_kappa": round(kappa, 4)
        }
