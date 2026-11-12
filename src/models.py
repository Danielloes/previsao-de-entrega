"""
Módulo de Modelagem, Baseline, Explicabilidade e Avaliação (Etapa 3).

Implementa:
1. OperationalBaselineModel: Baseline simples baseado nas estimativas de rota/pedido
   da plataforma + tempo adicional histórico por categoria no treino.
2. RidgeETAModel: Modelo supervisionado com regularização L2 (solução analítica exata via SciPy),
   estimativa de risco por resíduos de validação e contribuições locais em minutos.
3. Métricas de erro, ranking e probabilidade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import linalg, stats


# Mapa de nomes amigáveis em português para apresentação de explicabilidade
FEATURE_LABELS_PT = {
    "estimated_driving_min": "Tempo estimado de trânsito (loja -> cliente)",
    "estimated_order_place_min": "Tempo estimado de transmissão do pedido",
    "estimated_non_prep_min": "Soma das estimativas de rota e transmissão",
    "orders_per_onshift_dasher": "Pressão de demanda (pedidos ativos por entregador no turno)",
    "busy_dashers_ratio": "Taxa de ocupação dos entregadores na região",
    "total_outstanding_orders": "Total de pedidos em andamento num raio de 10 milhas",
    "total_busy_dashers": "Total de entregadores ocupados na região",
    "total_onshift_dashers": "Total de entregadores ativos no turno",
    "free_dashers_estimate": "Estimativa de entregadores livres na região",
    "is_dasher_overcapacity": "Alerta de sobrecarga (entregadores ocupados > turno)",
    "total_items": "Quantidade total de itens no pedido",
    "num_distinct_items": "Quantidade de itens distintos no cardápio",
    "subtotal_dollars": "Valor subtotal do pedido (US$)",
    "avg_item_price_dollars": "Preço médio por item (US$)",
    "price_range_dollars": "Amplitude de preço entre itens do pedido (US$)",
    "distinct_item_ratio": "Proporção de itens distintos sobre o total",
    "is_peak_dinner": "Horário de pico do jantar (18h às 21h)",
    "is_peak_lunch": "Horário de pico do almoço (11h às 14h)",
    "is_weekend": "Pedido realizado no fim de semana",
    "hour_sin": "Componente cíclico de horário (seno)",
    "hour_cos": "Componente cíclico de horário (cosseno)",
    "is_missing_dasher_info": "Indicador de dados de entregadores ausentes na praça",
}


@dataclass
class OperationalBaselineModel:
    """
    Baseline heurístico simples:
    Usa a referência sem a margem como previsão de duração. Para o risco,
    retorna a frequência de ultrapassagem observada no treino.
    """
    delay_rate_: float = 0.0

    def fit(self, df_train: pd.DataFrame, y_train_min: np.ndarray, pipeline: Any) -> "OperationalBaselineModel":
        reference = pipeline.compute_reference_eta_min(df_train).to_numpy()
        self.delay_rate_ = float(np.mean(y_train_min > reference))
        return self

    def predict_duration_min(self, df: pd.DataFrame, pipeline: Any) -> np.ndarray:
        # Baseline prevê o ETA operacional sem o buffer de SLA
        ref_eta = pipeline.compute_reference_eta_min(df).to_numpy(dtype=np.float64)
        return ref_eta - pipeline.sla_buffer_min

    def predict_delay_probability(self, df: pd.DataFrame, pipeline: Any) -> np.ndarray:
        return np.full(len(df), self.delay_rate_)


@dataclass
class RidgeETAModel:
    """
    Modelo supervisionado interpretável (Regressão Ridge com regularização L2)
    treinado nas features padronizadas, com estimativa de risco a partir da
    distribuição dos erros em validação e contribuições para a previsão linear.
    """
    alpha: float = 50.0
    intercept_: float = 0.0
    weights_: Optional[np.ndarray] = None
    residual_std_min_: float = 10.0
    validation_residuals_: Optional[np.ndarray] = None
    feature_names_: List[str] = field(default_factory=list)

    def fit(self, X_scaled: pd.DataFrame, y_train_min: np.ndarray) -> "RidgeETAModel":
        self.feature_names_ = X_scaled.columns.tolist()
        X_mat = X_scaled.to_numpy(dtype=np.float64)
        y_vec = np.asarray(y_train_min, dtype=np.float64)

        self.intercept_ = float(np.mean(y_vec))
        y_centered = y_vec - self.intercept_

        n_features = X_mat.shape[1]
        # Solução analítica fechada: (X^T X + alpha * I) w = X^T y_centered
        gram = X_mat.T @ X_mat + self.alpha * np.eye(n_features)
        rhs = X_mat.T @ y_centered
        self.weights_ = linalg.solve(gram, rhs, assume_a="pos")

        preds = self.predict_duration_min(X_scaled)
        residuals = y_vec - preds
        self.residual_std_min_ = max(float(np.std(residuals)), 1.0)
        self.validation_residuals_ = None
        return self

    def fit_risk(self, X_val_scaled: pd.DataFrame, y_val_min: np.ndarray) -> None:
        """Aprende a distribuição empírica dos erros fora do treino."""
        if len(X_val_scaled) == 0:
            raise ValueError("A estimativa de risco exige pedidos na validação.")
        self.validation_residuals_ = np.sort(y_val_min - self.predict_duration_min(X_val_scaled))

    def predict_duration_min(self, X_scaled: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de prever.")
        X_mat = X_scaled[self.feature_names_].to_numpy(dtype=np.float64)
        preds = self.intercept_ + X_mat @ self.weights_
        return np.clip(preds, 5.0, 240.0)

    def predict_delay_probability(
        self, X_scaled: pd.DataFrame, reference_eta_min: np.ndarray
    ) -> np.ndarray:
        """Estima P(real > referência) pela fração de erros que excedem a folga."""
        if self.validation_residuals_ is None:
            raise RuntimeError("Ajuste a estimativa de risco com a janela de validação.")
        gap = np.asarray(reference_eta_min, dtype=np.float64) - self.predict_duration_min(X_scaled)
        rank = np.searchsorted(self.validation_residuals_, gap, side="right")
        return (len(self.validation_residuals_) - rank) / len(self.validation_residuals_)

    def predict_normal_delay_probability(
        self, X_scaled: pd.DataFrame, reference_eta_min: np.ndarray
    ) -> np.ndarray:
        """Versão gaussiana original, usada como comparação na avaliação."""
        pred_min = self.predict_duration_min(X_scaled)
        ref_eta = np.asarray(reference_eta_min, dtype=np.float64)
        z = (ref_eta - pred_min) / self.residual_std_min_
        probs = stats.norm.sf(z)
        return np.clip(probs, 0.001, 0.999)

    def explain_instance(
        self,
        X_scaled_row: pd.Series,
        X_raw_row: pd.Series,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retorna os `top_k` fatores que mais impactaram a previsão de um pedido específico.
        Como X_scaled está padronizado (Z-score no treino), o produto `w_j * x_scaled_j`
        mede exatamente quantos minutos a feature `j` adicionou ou reduziu em relação
        ao intercepto da previsão linear, antes do limite aplicado à saída.
        """
        if self.weights_ is None:
            raise RuntimeError("O modelo precisa ser treinado antes de explicar previsões.")

        x_vals = X_scaled_row[self.feature_names_].to_numpy(dtype=np.float64)
        contributions_min = self.weights_ * x_vals

        ranked_indices = np.argsort(np.abs(contributions_min))[::-1][:top_k]
        factors = []
        for idx in ranked_indices:
            fname = self.feature_names_[idx]
            impact = float(contributions_min[idx])
            raw_val = float(X_raw_row.get(fname, 0.0))
            factors.append({
                "feature": fname,
                "description_pt": FEATURE_LABELS_PT.get(fname, fname),
                "raw_value": round(raw_val, 3),
                "impact_minutes": round(impact, 2),
                "direction": "aumenta_tempo" if impact >= 0 else "reduz_tempo",
            })
        return factors

    def get_global_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """
        Retorna a importância global das variáveis medida pelo impacto absoluto
        em minutos por 1 desvio-padrão de variação (|w_j|).
        """
        if self.weights_ is None:
            raise RuntimeError("Modelo não treinado.")
        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "description_pt": [FEATURE_LABELS_PT.get(f, f) for f in self.feature_names_],
            "weight_minutes_per_std": np.round(self.weights_, 3),
            "abs_importance_min": np.round(np.abs(self.weights_), 3),
        })
        return df_imp.sort_values("abs_importance_min", ascending=False).head(top_k).reset_index(drop=True)


def _compute_roc_auc_and_average_precision(
    y_true: np.ndarray, y_score: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcula ROC-AUC por postos e Average Precision por limiares de pontuação.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0:
        return None, None
    if n_neg == 0:
        return None, 1.0

    # ROC-AUC via ranks
    ranks = stats.rankdata(y_score)
    sum_pos_ranks = np.sum(ranks[y_true == 1])
    roc_auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # O mesmo limiar recupera todos os pedidos empatados de uma vez.
    desc_idx = np.argsort(y_score)[::-1]
    scores_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]
    threshold_ends = np.r_[np.flatnonzero(scores_sorted[:-1] != scores_sorted[1:]), len(y_true) - 1]
    tp = np.cumsum(y_true_sorted)[threshold_ends]
    precision = tp / (threshold_ends + 1)
    recall_increment = np.diff(np.r_[0, tp]) / n_pos
    return float(roc_auc), float(np.sum(precision * recall_increment))


def evaluate_predictions(
    y_true_min: np.ndarray,
    y_pred_min: np.ndarray,
    reference_eta_min: np.ndarray,
    delay_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Optional[float]]:
    """
    Calcula métricas completas de regressão (em minutos) e de classificação
    de atraso (quando actual_duration_min > reference_eta_min).
    """
    y_true = np.asarray(y_true_min, dtype=float)
    y_pred = np.asarray(y_pred_min, dtype=float)
    ref_eta = np.asarray(reference_eta_min, dtype=float)
    probs = np.asarray(delay_prob, dtype=float)

    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    medae = float(np.median(abs_err))
    p90_err = float(np.percentile(abs_err, 90))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Alvo binário de atraso em relação ao ETA operacional de referência
    y_delayed = (y_true > ref_eta).astype(int)
    pred_delayed = (probs >= threshold).astype(int)

    tp = int(np.sum((y_delayed == 1) & (pred_delayed == 1)))
    fp = int(np.sum((y_delayed == 0) & (pred_delayed == 1)))
    fn = int(np.sum((y_delayed == 1) & (pred_delayed == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    roc_auc, average_precision = _compute_roc_auc_and_average_precision(y_delayed, probs)

    return {
        "MAE_min": round(mae, 3),
        "RMSE_min": round(rmse, 3),
        "MedAE_min": round(medae, 3),
        "P90_Abs_Error_min": round(p90_err, 3),
        "R2": round(r2, 4),
        "Delay_Rate_Actual": round(float(np.mean(y_delayed)), 4),
        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
        "Average_Precision": round(average_precision, 4) if average_precision is not None else None,
        "Brier_Score": round(float(np.mean((probs - y_delayed) ** 2)), 4),
        "Mean_Predicted_Risk": round(float(np.mean(probs)), 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
    }


def calibration_bins(y_delayed: np.ndarray, probabilities: np.ndarray) -> List[Dict[str, Any]]:
    """Compara a previsão média com a frequência observada em cada faixa."""
    bins = []
    for index in range(5):
        low, high = index / 5, (index + 1) / 5
        in_bin = (probabilities >= low) & ((probabilities <= high) if index == 4 else (probabilities < high))
        count = int(in_bin.sum())
        if count:
            bins.append({
                "from": low,
                "to": high,
                "rows": count,
                "mean_predicted": round(float(probabilities[in_bin].mean()), 4),
                "observed_rate": round(float(y_delayed[in_bin].mean()), 4),
            })
    return bins
