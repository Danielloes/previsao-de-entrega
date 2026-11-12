"""
Suíte de Testes Automatizados (Etapa 5).

Valida:
1. Prevenção de vazamento de dados (nenhuma coluna de alvo entra em X e a separação é estritamente temporal).
2. Tratamento de inconsistências reais da DoorDash (valores negativos, divisão por zero em `busy_dashers_ratio`,
   categorias desconhecidas e outliers extremos).
3. Comportamento direcional de sanidade do modelo (maior pressão de pedidos/entregadores ocupados aumenta o tempo previsto).
4. Funcionamento da API (`/health` e `/predict`) e garantia de que a previsão funciona com ou sem LLM (inclusive se o LLM falhar).
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from src.data_processing import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    DeliveryFeaturePipeline,
    compute_target_and_clean_rows,
    temporal_train_val_test_split,
)
from src.generate_synthetic_fixture import generate_synthetic_doordash_sample
from src.models import _compute_roc_auc_and_average_precision
from src.train import run_training_pipeline
from src.api import create_app


class TestDeliveryDelaySystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_df = generate_synthetic_doordash_sample(n_rows=600, seed=123)

    def test_target_cleaning_and_outlier_filtering(self) -> None:
        cleaned, stats = compute_target_and_clean_rows(self.raw_df, is_training=True)
        # Deve ter removido o timestamp nulo e o outlier de 5 dias injetados no gerador
        self.assertGreaterEqual(stats["dropped_missing_timestamps"], 1)
        self.assertGreaterEqual(stats["dropped_extreme_duration_outliers"], 1)
        self.assertTrue((cleaned["actual_duration_min"] >= 5.0).all())
        self.assertTrue((cleaned["actual_duration_min"] <= 180.0).all())

    def test_strict_temporal_split_no_future_leakage(self) -> None:
        cleaned, _ = compute_target_and_clean_rows(self.raw_df, is_training=True)
        train_df, val_df, test_df = temporal_train_val_test_split(cleaned, 0.70, 0.15)
        self.assertLess(train_df["created_at"].max(), val_df["created_at"].min())
        self.assertLess(val_df["created_at"].max(), test_df["created_at"].min())

    def test_temporal_split_keeps_tied_timestamps_together(self) -> None:
        timestamps = pd.to_datetime(["2015-02-01"] * 6 + ["2015-02-02"] * 2 + ["2015-02-03"] * 2, utc=True)
        data = pd.DataFrame({"created_at": timestamps})
        train, val, test = temporal_train_val_test_split(data)
        self.assertEqual([len(train), len(val), len(test)], [6, 2, 2])

    def test_average_precision_uses_tied_thresholds(self) -> None:
        labels = np.array([1, 0, 0, 1, 0])
        constant_scores = np.full(5, 0.3)
        auc, average_precision = _compute_roc_auc_and_average_precision(labels, constant_scores)
        self.assertEqual(auc, 0.5)
        self.assertEqual(average_precision, 0.4)
        tied_scores = np.array([0.8, 0.8, 0.4, 0.4, 0.4])
        _, average_precision = _compute_roc_auc_and_average_precision(labels, tied_scores)
        self.assertAlmostEqual(average_precision, 0.45)

    def test_no_forbidden_columns_and_zero_division_safety(self) -> None:
        cleaned, _ = compute_target_and_clean_rows(self.raw_df, is_training=True)
        train_df, _, test_df = temporal_train_val_test_split(cleaned, 0.70, 0.15)
        pipeline = DeliveryFeaturePipeline(top_k_categories=8).fit(train_df)

        # Injetar casos extremos no teste: onshift = 0, busy = -10, categoria inédita
        edge_df = test_df.iloc[:3].copy()
        edge_df.loc[edge_df.index[0], "total_onshift_dashers"] = 0.0
        edge_df.loc[edge_df.index[0], "total_busy_dashers"] = 15.0
        edge_df.loc[edge_df.index[1], "total_onshift_dashers"] = -20.0
        edge_df.loc[edge_df.index[2], "store_primary_category"] = "unseen_mars_cuisine"

        X_raw = pipeline.transform(edge_df, standardize=False)
        X_scaled = pipeline.transform(edge_df, standardize=True)

        # Nenhuma coluna proibida pode estar em X
        for col in FORBIDDEN_LEAKAGE_COLUMNS:
            self.assertNotIn(col, X_raw.columns)
        self.assertNotIn("category_expected_prep_min", X_raw.columns)

        # Nenhum NaN ou Inf pode restar após o transform
        self.assertFalse(np.isnan(X_raw.to_numpy()).any())
        self.assertFalse(np.isinf(X_raw.to_numpy()).any())
        self.assertFalse(np.isnan(X_scaled.to_numpy()).any())
        self.assertFalse(np.isinf(X_scaled.to_numpy()).any())

    def test_model_directional_sanity_and_api_with_llm_fallback(self) -> None:
        asyncio.run(self.check_model_and_api())

    async def check_model_and_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report = run_training_pipeline(
                self.raw_df,
                "synthetic_ci_fixture",
                tmp_path,
            )
            self.assertIn("test_metrics_baseline", report)
            self.assertIn("test_metrics_ridge_model", report)
            self.assertGreater(report["test_all_positive_durations"]["rows"], 0)
            cleaned, _ = compute_target_and_clean_rows(self.raw_df)
            train, val, test = temporal_train_val_test_split(cleaned)
            split = report["temporal_split"]
            self.assertEqual(split["train_rows"], int((train.actual_delivery_time < val.created_at.min()).sum()))
            self.assertEqual(split["val_rows"], int((val.actual_delivery_time < test.created_at.min()).sum()))

            # Testar API com um LLM que falha (simula indisponibilidade de rede/API)
            def broken_llm(_prompt: str) -> str:
                raise RuntimeError("Simulated LLM timeout")

            app = create_app(bundle_path=tmp_path / "model_bundle.pkl", llm_callable=broken_llm)

            # 2. Pedido com baixa saturação vs. alta saturação
            low_load_order = {
                "market_id": 2,
                "created_at": "2015-02-06 14:30:00",
                "store_id": 120,
                "store_primary_category": "sandwiches",
                "order_protocol": 1,
                "total_items": 1,
                "subtotal": 1200,
                "num_distinct_items": 1,
                "min_item_price": 1200,
                "max_item_price": 1200,
                "total_onshift_dashers": 50,
                "total_busy_dashers": 15,
                "total_outstanding_orders": 15,
                "estimated_order_place_duration": 250,
                "estimated_store_to_consumer_driving_duration": 480,
                "use_llm_explanation": True,
            }
            high_load_order = dict(low_load_order)
            high_load_order.update({
                "total_items": 8,
                "total_onshift_dashers": 15,
                "total_busy_dashers": 18,
                "total_outstanding_orders": 45,
                "estimated_store_to_consumer_driving_duration": 1200,
            })

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                health = await client.get("/health")
                dashboard = await client.get("/")
                res_low = await client.post("/predict", json=low_load_order)
                res_high = await client.post("/predict", json=high_load_order)
                invalid = await client.post("/predict", json={"created_at": "data-invalida"})
                offset = await client.post("/predict", json={**low_load_order, "created_at": "2015-02-06T06:30:00-08:00"})
            self.assertEqual(health.status_code, 200)
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn('fetch("/predict"', dashboard.text)
            self.assertNotIn("MODEL_BUNDLE", dashboard.text)
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(res_low.status_code, 200)
            self.assertEqual(res_high.status_code, 200)
            self.assertEqual(offset.json(), res_low.json())

            data_low = res_low.json()
            data_high = res_high.json()

            # A previsão sob alta saturação e maior distância deve ser maior que sob baixa carga
            self.assertGreater(data_high["predicted_duration_min"], data_low["predicted_duration_min"])
            self.assertGreater(data_high["delay_risk_score"], data_low["delay_risk_score"])

            # Mesmo com `use_llm_explanation=True` e `broken_llm`, o fallback determinístico garante resposta
            self.assertEqual(data_low["explanation_mode"], "deterministic_fallback")
            self.assertEqual(len(data_low["top_influencing_factors"]), 5)
            self.assertEqual(data_low["model_data_source"], health.json()["data_source"])
            self.assertIn("model_base_duration_min", data_low)


if __name__ == "__main__":
    unittest.main()
