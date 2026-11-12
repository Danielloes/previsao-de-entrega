"""
API REST para Previsão de Tempo de Entrega, Risco de Atraso e Fatores de Influência (Etapa 4).

Recebe os dados de um pedido no momento da criação, executa o pipeline de features,
retorna a previsão em minutos, a probabilidade de atraso, os fatores que mais
influenciaram a estimativa e uma explicação textual ancorada exclusivamente no modelo.
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.explainer import OrderPredictionExplainer
from src.train import BUNDLE_VERSION, REAL_ARTIFACTS_DIR


class OrderPredictionRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    market_id: Optional[int] = Field(2, description="ID da região/cidade onde o pedido foi realizado")
    created_at: datetime = Field(description="Data e hora ISO 8601. Sem fuso, a hora é interpretada como UTC.")
    store_id: Optional[int] = Field(145, description="Identificador do restaurante")
    store_primary_category: Optional[str] = Field("pizza", description="Categoria culinária principal da loja")
    order_protocol: Optional[int] = Field(1, description="Protocolo de recebimento do pedido")
    total_items: int = Field(4, ge=1, description="Quantidade total de itens no pedido")
    subtotal: float = Field(3850.0, ge=0.0, description="Valor subtotal do pedido em centavos (ex.: 3850 = $38.50)")
    num_distinct_items: int = Field(3, ge=1, description="Quantidade de itens distintos no pedido")
    min_item_price: float = Field(450.0, ge=0.0, description="Preço do item mais barato em centavos")
    max_item_price: float = Field(1600.0, ge=0.0, description="Preço do item mais caro em centavos")
    total_onshift_dashers: Optional[float] = Field(20.0, description="Entregadores ativos no turno num raio de 10 milhas")
    total_busy_dashers: Optional[float] = Field(19.0, description="Entregadores atualmente ocupados num raio de 10 milhas")
    total_outstanding_orders: Optional[float] = Field(34.0, description="Pedidos em processamento num raio de 10 milhas")
    estimated_order_place_duration: float = Field(250.0, ge=0.0, description="Tempo estimado para transmitir o pedido (segundos)")
    estimated_store_to_consumer_driving_duration: float = Field(780.0, ge=0.0, description="Tempo estimado de rota loja -> cliente (segundos)")
    use_llm_explanation: bool = Field(False, description="Se True, tenta formatar a explicação via LLM restrito aos dados do modelo")

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, value: Any) -> datetime:
        if isinstance(value, str) and len(value) > 10 and value[10] in ("T", " "):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise ValueError("Informe data e hora no formato ISO 8601.")
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class InfluencingFactorResponse(BaseModel):
    feature: str
    description_pt: str
    raw_value: float
    impact_minutes: float
    direction: str


class OrderPredictionResponse(BaseModel):
    predicted_duration_min: float
    reference_eta_min: float
    expected_delay_excess_min: float
    delay_risk_score: float
    is_high_risk: bool
    top_influencing_factors: List[InfluencingFactorResponse]
    explanation_text: str
    explanation_mode: str
    model_data_source: str
    model_base_duration_min: float


def create_app(
    bundle_path: Optional[Path] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> FastAPI:
    app = FastAPI(
        title="Delivery Delay Risk & ETA Prediction API",
        description=(
            "API para estimativa de tempo de entrega (minutos), risco de atraso e fatores "
            "de influência baseada na estrutura do dataset DoorDash ETA Prediction."
        ),
        version="1.0.0",
    )

    configured_path = os.environ.get("ETA_MODEL_BUNDLE")
    model_path = bundle_path or (Path(configured_path) if configured_path else REAL_ARTIFACTS_DIR / "model_bundle.pkl")
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo ausente em {model_path}. Rode python -m src.train --data <caminho-do-CSV>.")

    with model_path.open("rb") as f:
        state_bundle: Dict[str, Any] = pickle.load(f)
    if state_bundle.get("version") != BUNDLE_VERSION:
        raise ValueError("Versão do modelo incompatível. Gere um novo artefato com python -m src.train.")

    explainer = OrderPredictionExplainer(llm_callable=llm_callable)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root_dashboard() -> HTMLResponse:
        """Exibe o Painel Visual Interativo de Previsão de Entrega e Risco de Atraso."""
        html_path = Path(__file__).parent / "dashboard.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {
            "status": "ok",
            "model_loaded": "true",
            "data_source": state_bundle.get("data_source_label", "unknown"),
        }

    @app.post("/predict", response_model=OrderPredictionResponse)
    def predict_order(payload: OrderPredictionRequest) -> Dict[str, Any]:
        pipeline = state_bundle["pipeline"]
        model = state_bundle["model"]

        order_dict = payload.model_dump()
        df_single = pd.DataFrame([order_dict])

        try:
            X_raw = pipeline.transform(df_single, standardize=False)
            X_scaled = pipeline.transform(df_single, standardize=True)
            ref_eta_min = float(pipeline.compute_reference_eta_min(df_single).iloc[0])
            pred_min = float(model.predict_duration_min(X_scaled)[0])
            risk_score = float(model.predict_delay_probability(X_scaled, [ref_eta_min])[0])
            factors = model.explain_instance(X_scaled.iloc[0], X_raw.iloc[0], top_k=5)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Erro ao processar pedido: {exc}") from exc

        result_core = {
            "predicted_duration_min": round(pred_min, 2),
            "reference_eta_min": round(ref_eta_min, 2),
            "expected_delay_excess_min": round(pred_min - ref_eta_min, 2),
            "delay_risk_score": round(risk_score, 4),
            "is_high_risk": bool(risk_score >= 0.50),
            "top_influencing_factors": factors,
            "model_data_source": state_bundle.get("data_source_label", "unknown"),
            "model_base_duration_min": round(model.intercept_, 2),
        }

        exp_info = explainer.explain(
            order_input=order_dict,
            prediction_result=result_core,
            use_llm=payload.use_llm_explanation,
        )
        result_core["explanation_text"] = exp_info["explanation_text"]
        result_core["explanation_mode"] = exp_info["explanation_mode"]
        return result_core

    return app
