"""
Módulo de Explicação em Linguagem Natural Restrita aos Dados do Modelo (Etapa 4).

Regras de Engenharia:
1. O sistema NUNCA depende do LLM para calcular a previsão ou os fatores de influência.
2. Se um cliente LLM for injetado na fábrica da API, o prompt recebe
   EXCLUSIVAMENTE o JSON de entrada e os fatores matemáticos retornados pelo modelo,
   com instrução estrita de não inventar causas externas (como chuva ou acidente não informados).
3. Se nenhum LLM estiver configurado ou se a chamada falhar, o sistema utiliza o gerador
   determinístico embutido (`generate_deterministic_explanation`).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


STRICT_LLM_SYSTEM_PROMPT = """Você é um assistente técnico de operações logísticas.
Sua única função é resumir em português claro, em no máximo 3 frases, a previsão de entrega gerada pelo modelo estatístico.
REGRAS OBRIGATÓRIAS:
1. Use SOMENTE os valores numéricos e fatores presentes no JSON fornecido.
2. NUNCA invente causas externas que não estejam nos dados (não mencione clima, chuva, acidentes ou feriados).
3. Cite explicitamente o tempo previsto em minutos, o ETA de referência, o risco de atraso (%) e os 2 principais fatores com seus impactos em minutos."""


def build_grounded_llm_prompt(
    order_input: Dict[str, Any],
    prediction_result: Dict[str, Any],
) -> str:
    """Monta o payload estritamente ancorado nos dados de entrada e saída do modelo."""
    payload = {
        "dados_entrada_pedido": {
            "total_items": order_input.get("total_items"),
            "subtotal_cents": order_input.get("subtotal"),
            "store_primary_category": order_input.get("store_primary_category"),
            "total_onshift_dashers": order_input.get("total_onshift_dashers"),
            "total_busy_dashers": order_input.get("total_busy_dashers"),
            "total_outstanding_orders": order_input.get("total_outstanding_orders"),
        },
        "saida_do_modelo": {
            "predicted_duration_min": prediction_result["predicted_duration_min"],
            "reference_eta_min": prediction_result["reference_eta_min"],
            "delay_risk_probability": prediction_result["delay_risk_score"],
            "is_high_risk": prediction_result["is_high_risk"],
            "top_influencing_factors": prediction_result["top_influencing_factors"][:3],
        },
    }
    return (
        f"{STRICT_LLM_SYSTEM_PROMPT}\n\n"
        f"JSON_DO_MODELO:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def generate_deterministic_explanation(prediction_result: Dict[str, Any]) -> str:
    """
    Gera uma explicação textual determinística e auditável usando apenas os números
    retornados pelo modelo (funciona 100% offline sem LLM).
    """
    pred_min = prediction_result["predicted_duration_min"]
    ref_min = prediction_result["reference_eta_min"]
    risk_pct = round(prediction_result["delay_risk_score"] * 100, 1)
    diff_min = round(pred_min - ref_min, 1)
    risk_label = "ALTO" if prediction_result["is_high_risk"] else "BAIXO/MODERADO"

    factors: List[Dict[str, Any]] = prediction_result.get("top_influencing_factors", [])
    top_descriptions = []
    for f in factors[:3]:
        sign = "+" if f["impact_minutes"] >= 0 else ""
        top_descriptions.append(
            f"{f['description_pt']} (valor={f['raw_value']}, impacto={sign}{f['impact_minutes']} min)"
        )
    factors_str = "; ".join(top_descriptions) if top_descriptions else "sem fatores adicionais destacados"

    comparison_str = f"igual ao prazo de referência ({ref_min} min)"
    if diff_min != 0:
        position = "acima" if diff_min > 0 else "abaixo"
        comparison_str = f"{abs(diff_min)} min {position} do prazo de referência ({ref_min} min)"

    return (
        f"Risco {risk_label} de ultrapassar a referência ({risk_pct}%): o tempo estimado é de "
        f"{pred_min} min ({comparison_str}). "
        f"Principais contribuições para a previsão linear, em relação à base do modelo: {factors_str}."
    )


class OrderPredictionExplainer:
    """
    Camada de explicação que suporta um cliente LLM injetável (opcional),
    com fallback determinístico caso o LLM não esteja
    ativo ou falhe.
    """
    def __init__(self, llm_callable: Optional[Callable[[str], str]] = None) -> None:
        self.llm_callable = llm_callable

    def explain(
        self,
        order_input: Dict[str, Any],
        prediction_result: Dict[str, Any],
        use_llm: bool = False,
    ) -> Dict[str, str]:
        if use_llm and self.llm_callable is not None:
            prompt = build_grounded_llm_prompt(order_input, prediction_result)
            try:
                llm_text = self.llm_callable(prompt)
                if llm_text and llm_text.strip():
                    return {
                        "explanation_text": llm_text.strip(),
                        "explanation_mode": "llm_optional",
                    }
            except Exception:
                # Fallback transparente: a previsão jamais falha se o LLM cair
                pass

        return {
            "explanation_text": generate_deterministic_explanation(prediction_result),
            "explanation_mode": "deterministic_fallback",
        }
