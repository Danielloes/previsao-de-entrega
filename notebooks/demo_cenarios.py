"""
Demonstração Prática de Previsão e Explicabilidade (Etapa 4).
Consulta a API em execução com dois pedidos de exemplo.
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_demo(base_url: str) -> None:
    scenarios = [
        (
            "Pedido simples fora de pico",
            {
                "market_id": 1,
                "created_at": "2015-02-04 15:10:00",
                "store_id": 110,
                "store_primary_category": "sandwiches",
                "order_protocol": 1,
                "total_items": 1,
                "subtotal": 1150,
                "num_distinct_items": 1,
                "min_item_price": 1150,
                "max_item_price": 1150,
                "total_onshift_dashers": 45,
                "total_busy_dashers": 18,
                "total_outstanding_orders": 16,
                "estimated_order_place_duration": 250,
                "estimated_store_to_consumer_driving_duration": 420,
                "use_llm_explanation": False,
            },
        ),
        (
            "Jantar de sexta-feira com fila",
            {
                "market_id": 2,
                "created_at": "2015-02-06T19:45:00-08:00",
                "store_id": 145,
                "store_primary_category": "pizza",
                "order_protocol": 1,
                "total_items": 7,
                "subtotal": 6400,
                "num_distinct_items": 5,
                "min_item_price": 500,
                "max_item_price": 2200,
                "total_onshift_dashers": 16,
                "total_busy_dashers": 18,
                "total_outstanding_orders": 42,
                "estimated_order_place_duration": 440,
                "estimated_store_to_consumer_driving_duration": 1140,
                "use_llm_explanation": False,
            },
        ),
    ]

    for title, payload in scenarios:
        request = Request(
            f"{base_url.rstrip('/')}/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                result = json.load(response)
        except URLError as exc:
            raise SystemExit(f"Falha ao consultar {base_url}. Confira se a API está em execução. {exc}") from exc
        print(f"\n{title}")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Endereço base da API")
    run_demo(parser.parse_args().url)
