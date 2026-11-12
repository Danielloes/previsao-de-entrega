"""
Gerador de Amostra Sintética para Testes Automatizados e Execução Offline (CI/CD).

AVISO IMPORTANTE DE TRANSPARÊNCIA:
Este arquivo gera dados sintéticos APENAS para testar o pipeline de código, os testes
unitários (pytest/unittest) e a API localmente sem exigir o download imediato dos
197.428 registros do Kaggle. Dados sintéticos refletem somente as equações programadas
abaixo e NÃO representam o comportamento real das operações de entrega.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_doordash_sample(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Gera um DataFrame com o mesmo schema e tipos de inconsistências do arquivo
    `historical_data.csv` do dataset DoorDash ETA Prediction (inclusive nulos e ruídos).
    """
    rng = np.random.default_rng(seed)

    base_ts = pd.Timestamp("2015-01-20 00:00:00", tz="UTC")
    # Pedidos distribuídos ao longo de 14 dias em ordem cronológica aproximada
    offsets_sec = np.sort(rng.integers(0, 14 * 24 * 3600, size=n_rows))
    created_at = [base_ts + pd.Timedelta(seconds=int(s)) for s in offsets_sec]

    categories = ["american", "pizza", "mexican", "chinese", "fast", "italian", "thai", "japanese", "sandwiches", None]
    cat_probs = [0.18, 0.16, 0.14, 0.10, 0.10, 0.08, 0.06, 0.06, 0.07, 0.05]
    store_cats = rng.choice(categories, size=n_rows, p=cat_probs)

    market_ids = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan], size=n_rows, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.08, 0.02])
    protocols = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], size=n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02])
    store_ids = rng.integers(100, 260, size=n_rows)

    total_items = rng.integers(1, 10, size=n_rows)
    num_distinct = np.maximum(1, total_items - rng.integers(0, 3, size=n_rows))
    min_price = rng.integers(300, 1500, size=n_rows)
    max_price = min_price + rng.integers(0, 2500, size=n_rows)
    subtotal = total_items * ((min_price + max_price) // 2)

    onshift = rng.integers(0, 80, size=n_rows).astype(float)
    # Inserir alguns valores negativos e NaNs para exercitar o filtro de inconsistências
    busy = np.round(onshift * rng.uniform(0.4, 1.15, size=n_rows)).astype(float)
    outstanding = np.round(busy * rng.uniform(0.8, 1.8, size=n_rows)).astype(float)

    est_place_sec = rng.choice([250.0, 440.0], size=n_rows, p=[0.7, 0.3])
    est_drive_sec = rng.uniform(300.0, 1800.0, size=n_rows)

    # Equação sintética conhecida APENAS para verificação de sanidade do código
    cat_prep_map = {
        "fast": 10.0, "sandwiches": 11.0, "american": 16.0, "mexican": 15.0,
        "chinese": 17.0, "thai": 18.0, "japanese": 19.0, "italian": 21.0, "pizza": 23.0, None: 16.0
    }
    base_prep_min = np.array([cat_prep_map[c] for c in store_cats])
    safe_onshift = np.where(onshift > 0, onshift, 1.0)
    pressure_min = 8.0 * (busy / safe_onshift) + 4.5 * (outstanding / safe_onshift)
    items_min = 1.2 * total_items
    noise_min = rng.normal(0.0, 5.5, size=n_rows)

    duration_min = np.clip(
        (est_place_sec + est_drive_sec) / 60.0 + base_prep_min + pressure_min + items_min + noise_min,
        8.0,
        150.0,
    )
    duration_sec = np.round(duration_min * 60.0).astype(int)
    actual_delivery = [c + pd.Timedelta(seconds=int(d)) for c, d in zip(created_at, duration_sec)]

    df = pd.DataFrame({
        "market_id": market_ids,
        "created_at": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in created_at],
        "actual_delivery_time": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in actual_delivery],
        "store_id": store_ids,
        "store_primary_category": store_cats,
        "order_protocol": protocols,
        "total_items": total_items,
        "subtotal": subtotal,
        "num_distinct_items": num_distinct,
        "min_item_price": min_price,
        "max_item_price": max_price,
        "total_onshift_dashers": onshift,
        "total_busy_dashers": busy,
        "total_outstanding_orders": outstanding,
        "estimated_order_place_duration": est_place_sec,
        "estimated_store_to_consumer_driving_duration": np.round(est_drive_sec, 1),
    })

    # Injetar anomalias documentadas no dataset real (negativos, nulos e 2 outliers extremos)
    df.loc[0, "total_onshift_dashers"] = -5.0
    df.loc[1, "total_busy_dashers"] = np.nan
    df.loc[2, "actual_delivery_time"] = None
    df.loc[3, "actual_delivery_time"] = (
        pd.Timestamp(df.loc[3, "created_at"], tz="UTC") + pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d %H:%M:%S")

    return df
