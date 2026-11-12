"""
Módulo de Engenharia e Processamento de Dados (Etapa 2).

Responsável por:
1. Calcular a duração real e o prazo de referência definido pelo projeto.
2. Filtrar inconsistências físicas e outliers extremos documentados no dataset da DoorDash.
3. Realizar separação estritamente temporal (treino vs. validação/teste) usando `created_at`.
4. Tratar valores ausentes, razões com denominador zero e variáveis categóricas ajustando
   estatísticas exclusivamente no conjunto de treino (prevenção de Data Leakage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Colunas que JAMAIS podem entrar como features (X) pois contêm informação do futuro (vazamento)
FORBIDDEN_LEAKAGE_COLUMNS = {
    "actual_delivery_time",
    "actual_duration_seconds",
    "actual_duration_min",
    "actual_prep_duration_min",
    "delay_minutes",
    "is_delayed",
}


def compute_target_and_clean_rows(
    df: pd.DataFrame,
    min_duration_min: float = 5.0,
    max_duration_min: float = 180.0,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Converte timestamps, calcula o tempo real de entrega em minutos e remove
    registros sem alvo ou com durações fisicamente implausíveis no treino.
    """
    data = df.copy()
    audit_stats = {"initial_rows": len(data)}

    # 1. Remover linhas sem timestamp de entrega ou criação
    data["created_at"] = pd.to_datetime(data["created_at"], utc=True, errors="coerce")
    data["actual_delivery_time"] = pd.to_datetime(data["actual_delivery_time"], utc=True, errors="coerce")

    valid_timestamps = data["created_at"].notna() & data["actual_delivery_time"].notna()
    audit_stats["dropped_missing_timestamps"] = int((~valid_timestamps).sum())
    data = data.loc[valid_timestamps].copy()

    # 2. Calcular duração real em segundos e minutos
    data["actual_duration_seconds"] = (
        data["actual_delivery_time"] - data["created_at"]
    ).dt.total_seconds()
    data["actual_duration_min"] = data["actual_duration_seconds"] / 60.0

    # 3. Filtrar durações negativas/zero ou outliers extremos (ex.: pedidos esquecidos abertos por dias)
    if is_training:
        valid_duration = (
            (data["actual_duration_min"] >= min_duration_min)
            & (data["actual_duration_min"] <= max_duration_min)
        )
        audit_stats["dropped_extreme_duration_outliers"] = int((~valid_duration).sum())
        data = data.loc[valid_duration].copy()
    else:
        valid_duration = data["actual_duration_min"] > 0
        audit_stats["dropped_extreme_duration_outliers"] = int((~valid_duration).sum())
        data = data.loc[valid_duration].copy()

    audit_stats["final_rows"] = len(data)
    return data.reset_index(drop=True), audit_stats


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza a separação temporal estrita ordenada por `created_at`.
    Garante que o modelo treine apenas no passado e seja avaliado no futuro.
    """
    if "created_at" not in df.columns:
        raise ValueError("Coluna 'created_at' obrigatória para separação temporal.")

    if not (0 < train_frac < 1 and 0 < val_frac < 1 - train_frac):
        raise ValueError("As frações devem deixar espaço para treino, validação e teste.")

    sorted_df = df.sort_values("created_at", kind="stable").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    if not 0 < train_end < val_end < n:
        raise ValueError("Dados insuficientes para dividir em três janelas.")

    # Pedidos com o mesmo timestamp ficam na mesma janela.
    train_cut = sorted_df["created_at"].iloc[train_end]
    val_cut = sorted_df["created_at"].iloc[val_end]
    train_df = sorted_df.loc[sorted_df["created_at"] < train_cut].reset_index(drop=True)
    val_df = sorted_df.loc[sorted_df["created_at"].between(train_cut, val_cut, inclusive="left")].reset_index(drop=True)
    test_df = sorted_df.loc[sorted_df["created_at"] >= val_cut].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("São necessários timestamps distintos em cada janela.")
    return train_df, val_df, test_df


@dataclass
class DeliveryFeaturePipeline:
    """
    Pipeline stateful de pré-processamento e engenharia de atributos.
    Ajusta imputação, categorias e escalas no treino. O tempo adicional histórico
    é usado somente para calcular o prazo de referência.
    """
    top_k_categories: int = 15
    sla_buffer_min: float = 5.0
    smoothing_weight: float = 10.0

    # Atributos aprendidos no fit()
    numeric_medians_: Dict[str, float] = field(default_factory=dict)
    store_category_mode_: Dict[str, str] = field(default_factory=dict)
    top_categories_: List[str] = field(default_factory=list)
    markets_: List[str] = field(default_factory=list)
    protocols_: List[str] = field(default_factory=list)
    category_additional_median_min_: Dict[str, float] = field(default_factory=dict)
    global_additional_median_min_: float = 15.0
    feature_names_: List[str] = field(default_factory=list)
    feature_means_: Optional[np.ndarray] = None
    feature_stds_: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame) -> "DeliveryFeaturePipeline":
        df = train_df.copy()

        # 1. Aprender moda de store_primary_category por store_id (apenas no treino)
        valid_cat = df.dropna(subset=["store_id", "store_primary_category"])
        if not valid_cat.empty:
            modes = (
                valid_cat.groupby("store_id")["store_primary_category"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
            self.store_category_mode_ = modes.to_dict()

        # Imputar categoria antes de construir o one-hot e o prazo de referência.
        cat_series = self._impute_category(df)
        top_cats = cat_series.value_counts().head(self.top_k_categories).index.tolist()
        self.top_categories_ = [c for c in top_cats if c != "unknown"]

        # 2. Aprender medianas das colunas numéricas brutas (após limpar negativos)
        num_cols = [
            "total_items",
            "subtotal",
            "num_distinct_items",
            "min_item_price",
            "max_item_price",
            "total_onshift_dashers",
            "total_busy_dashers",
            "total_outstanding_orders",
            "estimated_order_place_duration",
            "estimated_store_to_consumer_driving_duration",
        ]
        for col in num_cols:
            cleaned = self._clean_numeric(df, col)
            self.numeric_medians_[col] = float(cleaned.median()) if cleaned.notna().any() else 0.0

        # 3. Estimar a parcela adicional para a referência. O restante pode incluir
        # preparo, espera, deslocamento até a loja e erros nas estimativas parciais.
        drive_min, place_min = self._estimated_stage_minutes(df)
        additional_min = (df["actual_duration_min"] - drive_min - place_min).clip(lower=2.0)
        self.global_additional_median_min_ = float(additional_min.median())
        stats = additional_min.groupby(cat_series).agg(["count", "median"])
        smoothed = (
            (stats["count"] * stats["median"] + self.smoothing_weight * self.global_additional_median_min_)
            / (stats["count"] + self.smoothing_weight)
        )
        self.category_additional_median_min_ = smoothed.to_dict()

        # 4. Aprender domínios de market_id e order_protocol no treino
        self.markets_ = sorted(
            df["market_id"].dropna().astype(float).astype(int).astype(str).unique().tolist()
        )[:10]
        self.protocols_ = sorted(
            df["order_protocol"].dropna().astype(float).astype(int).astype(str).unique().tolist()
        )[:10]

        # 5. Construir matriz do treino para registrar nomes e padronização (média/desvio)
        features_df = self._build_features_dataframe(df)
        self.feature_names_ = features_df.columns.tolist()
        X_mat = features_df.to_numpy(dtype=np.float64)
        self.feature_means_ = X_mat.mean(axis=0)
        stds = X_mat.std(axis=0)
        self.feature_stds_ = np.where(stds < 1e-8, 1.0, stds)
        return self

    def _impute_category(self, df: pd.DataFrame) -> pd.Series:
        raw_cat = df.get("store_primary_category", pd.Series(["unknown"] * len(df), index=df.index)).copy()
        raw_cat = raw_cat.astype(object).where(raw_cat.notna(), None)
        store_ids = df.get("store_id", pd.Series([None] * len(df), index=df.index))
        mapped_modes = store_ids.map(lambda sid: self.store_category_mode_.get(sid, "unknown"))
        filled = pd.Series(
            [c if (c is not None and str(c).strip() != "" and str(c) != "nan") else m for c, m in zip(raw_cat, mapped_modes)],
            index=df.index,
            dtype=str,
        )
        return filled.fillna("unknown").str.lower().str.strip()

    def _clean_numeric(self, df: pd.DataFrame, column: str) -> pd.Series:
        values = pd.to_numeric(df.get(column, pd.Series(np.nan, index=df.index)), errors="coerce")
        return values.where(np.isfinite(values) & (values >= 0))

    def _estimated_stage_minutes(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        drive = "estimated_store_to_consumer_driving_duration"
        place = "estimated_order_place_duration"
        return (
            self._clean_numeric(df, drive).fillna(self.numeric_medians_[drive]) / 60,
            self._clean_numeric(df, place).fillna(self.numeric_medians_[place]) / 60,
        )

    def compute_reference_eta_min(self, df: pd.DataFrame) -> pd.Series:
        """Prazo do projeto, calculado com estimativas parciais e histórico de treino."""
        categories = self._impute_category(df)
        additional = categories.map(self.category_additional_median_min_).fillna(self.global_additional_median_min_)
        drive_min, place_min = self._estimated_stage_minutes(df)
        return drive_min + place_min + additional + self.sla_buffer_min

    def _build_features_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Verificação explícita contra vazamento: nenhuma coluna proibida é usada nos cálculos abaixo
        out = pd.DataFrame(index=df.index)

        # Flag de ausência de dados de entregadores (informativa em certas praças)
        numeric = {col: self._clean_numeric(df, col) for col in self.numeric_medians_}
        missing = numeric["total_onshift_dashers"].isna() | numeric["total_busy_dashers"].isna()
        out["is_missing_dasher_info"] = (missing | numeric["total_outstanding_orders"].isna()).astype(float)
        cleaned_num = {col: values.fillna(self.numeric_medians_[col]) for col, values in numeric.items()}

        # Features de pedido
        total_items = cleaned_num["total_items"].clip(lower=1.0)
        num_distinct = np.minimum(cleaned_num["num_distinct_items"].clip(lower=1.0), total_items)
        subtotal_dollars = cleaned_num["subtotal"] / 100.0
        min_price_dollars = cleaned_num["min_item_price"] / 100.0
        max_price_dollars = np.maximum(cleaned_num["max_item_price"] / 100.0, min_price_dollars)

        out["total_items"] = total_items
        out["num_distinct_items"] = num_distinct
        out["subtotal_dollars"] = subtotal_dollars
        out["avg_item_price_dollars"] = subtotal_dollars / total_items
        out["price_range_dollars"] = max_price_dollars - min_price_dollars
        out["distinct_item_ratio"] = num_distinct / total_items

        # Features de oferta/demanda de entregadores (marketplace pressure)
        onshift = cleaned_num["total_onshift_dashers"]
        busy = cleaned_num["total_busy_dashers"]
        outstanding = cleaned_num["total_outstanding_orders"]

        out["total_onshift_dashers"] = onshift
        out["total_busy_dashers"] = busy
        out["total_outstanding_orders"] = outstanding

        # Razão segura sem divisão por zero (evita inf/-inf)
        safe_onshift = np.where(onshift > 0, onshift, 1.0)
        raw_busy_ratio = np.where(onshift > 0, busy / safe_onshift, 1.0)
        out["busy_dashers_ratio"] = np.clip(raw_busy_ratio, 0.0, 2.5)
        out["is_dasher_overcapacity"] = (busy > onshift).astype(float)
        out["orders_per_onshift_dasher"] = np.clip(outstanding / safe_onshift, 0.0, 10.0)
        out["free_dashers_estimate"] = np.maximum(onshift - busy, 0.0)

        # Features de estimativas operacionais parciais (em minutos)
        est_drive_min = cleaned_num["estimated_store_to_consumer_driving_duration"] / 60.0
        est_place_min = cleaned_num["estimated_order_place_duration"] / 60.0
        out["estimated_driving_min"] = est_drive_min
        out["estimated_order_place_min"] = est_place_min
        out["estimated_non_prep_min"] = est_drive_min + est_place_min

        cat_series = self._impute_category(df)

        # Features temporais extraídas de created_at (convertido para US/Pacific, fuso original do dataset)
        created_ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        local_ts = created_ts.dt.tz_convert("America/Los_Angeles")

        hour = local_ts.dt.hour.fillna(18).astype(float)
        dow = local_ts.dt.day_of_week.fillna(4).astype(float)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["is_weekend"] = (dow >= 5).astype(float)
        out["is_peak_lunch"] = ((hour >= 11) & (hour <= 14)).astype(float)
        out["is_peak_dinner"] = ((hour >= 18) & (hour <= 21)).astype(float)

        # One-Hot Encoding controlado para market_id, order_protocol e top categorias
        market_str = (
            pd.to_numeric(df.get("market_id", pd.Series(-1, index=df.index)), errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        for m in self.markets_:
            out[f"market_{m}"] = (market_str == m).astype(float)

        proto_str = (
            pd.to_numeric(df.get("order_protocol", pd.Series(-1, index=df.index)), errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        for p in self.protocols_:
            out[f"protocol_{p}"] = (proto_str == p).astype(float)

        for c in self.top_categories_:
            out[f"cat_{c}"] = (cat_series == c).astype(float)

        # Garantir ausência total de NaN ou Inf
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def transform(self, df: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
        features_df = self._build_features_dataframe(df)
        # Alinhar colunas na ordem exata do treino
        for col in self.feature_names_:
            if col not in features_df.columns:
                features_df[col] = 0.0
        features_df = features_df[self.feature_names_]

        if standardize and self.feature_means_ is not None and self.feature_stds_ is not None:
            scaled_vals = (features_df.to_numpy(dtype=np.float64) - self.feature_means_) / self.feature_stds_
            return pd.DataFrame(scaled_vals, index=features_df.index, columns=self.feature_names_)
        return features_df
