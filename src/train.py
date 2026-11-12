"""
Script de Treinamento, Validação Temporal e Registro de Artefatos (Etapa 3).

Pode ser executado:
1. Com o dataset real da DoorDash baixado do Kaggle:
   python -m src.train --data data/raw/historical_data.csv
2. Em modo de verificação de pipeline com a amostra sintética de teste:
   python -m src.train --synthetic-check
"""
from __future__ import annotations

import argparse
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from src.data_processing import (
    DeliveryFeaturePipeline,
    compute_target_and_clean_rows,
    temporal_train_val_test_split,
)
from src.models import OperationalBaselineModel, RidgeETAModel, calibration_bins, evaluate_predictions
from src.generate_synthetic_fixture import generate_synthetic_doordash_sample

BUNDLE_VERSION = 2
REAL_ARTIFACTS_DIR = Path("models")
SYNTHETIC_ARTIFACTS_DIR = Path("models_synthetic")


def run_training_pipeline(
    df_raw: pd.DataFrame,
    data_source_label: str,
    artifacts_dir: Path,
    alpha: float = 25.0,
) -> Dict[str, Any]:
    """
    Executa o fluxo completo: limpeza -> split temporal -> fit no treino ->
    avaliação comparativa contra o baseline no conjunto de teste futuro.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Durações positivas formam a população mais ampla usada no teste de sensibilidade.
    positive_df, audit_stats = compute_target_and_clean_rows(df_raw, is_training=False)
    duration_in_range = positive_df["actual_duration_min"].between(5.0, 180.0)
    cleaned_df = positive_df.loc[duration_in_range]
    audit_stats.update({
        "dropped_nonpositive_duration": audit_stats.pop("dropped_extreme_duration_outliers"),
        "dropped_outside_5_to_180_min": int((~duration_in_range).sum()),
        "final_rows": len(cleaned_df),
    })

    # 2. Separação temporal estrita (70% passado = treino, 15% = validação, 15% futuro = teste)
    train_df, val_df, test_df = temporal_train_val_test_split(
        cleaned_df, train_frac=0.70, val_frac=0.15
    )

    validation_cut = val_df["created_at"].min()
    test_cut = test_df["created_at"].min()
    train_pending = train_df["actual_delivery_time"] >= validation_cut
    val_pending = val_df["actual_delivery_time"] >= test_cut
    train_df = train_df.loc[~train_pending]
    val_df = val_df.loc[~val_pending]
    if train_df.empty or val_df.empty:
        raise ValueError("Faltam pedidos já entregues nos cortes de treino e validação.")

    # 3. Ajustar imputação, escalas e prazo de referência com pedidos de treino.
    pipeline = DeliveryFeaturePipeline(top_k_categories=12, sla_buffer_min=5.0)
    pipeline.fit(train_df)

    X_train_scaled = pipeline.transform(train_df, standardize=True)
    y_train_min = train_df["actual_duration_min"].to_numpy()

    # 4. Ajustar os modelos no treino e a distribuição de erros na validação.
    baseline = OperationalBaselineModel().fit(train_df, y_train_min, pipeline)
    model = RidgeETAModel(alpha=alpha).fit(X_train_scaled, y_train_min)
    model.fit_risk(pipeline.transform(val_df), val_df["actual_duration_min"].to_numpy())

    def evaluate_window(frame: pd.DataFrame) -> Dict[str, Any]:
        y = frame["actual_duration_min"].to_numpy()
        X = pipeline.transform(frame)
        reference = pipeline.compute_reference_eta_min(frame).to_numpy()
        prediction = model.predict_duration_min(X)
        risk = model.predict_delay_probability(X, reference)
        normal_risk = model.predict_normal_delay_probability(X, reference)
        baseline_risk = baseline.predict_delay_probability(frame, pipeline)
        return {
            "rows": len(frame),
            "median_baseline": evaluate_predictions(y, np.full(len(y), np.median(y_train_min)), reference, baseline_risk),
            "operational_baseline": evaluate_predictions(y, baseline.predict_duration_min(frame, pipeline), reference, baseline_risk),
            "ridge": evaluate_predictions(y, prediction, reference, risk),
            "ridge_normal_risk": evaluate_predictions(y, prediction, reference, normal_risk),
            "risk_calibration": calibration_bins(y > reference, risk),
        }

    # Mesmo corte temporal. A segunda avaliação inclui durações positivas fora de [5, 180].
    primary_test = evaluate_window(test_df)
    positive_test = evaluate_window(positive_df.loc[positive_df["created_at"] >= test_cut])

    top_features_df = model.get_global_feature_importance(top_k=10)

    # Salvar artefatos para uso na API
    bundle = {
        "version": BUNDLE_VERSION,
        "pipeline": pipeline,
        "baseline": baseline,
        "model": model,
        "data_source_label": data_source_label,
    }
    with open(artifacts_dir / "model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    report = {
        "data_source_label": data_source_label,
        "warning_pt": (
            "ATENÇÃO: Resultados obtidos em amostra sintética servem APENAS para validar o "
            "funcionamento do código e NÃO representam o comportamento real de entregas."
            if "sintética" in data_source_label.lower() or "synthetic" in data_source_label.lower()
            else "Avaliação histórica na amostra DoorDash de 2015 com ruído na origem."
        ),
        "audit_cleaning": audit_stats,
        "temporal_split": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_excluded_pending_at_validation_cut": int(train_pending.sum()),
            "validation_excluded_pending_at_test_cut": int(val_pending.sum()),
            "validation_cut_utc": str(validation_cut),
            "test_cut_utc": str(test_cut),
            "train_period": f"{train_df['created_at'].min()} -> {train_df['created_at'].max()}",
            "validation_period": f"{val_df['created_at'].min()} -> {val_df['created_at'].max()}",
            "test_period": f"{test_df['created_at'].min()} -> {test_df['created_at'].max()}",
        },
        "configuration": {
            "ridge_alpha": alpha,
            "reference_margin_min": pipeline.sla_buffer_min,
            "primary_duration_range_min": [5.0, 180.0],
            "risk_method": "empirical_validation_residuals",
        },
        "test_metrics_median_baseline": primary_test["median_baseline"],
        "test_metrics_baseline": primary_test["operational_baseline"],
        "test_metrics_ridge_model": primary_test["ridge"],
        "test_metrics_ridge_normal_risk": primary_test["ridge_normal_risk"],
        "test_risk_calibration": primary_test["risk_calibration"],
        "test_all_positive_durations": positive_test,
        "top_global_features": top_features_df.to_dict(orient="records"),
    }

    with open(artifacts_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento do modelo de previsão de entrega e risco de atraso.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Caminho para historical_data.csv da DoorDash")
    source.add_argument("--synthetic-check", action="store_true", help="Executa validação técnica com amostra sintética")
    parser.add_argument("--artifacts-dir", type=Path, help="Diretório de saída dos artefatos")
    args = parser.parse_args()

    if args.data is not None:
        if not args.data.is_file():
            parser.error(f"Arquivo CSV ausente: {args.data}")
        df_raw = pd.read_csv(args.data)
        source_label = "real_doordash_csv"
        artifacts_dir = args.artifacts_dir or REAL_ARTIFACTS_DIR
    else:
        artifacts_dir = args.artifacts_dir or SYNTHETIC_ARTIFACTS_DIR
        if artifacts_dir.resolve() == REAL_ARTIFACTS_DIR.resolve():
            parser.error("Use um diretório diferente de models/ para a execução sintética.")
        if (artifacts_dir / "model_bundle.pkl").exists() or (artifacts_dir / "evaluation_report.json").exists():
            report_path = artifacts_dir / "evaluation_report.json"
            try:
                previous_source = json.loads(report_path.read_text(encoding="utf-8"))["data_source_label"]
            except (OSError, ValueError, KeyError, TypeError):
                parser.error("A pasta contém artefatos de origem desconhecida. Use uma pasta vazia.")
            if not isinstance(previous_source, str) or not previous_source.startswith("synthetic_ci_fixture"):
                parser.error("A pasta já contém artefatos de outro tipo. Use uma pasta de teste.")
        df_raw = generate_synthetic_doordash_sample(n_rows=1500, seed=42)
        source_label = "synthetic_ci_fixture (APENAS TESTE DE SOFTWARE - NÃO REPRESENTA ENTREGAS REAIS)"

    report = run_training_pipeline(df_raw, source_label, artifacts_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
