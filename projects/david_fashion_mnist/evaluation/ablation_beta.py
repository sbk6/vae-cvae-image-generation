"""
Étude d'ablation de la valeur de bêta pour les modèles VAE et CVAE.

Ce script exploite :

1. les six historiques d'entraînement enregistrés dans
   results/training_histories ;
2. les métriques finales calculées sur les 10 000 images de test dans
   results/evaluation_metrics.csv.

Il produit :

- un tableau récapitulatif CSV ;
- les courbes de reconstruction de validation ;
- les courbes KL de validation ;
- une comparaison des reconstructions sur le jeu de test ;
- une comparaison des termes KL sur le jeu de test ;
- un graphique du compromis reconstruction / régularisation.

La loss totale n'est pas comparée entre des valeurs différentes de bêta,
car sa définition dépend directement de bêta :

    loss totale = reconstruction + bêta × KL

Exécution depuis la racine du projet :

    python -m evaluation.ablation_beta
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Final

import matplotlib

# Le backend Agg permet de créer les fichiers PNG sans ouvrir de fenêtre.
# Il est adapté à une exécution depuis PowerShell ou un serveur sans écran.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ================================================================
# CHEMINS DU PROJET
# ================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
HISTORY_DIR: Final[Path] = RESULTS_DIR / "training_histories"
EVALUATION_METRICS_PATH: Final[Path] = (
    RESULTS_DIR / "evaluation_metrics.csv"
)
DEFAULT_OUTPUT_DIR: Final[Path] = RESULTS_DIR / "ablation"


# ================================================================
# CONFIGURATION DES SIX EXPÉRIENCES
# ================================================================

# Chaque entrée relie :
#
# - le nom interne de l'expérience ;
# - le type de modèle ;
# - la valeur de bêta ;
# - le fichier CSV d'historique correspondant.
EXPERIMENTS: Final[list[dict[str, object]]] = [
    {
        "run_name": "vae_beta_01",
        "model_type": "VAE",
        "beta": 0.1,
        "history_file": "vae_beta_01_history.csv",
    },
    {
        "run_name": "vae_beta_1",
        "model_type": "VAE",
        "beta": 1.0,
        "history_file": "vae_beta_1_history.csv",
    },
    {
        "run_name": "vae_beta_4",
        "model_type": "VAE",
        "beta": 4.0,
        "history_file": "vae_beta_4_history.csv",
    },
    {
        "run_name": "cvae_beta_01",
        "model_type": "CVAE",
        "beta": 0.1,
        "history_file": "cvae_beta_01_history.csv",
    },
    {
        "run_name": "cvae_beta_1",
        "model_type": "CVAE",
        "beta": 1.0,
        "history_file": "cvae_beta_1_history.csv",
    },
    {
        "run_name": "cvae_beta_4",
        "model_type": "CVAE",
        "beta": 4.0,
        "history_file": "cvae_beta_4_history.csv",
    },
]


HISTORY_REQUIRED_COLUMNS: Final[set[str]] = {
    "epoch",
    "train_total",
    "train_reconstruction",
    "train_kl",
    "validation_total",
    "validation_reconstruction",
    "validation_kl",
}


EVALUATION_REQUIRED_COLUMNS: Final[set[str]] = {
    "checkpoint",
    "model_type",
    "beta",
    "best_epoch",
    "best_validation_loss",
    "test_total",
    "test_reconstruction",
    "test_kl",
    "processed_test_images",
}


def beta_display(beta: float) -> str:
    """
    Transforme une valeur de bêta en texte lisible.

    Exemples
    --------
    0.1 devient "0,1"
    1.0 devient "1"
    4.0 devient "4"
    """

    if math.isclose(beta, round(beta)):
        return str(int(round(beta)))

    return str(beta).replace(".", ",")


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_path: Path,
) -> None:
    """
    Vérifie qu'un fichier CSV contient toutes les colonnes attendues.
    """

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Le fichier {file_path} ne contient pas toutes les "
            f"colonnes attendues. Colonnes absentes : "
            f"{sorted(missing_columns)}"
        )


def load_history(
    history_path: Path,
) -> pd.DataFrame:
    """
    Charge et vérifie un historique d'entraînement.
    """

    if not history_path.exists():
        raise FileNotFoundError(
            f"Historique introuvable : {history_path}"
        )

    dataframe = pd.read_csv(history_path)

    validate_columns(
        dataframe=dataframe,
        required_columns=HISTORY_REQUIRED_COLUMNS,
        file_path=history_path,
    )

    numeric_columns = sorted(HISTORY_REQUIRED_COLUMNS)

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe = dataframe.sort_values("epoch").reset_index(drop=True)

    if dataframe.empty:
        raise ValueError(
            f"L'historique est vide : {history_path}"
        )

    if dataframe["epoch"].duplicated().any():
        duplicated_epochs = (
            dataframe.loc[
                dataframe["epoch"].duplicated(),
                "epoch",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"Des époques sont répétées dans {history_path} : "
            f"{duplicated_epochs}"
        )

    return dataframe


def load_all_histories() -> dict[str, pd.DataFrame]:
    """
    Charge les six historiques dans un dictionnaire.
    """

    histories: dict[str, pd.DataFrame] = {}

    for experiment in EXPERIMENTS:
        run_name = str(experiment["run_name"])
        history_file = str(experiment["history_file"])
        history_path = HISTORY_DIR / history_file

        histories[run_name] = load_history(history_path)

    return histories


def load_evaluation_metrics() -> pd.DataFrame:
    """
    Charge les métriques finales calculées sur le jeu de test.
    """

    if not EVALUATION_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Le fichier des métriques de test est introuvable : "
            f"{EVALUATION_METRICS_PATH}"
        )

    dataframe = pd.read_csv(EVALUATION_METRICS_PATH)

    validate_columns(
        dataframe=dataframe,
        required_columns=EVALUATION_REQUIRED_COLUMNS,
        file_path=EVALUATION_METRICS_PATH,
    )

    numeric_columns = [
        "beta",
        "best_epoch",
        "best_validation_loss",
        "test_total",
        "test_reconstruction",
        "test_kl",
        "processed_test_images",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe["model_type"] = (
        dataframe["model_type"].astype(str).str.upper()
    )

    if len(dataframe) != 6:
        raise ValueError(
            "Le fichier evaluation_metrics.csv devrait contenir "
            f"6 modèles, mais {len(dataframe)} lignes ont été trouvées."
        )

    if not (dataframe["processed_test_images"] == 10_000).all():
        invalid_rows = dataframe.loc[
            dataframe["processed_test_images"] != 10_000,
            ["checkpoint", "processed_test_images"],
        ]

        raise ValueError(
            "Les métriques ne proviennent pas toutes de l'évaluation "
            "complète sur 10 000 images.\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    return dataframe


def find_evaluation_row(
    evaluation_metrics: pd.DataFrame,
    model_type: str,
    beta: float,
) -> pd.Series:
    """
    Trouve la ligne d'évaluation correspondant à un modèle et un bêta.
    """

    beta_mask = np.isclose(
        evaluation_metrics["beta"].to_numpy(dtype=float),
        beta,
    )

    model_mask = (
        evaluation_metrics["model_type"].to_numpy(dtype=str)
        == model_type
    )

    matches = evaluation_metrics.loc[beta_mask & model_mask]

    if len(matches) != 1:
        raise ValueError(
            "Une seule ligne d'évaluation était attendue pour "
            f"{model_type}, bêta={beta}, mais {len(matches)} "
            "ligne(s) ont été trouvée(s)."
        )

    return matches.iloc[0]


def build_summary(
    histories: dict[str, pd.DataFrame],
    evaluation_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit un tableau récapitulatif des six expériences.

    Pour chaque expérience, la meilleure époque est déterminée par la
    plus petite loss totale de validation, comme pendant l'entraînement.
    """

    summary_rows: list[dict[str, object]] = []

    for experiment in EXPERIMENTS:
        run_name = str(experiment["run_name"])
        model_type = str(experiment["model_type"])
        beta = float(experiment["beta"])

        history = histories[run_name]

        best_index = history["validation_total"].idxmin()
        best_history_row = history.loc[best_index]

        evaluation_row = find_evaluation_row(
            evaluation_metrics=evaluation_metrics,
            model_type=model_type,
            beta=beta,
        )

        history_best_epoch = int(best_history_row["epoch"])
        checkpoint_best_epoch = int(evaluation_row["best_epoch"])

        if history_best_epoch != checkpoint_best_epoch:
            raise ValueError(
                "L'époque du meilleur résultat dans l'historique ne "
                "correspond pas à l'époque enregistrée dans le "
                f"checkpoint pour {model_type}, bêta={beta}. "
                f"Historique={history_best_epoch}, "
                f"checkpoint={checkpoint_best_epoch}."
            )

        history_best_loss = float(
            best_history_row["validation_total"]
        )
        checkpoint_best_loss = float(
            evaluation_row["best_validation_loss"]
        )

        if not math.isclose(
            history_best_loss,
            checkpoint_best_loss,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "La meilleure loss de validation dans l'historique "
                "ne correspond pas à celle du checkpoint pour "
                f"{model_type}, bêta={beta}. "
                f"Historique={history_best_loss}, "
                f"checkpoint={checkpoint_best_loss}."
            )

        summary_rows.append(
            {
                "run_name": run_name,
                "model_type": model_type,
                "beta": beta,
                "best_epoch": history_best_epoch,
                "validation_total": history_best_loss,
                "validation_reconstruction": float(
                    best_history_row["validation_reconstruction"]
                ),
                "validation_kl": float(
                    best_history_row["validation_kl"]
                ),
                "test_total": float(
                    evaluation_row["test_total"]
                ),
                "test_reconstruction": float(
                    evaluation_row["test_reconstruction"]
                ),
                "test_kl": float(
                    evaluation_row["test_kl"]
                ),
                "processed_test_images": int(
                    evaluation_row["processed_test_images"]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        by=["model_type", "beta"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary


def save_summary_csv(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Sauvegarde le tableau récapitulatif au format CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )


def save_validation_curve(
    histories: dict[str, pd.DataFrame],
    model_type: str,
    metric_column: str,
    y_label: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Trace une métrique de validation au fil des époques.

    Une courbe est créée pour chaque valeur de bêta.
    """

    figure, axis = plt.subplots(figsize=(9, 5.5))

    for experiment in EXPERIMENTS:
        if str(experiment["model_type"]) != model_type:
            continue

        run_name = str(experiment["run_name"])
        beta = float(experiment["beta"])
        history = histories[run_name]

        axis.plot(
            history["epoch"],
            history[metric_column],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"β = {beta_display(beta)}",
        )

    axis.set_title(title)
    axis.set_xlabel("Époque")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_xticks(
        sorted(
            histories[
                next(
                    str(exp["run_name"])
                    for exp in EXPERIMENTS
                    if str(exp["model_type"]) == model_type
                )
            ]["epoch"]
            .astype(int)
            .unique()
            .tolist()
        )
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_grouped_test_chart(
    summary: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Crée un diagramme en barres comparant VAE et CVAE.

    Les groupes correspondent aux trois valeurs de bêta.
    """

    beta_values = [0.1, 1.0, 4.0]
    beta_labels = [beta_display(beta) for beta in beta_values]

    vae_values: list[float] = []
    cvae_values: list[float] = []

    for beta in beta_values:
        vae_row = summary.loc[
            (summary["model_type"] == "VAE")
            & np.isclose(summary["beta"], beta)
        ]

        cvae_row = summary.loc[
            (summary["model_type"] == "CVAE")
            & np.isclose(summary["beta"], beta)
        ]

        if len(vae_row) != 1 or len(cvae_row) != 1:
            raise ValueError(
                "Impossible de construire le graphique : une valeur "
                f"unique est attendue pour bêta={beta}."
            )

        vae_values.append(
            float(vae_row.iloc[0][metric_column])
        )
        cvae_values.append(
            float(cvae_row.iloc[0][metric_column])
        )

    positions = np.arange(len(beta_values))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    vae_bars = axis.bar(
        positions - bar_width / 2,
        vae_values,
        width=bar_width,
        label="VAE",
    )

    cvae_bars = axis.bar(
        positions + bar_width / 2,
        cvae_values,
        width=bar_width,
        label="CVAE",
    )

    axis.set_title(title)
    axis.set_xlabel("Valeur de β")
    axis.set_ylabel(y_label)
    axis.set_xticks(positions)
    axis.set_xticklabels(beta_labels)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()

    axis.bar_label(
        vae_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        cvae_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_tradeoff_chart(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Représente le compromis entre reconstruction et régularisation.

    Axe horizontal :
        perte de reconstruction sur le jeu de test.

    Axe vertical :
        terme KL sur le jeu de test.

    Une reconstruction plus faible est meilleure.
    Un KL plus faible indique un espace latent davantage régularisé.
    """

    figure, axis = plt.subplots(figsize=(8.5, 6))

    markers = {
        "VAE": "o",
        "CVAE": "s",
    }

    for model_type in ["VAE", "CVAE"]:
        model_rows = summary.loc[
            summary["model_type"] == model_type
        ].sort_values("beta")

        axis.scatter(
            model_rows["test_reconstruction"],
            model_rows["test_kl"],
            marker=markers[model_type],
            s=80,
            label=model_type,
        )

        axis.plot(
            model_rows["test_reconstruction"],
            model_rows["test_kl"],
            linewidth=1.2,
        )

        for _, row in model_rows.iterrows():
            axis.annotate(
                f"β={beta_display(float(row['beta']))}",
                (
                    float(row["test_reconstruction"]),
                    float(row["test_kl"]),
                ),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

    axis.set_title(
        "Compromis reconstruction / régularisation sur le jeu de test"
    )
    axis.set_xlabel(
        "Perte de reconstruction test — plus faible = meilleure"
    )
    axis.set_ylabel(
        "Terme KL test — plus faible = espace plus régularisé"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments utilisables depuis PowerShell.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Créer les graphiques de l'étude d'ablation de bêta "
            "pour les modèles VAE et CVAE."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Dossier de sortie. Valeur par défaut : "
            "results/ablation."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help=(
            "Résolution des images PNG. Valeur par défaut : 200."
        ),
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Vérifie les arguments reçus depuis le terminal.
    """

    if args.dpi <= 0:
        raise ValueError(
            "dpi doit être strictement positif."
        )


def resolve_output_dir(output_dir: Path) -> Path:
    """
    Transforme le dossier de sortie en chemin absolu.

    Un chemin relatif est interprété depuis la racine du projet.
    """

    if output_dir.is_absolute():
        return output_dir.resolve()

    return (PROJECT_ROOT / output_dir).resolve()


def print_summary(summary: pd.DataFrame) -> None:
    """
    Affiche un résumé lisible dans le terminal.
    """

    print()
    print("RÉSUMÉ DES RÉSULTATS FINAUX")
    print("-" * 92)
    print(
        f"{'Modèle':<8}"
        f"{'β':>7}"
        f"{'Époque':>10}"
        f"{'Val recon':>14}"
        f"{'Val KL':>12}"
        f"{'Test recon':>14}"
        f"{'Test KL':>12}"
    )
    print("-" * 92)

    for _, row in summary.iterrows():
        print(
            f"{str(row['model_type']):<8}"
            f"{beta_display(float(row['beta'])):>7}"
            f"{int(row['best_epoch']):>10}"
            f"{float(row['validation_reconstruction']):>14.4f}"
            f"{float(row['validation_kl']):>12.4f}"
            f"{float(row['test_reconstruction']):>14.4f}"
            f"{float(row['test_kl']):>12.4f}"
        )

    print("-" * 92)


def main() -> None:
    """
    Point d'entrée principal du script.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)

    output_dir = resolve_output_dir(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("ÉTUDE D'ABLATION DE BÊTA — VAE ET CVAE")
    print("=" * 72)
    print(f"Historiques : {HISTORY_DIR}")
    print(f"Métriques   : {EVALUATION_METRICS_PATH}")
    print(f"Sorties     : {output_dir}")
    print("=" * 72)

    histories = load_all_histories()
    evaluation_metrics = load_evaluation_metrics()

    summary = build_summary(
        histories=histories,
        evaluation_metrics=evaluation_metrics,
    )

    summary_path = output_dir / "ablation_summary.csv"

    save_summary_csv(
        summary=summary,
        output_path=summary_path,
    )

    generated_files: list[Path] = [summary_path]

    plot_configurations = [
        {
            "model_type": "VAE",
            "metric_column": "validation_reconstruction",
            "y_label": "Perte de reconstruction de validation",
            "title": (
                "VAE — reconstruction de validation selon β"
            ),
            "filename": (
                "vae_validation_reconstruction_curves.png"
            ),
        },
        {
            "model_type": "VAE",
            "metric_column": "validation_kl",
            "y_label": "Terme KL de validation",
            "title": "VAE — terme KL de validation selon β",
            "filename": "vae_validation_kl_curves.png",
        },
        {
            "model_type": "CVAE",
            "metric_column": "validation_reconstruction",
            "y_label": "Perte de reconstruction de validation",
            "title": (
                "CVAE — reconstruction de validation selon β"
            ),
            "filename": (
                "cvae_validation_reconstruction_curves.png"
            ),
        },
        {
            "model_type": "CVAE",
            "metric_column": "validation_kl",
            "y_label": "Terme KL de validation",
            "title": "CVAE — terme KL de validation selon β",
            "filename": "cvae_validation_kl_curves.png",
        },
    ]

    for configuration in plot_configurations:
        output_path = output_dir / str(
            configuration["filename"]
        )

        save_validation_curve(
            histories=histories,
            model_type=str(configuration["model_type"]),
            metric_column=str(configuration["metric_column"]),
            y_label=str(configuration["y_label"]),
            title=str(configuration["title"]),
            output_path=output_path,
            dpi=args.dpi,
        )

        generated_files.append(output_path)

    test_reconstruction_path = (
        output_dir / "test_reconstruction_by_beta.png"
    )

    save_grouped_test_chart(
        summary=summary,
        metric_column="test_reconstruction",
        y_label="Perte de reconstruction sur le jeu de test",
        title="Reconstruction test selon β — VAE contre CVAE",
        output_path=test_reconstruction_path,
        dpi=args.dpi,
    )

    generated_files.append(test_reconstruction_path)

    test_kl_path = output_dir / "test_kl_by_beta.png"

    save_grouped_test_chart(
        summary=summary,
        metric_column="test_kl",
        y_label="Terme KL sur le jeu de test",
        title="Terme KL test selon β — VAE contre CVAE",
        output_path=test_kl_path,
        dpi=args.dpi,
    )

    generated_files.append(test_kl_path)

    tradeoff_path = (
        output_dir / "test_reconstruction_kl_tradeoff.png"
    )

    save_tradeoff_chart(
        summary=summary,
        output_path=tradeoff_path,
        dpi=args.dpi,
    )

    generated_files.append(tradeoff_path)

    print_summary(summary)

    print()
    print(
        "Remarque : la loss totale n'est pas comparée entre des "
        "valeurs différentes de β, car sa définition dépend de β."
    )

    print()
    print("FICHIERS CRÉÉS")
    print("-" * 72)

    for file_path in generated_files:
        print(f"{file_path} -> {file_path.stat().st_size} octets")

    print("-" * 72)
    print("ÉTUDE D'ABLATION TERMINÉE")


if __name__ == "__main__":
    main()