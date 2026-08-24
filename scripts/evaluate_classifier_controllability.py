"""Contrôlabilité du CVAE mesurée avec un vrai classifieur dédié (plutôt que le
proxy plus-proche-centroïde de scripts/evaluate.py), sur les 3 checkpoints
multi-seed déjà utilisés pour la validation multi-seed de l'ablation (0, 42, 123).
Même principe que le classifieur Fashion-MNIST de David.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch

from src.data.datasets import build_dataloaders
from src.models.classifier import DigitClassifier
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


def rescale_to_classifier_range(x: torch.Tensor) -> torch.Tensor:
    """Le décodeur MNIST a un ReLU juste avant la Tanh finale (bug documenté,
    section methode.tex), ce qui confine sa sortie à [0, 1) au lieu de [-1, 1].
    Le classifieur a appris sur de vraies images normalisées en [-1, 1] (fond
    noir = -1). Sans cette remise à l'échelle, un fond gris au lieu de noir
    suffit à dérégler complètement les BatchNorm du classifieur et à produire
    une précision proche du hasard, indépendamment de la classe réellement
    dessinée. On réétire donc chaque image générée, individuellement, vers la
    pleine plage [-1, 1] (min-max par image) avant classification : ça corrige
    le contraste sans changer quel chiffre est dessiné."""
    flat = x.view(x.size(0), -1)
    x_min = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    x_max = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + 1e-8) * 2 - 1


@torch.no_grad()
def evaluate_one_seed(cvae_config_path: str, cvae_checkpoint_path: str, classifier: torch.nn.Module,
                       device: torch.device, n_per_class: int = 200) -> dict:
    config = load_yaml_config(cvae_config_path)
    _, _, _, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, cvae_checkpoint_path, device)

    per_class_accuracy = {}
    total_correct, total = 0, 0
    for cls in range(dataset_info.num_conditions):
        samples = model.sample(cls, n=n_per_class).to(device)
        samples = rescale_to_classifier_range(samples)
        predicted = classifier(samples).argmax(dim=1)
        correct = (predicted == cls).sum().item()
        per_class_accuracy[str(cls)] = correct / n_per_class
        total_correct += correct
        total += n_per_class

    return {"per_class_accuracy": per_class_accuracy, "overall_accuracy": total_correct / total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-config", type=str, default="configs/mnist_vae.yaml")
    parser.add_argument("--classifier-checkpoint", type=str, default="reports/experiments/mnist_classifier/best_checkpoint.pth")
    parser.add_argument("--cvae-config", type=str, default="configs/mnist_cvae.yaml")
    parser.add_argument("--seeds-dir", type=str, default="reports/experiments/cvae_seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--output", type=str, default="reports/experiments/classifier_controllability.json")
    args = parser.parse_args()

    classifier_config = load_yaml_config(args.classifier_config)
    device = get_device(classifier_config["training"].get("device", "auto"))
    _, _, _, dataset_info = build_dataloaders(classifier_config)

    classifier = DigitClassifier(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        num_classes=dataset_info.num_conditions,
    ).to(device)
    classifier.load_state_dict(torch.load(args.classifier_checkpoint, map_location=device))
    classifier.eval()

    classifier_summary_path = Path(args.classifier_checkpoint).parent / "summary.json"
    with open(classifier_summary_path, "r", encoding="utf-8") as f:
        classifier_summary = json.load(f)

    per_seed = {}
    accuracies = []
    for seed in args.seeds:
        checkpoint_path = str(Path(args.seeds_dir) / f"seed_{seed}" / "best_checkpoint.pth")
        print(f"Évaluation seed {seed} ({checkpoint_path})...")
        result = evaluate_one_seed(args.cvae_config, checkpoint_path, classifier, device, args.n_per_class)
        print(f"  overall_accuracy = {result['overall_accuracy']:.4f}")
        per_seed[str(seed)] = result
        accuracies.append(result["overall_accuracy"])

    mean_acc = sum(accuracies) / len(accuracies)
    std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)) ** 0.5

    results = {
        "classifier_test_accuracy": classifier_summary["test_accuracy"],
        "per_seed": per_seed,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "n_per_class": args.n_per_class,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nMoyenne sur {len(args.seeds)} graines : {mean_acc:.4f} (écart-type {std_acc:.4f})")
    print(f"Résultats sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
