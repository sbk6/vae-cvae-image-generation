"""
Fonctions de perte utilisées pour entraîner le VAE et le CVAE.

Un VAE est entraîné en minimisant une fonction composée de deux termes :

1. la perte de reconstruction ;
2. la divergence KL.

La formule utilisée dans ce projet est :

    loss_totale = reconstruction_loss + beta * kl_loss

Le paramètre beta permet de contrôler l'importance de la régularisation
de l'espace latent.

Cette fonction peut être utilisée aussi bien avec le VAE classique
qu'avec le CVAE, car les deux modèles produisent :

    reconstruction, mu, logvar, z
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calcule la fonction de perte d'un VAE ou d'un CVAE.

    Parameters
    ----------
    reconstruction:
        Images reconstruites par le décodeur.

        Forme attendue :

            [batch_size, 1, 28, 28]

    target:
        Images originales que le modèle cherche à reconstruire.

        Forme attendue :

            [batch_size, 1, 28, 28]

    mu:
        Moyennes produites par l'encodeur.

        Forme attendue :

            [batch_size, latent_dim]

    logvar:
        Logarithmes des variances produits par l'encodeur.

        Forme attendue :

            [batch_size, latent_dim]

    beta:
        Poids appliqué au terme KL.

        beta = 1 correspond au VAE classique.

        beta différent de 1 correspond à une expérience de type
        beta-VAE.

    Returns
    -------
    total_loss:
        Somme de la reconstruction loss et du terme KL pondéré.

    reconstruction_loss:
        Erreur moyenne de reconstruction par image.

    kl_loss:
        Divergence KL moyenne par image.

    Notes
    -----
    Dans la théorie, l'ELBO est une quantité que l'on cherche à maximiser.

    En pratique, les bibliothèques d'apprentissage minimisent une loss.
    On minimise donc l'opposé de l'ELBO, qui prend la forme :

        reconstruction_loss + beta * kl_loss
    """

    # Vérifie que les dimensions des tenseurs sont compatibles.
    _validate_loss_inputs(
        reconstruction=reconstruction,
        target=target,
        mu=mu,
        logvar=logvar,
        beta=beta,
    )

    # Nombre d'images présentes dans le batch.
    batch_size = reconstruction.shape[0]

    # =============================================================
    # 1. PERTE DE RECONSTRUCTION
    # =============================================================
    #
    # Binary Cross-Entropy est adaptée ici parce que :
    #
    # - les images ont des pixels entre 0 et 1 ;
    # - le décodeur termine par une fonction Sigmoid ;
    # - Fashion-MNIST est constitué d'images en niveaux de gris.
    #
    # reduction="sum" additionne l'erreur sur :
    #
    # - tous les pixels ;
    # - toutes les images du batch.
    #
    # On divise ensuite par batch_size afin d'obtenir une perte moyenne
    # par image. Cela rend les résultats comparables même si la taille
    # du batch change.
    reconstruction_loss = F.binary_cross_entropy(
        input=reconstruction,
        target=target,
        reduction="sum",
    )

    reconstruction_loss = reconstruction_loss / batch_size

    # =============================================================
    # 2. DIVERGENCE KL
    # =============================================================
    #
    # L'encodeur produit une distribution :
    #
    #     q(z | x) = N(mu, sigma²)
    #
    # Le terme KL cherche à la rapprocher d'une loi normale standard :
    #
    #     p(z) = N(0, I)
    #
    # La formule analytique est :
    #
    #     KL = -0.5 * somme(
    #         1 + logvar - mu² - exp(logvar)
    #     )
    #
    # La somme est effectuée sur toutes les dimensions latentes et
    # toutes les images. On divise ensuite par batch_size pour obtenir
    # une valeur moyenne par image.
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    kl_loss = kl_loss / batch_size

    # =============================================================
    # 3. LOSS TOTALE DU BETA-VAE
    # =============================================================
    #
    # beta contrôle l'importance du terme KL.
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


def _validate_loss_inputs(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float,
) -> None:
    """
    Vérifie que les entrées de la fonction de perte sont valides.

    Ces vérifications permettent d'obtenir des messages d'erreur plus
    faciles à comprendre qu'une erreur interne de PyTorch.
    """

    # Les quatre entrées doivent être des tenseurs PyTorch.
    tensor_arguments = {
        "reconstruction": reconstruction,
        "target": target,
        "mu": mu,
        "logvar": logvar,
    }

    for argument_name, argument_value in tensor_arguments.items():
        if not isinstance(argument_value, Tensor):
            raise TypeError(
                f"{argument_name} doit être un tenseur PyTorch."
            )

    # L'image reconstruite doit avoir la même forme que l'image originale.
    if reconstruction.shape != target.shape:
        raise ValueError(
            "La reconstruction et l'image cible doivent avoir "
            "exactement la même forme. "
            f"Reconstruction : {tuple(reconstruction.shape)}. "
            f"Cible : {tuple(target.shape)}."
        )

    # Le projet utilise des images Fashion-MNIST :
    # [batch_size, 1, 28, 28].
    if reconstruction.ndim != 4:
        raise ValueError(
            "La reconstruction doit avoir quatre dimensions : "
            "[batch_size, canal, hauteur, largeur]."
        )

    if reconstruction.shape[1:] != (1, 28, 28):
        raise ValueError(
            "Le format attendu pour les images est "
            "[batch_size, 1, 28, 28]. "
            f"Format reçu : {tuple(reconstruction.shape)}."
        )

    # Le batch ne doit pas être vide.
    if reconstruction.shape[0] == 0:
        raise ValueError(
            "Le batch d'images ne doit pas être vide."
        )

    # mu et logvar doivent décrire la même distribution latente.
    if mu.shape != logvar.shape:
        raise ValueError(
            "mu et logvar doivent avoir exactement la même forme. "
            f"Mu : {tuple(mu.shape)}. "
            f"Logvar : {tuple(logvar.shape)}."
        )

    # Les paramètres latents doivent avoir la forme :
    # [batch_size, latent_dim].
    if mu.ndim != 2:
        raise ValueError(
            "mu et logvar doivent avoir la forme "
            "[batch_size, latent_dim]."
        )

    # Il doit exister un vecteur latent pour chaque image.
    if mu.shape[0] != reconstruction.shape[0]:
        raise ValueError(
            "Le nombre de vecteurs latents doit correspondre "
            "au nombre d'images. "
            f"Images : {reconstruction.shape[0]}. "
            f"Vecteurs latents : {mu.shape[0]}."
        )

    # La dimension latente doit contenir au moins une valeur.
    if mu.shape[1] == 0:
        raise ValueError(
            "La dimension de l'espace latent ne peut pas être nulle."
        )

    # Les calculs de loss nécessitent des nombres flottants.
    if not reconstruction.is_floating_point():
        raise TypeError(
            "La reconstruction doit contenir des nombres flottants."
        )

    if not target.is_floating_point():
        raise TypeError(
            "L'image cible doit contenir des nombres flottants."
        )

    if not mu.is_floating_point() or not logvar.is_floating_point():
        raise TypeError(
            "mu et logvar doivent contenir des nombres flottants."
        )

    # beta doit être un nombre réel positif ou nul.
    if not isinstance(beta, (int, float)):
        raise TypeError(
            "beta doit être un nombre entier ou flottant."
        )

    if not math.isfinite(beta):
        raise ValueError(
            "beta doit être un nombre fini."
        )

    if beta < 0:
        raise ValueError(
            "beta doit être supérieur ou égal à zéro."
        )


def test_vae_loss() -> None:
    """
    Teste la fonction de perte avec des valeurs simples.

    Le test vérifie :

    - la formule de reconstruction ;
    - la formule de KL ;
    - l'effet du paramètre beta ;
    - le fonctionnement de la rétropropagation.
    """

    # Rend le test reproductible.
    torch.manual_seed(42)

    batch_size = 4
    latent_dim = 16

    # Les images cibles sont entièrement noires.
    target = torch.zeros(
        batch_size,
        1,
        28,
        28,
        dtype=torch.float32,
    )

    # La reconstruction contient uniquement des pixels égaux à 0.5.
    reconstruction = torch.full(
        size=(batch_size, 1, 28, 28),
        fill_value=0.5,
        dtype=torch.float32,
    )

    # mu = 1 et logvar = 0 donnent une KL simple à calculer.
    #
    # Pour chaque dimension latente :
    #
    #     KL = 0.5
    #
    # Avec 16 dimensions :
    #
    #     KL par image = 8
    mu = torch.ones(
        batch_size,
        latent_dim,
        dtype=torch.float32,
    )

    logvar = torch.zeros(
        batch_size,
        latent_dim,
        dtype=torch.float32,
    )

    # Calcule les pertes pour les trois valeurs prévues dans l'ablation.
    results = {}

    for beta in (0.1, 1.0, 4.0):
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            target=target,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        results[beta] = {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "kl": kl_loss,
        }

    # Pour un pixel cible égal à 0 et une prédiction égale à 0.5 :
    #
    #     BCE = -log(1 - 0.5) = log(2)
    #
    # Une image contient 28 × 28 = 784 pixels.
    expected_reconstruction = torch.tensor(
        28 * 28 * math.log(2),
        dtype=torch.float32,
    )

    # Avec mu=1, logvar=0 et 16 dimensions :
    expected_kl = torch.tensor(
        0.5 * latent_dim,
        dtype=torch.float32,
    )

    # Vérifie les valeurs théoriques.
    assert torch.isclose(
        results[1.0]["reconstruction"],
        expected_reconstruction,
        rtol=1e-5,
        atol=1e-5,
    )

    assert torch.isclose(
        results[1.0]["kl"],
        expected_kl,
        rtol=1e-5,
        atol=1e-5,
    )

    # Vérifie que la loss totale augmente lorsque beta augmente,
    # puisque le terme KL reçoit davantage de poids.
    assert (
        results[0.1]["total"]
        < results[1.0]["total"]
        < results[4.0]["total"]
    )

    # =============================================================
    # TEST DE RÉTROPROPAGATION
    # =============================================================

    # Les logits sont des valeurs avant l'application de Sigmoid.
    # requires_grad=True demande à PyTorch de calculer leurs gradients.
    logits = torch.zeros(
        batch_size,
        1,
        28,
        28,
        requires_grad=True,
    )

    differentiable_reconstruction = torch.sigmoid(logits)

    differentiable_mu = torch.zeros(
        batch_size,
        latent_dim,
        requires_grad=True,
    )

    differentiable_logvar = torch.zeros(
        batch_size,
        latent_dim,
        requires_grad=True,
    )

    total_loss, _, _ = vae_loss(
        reconstruction=differentiable_reconstruction,
        target=target,
        mu=differentiable_mu,
        logvar=differentiable_logvar,
        beta=1.0,
    )

    # Calcule les gradients nécessaires à l'entraînement.
    total_loss.backward()

    # Vérifie que les gradients ont bien été produits.
    assert logits.grad is not None
    assert differentiable_mu.grad is not None
    assert differentiable_logvar.grad is not None

    print("Test de la fonction de perte réussi.")
    print()
    print("Valeurs obtenues pour l'étude de beta :")

    for beta in (0.1, 1.0, 4.0):
        print(
            f"beta={beta:<3} | "
            f"reconstruction="
            f"{results[beta]['reconstruction'].item():.4f} | "
            f"KL={results[beta]['kl'].item():.4f} | "
            f"total={results[beta]['total'].item():.4f}"
        )

    print()
    print("Rétropropagation réussie : les gradients existent.")


# Le test est lancé uniquement lorsque ce fichier est exécuté directement.
#
# Il ne sera pas lancé lorsque vae_loss sera importée dans les scripts
# d'entraînement.
if __name__ == "__main__":
    test_vae_loss()