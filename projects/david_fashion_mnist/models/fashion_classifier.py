"""
Classifieur convolutionnel indépendant pour Fashion-MNIST.

Ce modèle n'est ni un VAE ni un CVAE.

Son rôle est d'apprendre à reconnaître les dix classes de Fashion-MNIST,
puis de servir d'évaluateur externe pour mesurer la cohérence
conditionnelle des images générées par le CVAE.

Classes Fashion-MNIST
---------------------
0 : T-shirt/top
1 : Trouser
2 : Pullover
3 : Dress
4 : Coat
5 : Sandal
6 : Shirt
7 : Sneaker
8 : Bag
9 : Ankle boot

Le classifieur reçoit une image de forme :

    [batch_size, 1, 28, 28]

et produit dix logits :

    [batch_size, 10]

Un logit est un score brut associé à une classe.
La classe prédite correspond au logit le plus élevé.
"""

import torch
from torch import Tensor, nn


# Noms officiels des classes Fashion-MNIST.
FASHION_MNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


class FashionMNISTClassifier(nn.Module):
    """
    Réseau convolutionnel pour classifier Fashion-MNIST.

    L'architecture reste volontairement indépendante des VAE/CVAE.

    Elle contient :

    - deux blocs convolutionnels ;
    - deux opérations de max-pooling ;
    - une couche entièrement connectée ;
    - du dropout pour limiter le surapprentissage ;
    - une couche finale produisant dix logits.

    Parameters
    ----------
    num_classes:
        Nombre de classes à prédire.

        Fashion-MNIST possède dix classes.
    """

    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        """
        Initialise le classifieur.
        """

        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes doit être supérieur à un."
            )

        self.num_classes = num_classes

        # =========================================================
        # 1. EXTRACTION DES CARACTÉRISTIQUES
        # =========================================================
        #
        # Entrée :
        #
        # [batch_size, 1, 28, 28]
        #
        # Premier bloc :
        #
        # [batch_size, 32, 28, 28]
        #              ↓ MaxPool
        # [batch_size, 32, 14, 14]
        #
        # Deuxième bloc :
        #
        # [batch_size, 64, 14, 14]
        #              ↓ MaxPool
        # [batch_size, 64, 7, 7]
        #
        self.features = nn.Sequential(
            # -----------------------------------------------------
            # Bloc convolutionnel 1
            # -----------------------------------------------------
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # -----------------------------------------------------
            # Bloc convolutionnel 2
            # -----------------------------------------------------
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # Après les deux MaxPool :
        #
        # 28 × 28
        #   ↓
        # 14 × 14
        #   ↓
        # 7 × 7
        #
        # avec 64 canaux.
        self.feature_output_dim = 64 * 7 * 7

        # =========================================================
        # 2. CLASSIFICATION
        # =========================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=self.feature_output_dim,
                out_features=128,
            ),
            nn.ReLU(),

            # Le dropout est utilisé uniquement en mode entraînement.
            nn.Dropout(p=0.30),

            nn.Linear(
                in_features=128,
                out_features=num_classes,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Effectue une prédiction.

        Parameters
        ----------
        x:
            Images de forme [batch_size, 1, 28, 28].

        Returns
        -------
        logits:
            Scores bruts de forme [batch_size, num_classes].
        """

        self._validate_images(x)

        features = self.features(x)

        logits = self.classifier(features)

        return logits

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Retourne directement les classes prédites.

        Parameters
        ----------
        x:
            Images de forme [batch_size, 1, 28, 28].

        Returns
        -------
        predictions:
            Indices des classes prédites, de forme [batch_size].
        """

        logits = self.forward(x)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        return predictions

    @staticmethod
    def _validate_images(
        x: Tensor,
    ) -> None:
        """
        Vérifie que les images respectent le format Fashion-MNIST.
        """

        if not isinstance(x, Tensor):
            raise TypeError(
                "Les images doivent être fournies sous forme de Tensor."
            )

        if x.ndim != 4:
            raise ValueError(
                "Les images doivent avoir quatre dimensions : "
                "[batch_size, canal, hauteur, largeur]."
            )

        if x.shape[1:] != (1, 28, 28):
            raise ValueError(
                "Le format attendu est "
                "[batch_size, 1, 28, 28]. "
                f"Format reçu : {tuple(x.shape)}."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "Le batch d'images ne doit pas être vide."
            )


def test_fashion_classifier() -> None:
    """
    Effectue un test technique rapide du classifieur.

    Ce test ne réalise aucun entraînement.

    Il vérifie :

    - les dimensions d'entrée ;
    - les dimensions des logits ;
    - la méthode predict ;
    - la validité des indices de classes prédits.
    """

    # Rend le test reproductible.
    torch.manual_seed(42)

    batch_size = 8
    num_classes = 10

    # Création du modèle.
    model = FashionMNISTClassifier(
        num_classes=num_classes,
    )

    # Le mode évaluation désactive notamment le Dropout
    # et utilise les statistiques de BatchNorm disponibles.
    model.eval()

    # Création d'un faux batch Fashion-MNIST.
    fake_images = torch.rand(
        batch_size,
        1,
        28,
        28,
    )

    # Calcul des logits.
    with torch.no_grad():
        logits = model(fake_images)

    # Vérification de leur dimension.
    assert logits.shape == (
        batch_size,
        num_classes,
    )

    # Calcul des classes prédites.
    predictions = model.predict(fake_images)

    assert predictions.shape == (batch_size,)

    # Toutes les prédictions doivent appartenir
    # à l'intervalle [0, 9].
    assert predictions.min().item() >= 0
    assert predictions.max().item() < num_classes

    # Affichage d'informations utiles.
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Test du classifieur Fashion-MNIST réussi.")
    print(
        f"Images d'entrée       : "
        f"{tuple(fake_images.shape)}"
    )
    print(
        f"Logits                : "
        f"{tuple(logits.shape)}"
    )
    print(
        f"Prédictions           : "
        f"{tuple(predictions.shape)}"
    )
    print(
        f"Classes prédites      : "
        f"{predictions.tolist()}"
    )
    print(
        f"Paramètres totaux     : "
        f"{total_parameters:,}"
    )
    print(
        f"Paramètres entraînables : "
        f"{trainable_parameters:,}"
    )


if __name__ == "__main__":
    test_fashion_classifier()