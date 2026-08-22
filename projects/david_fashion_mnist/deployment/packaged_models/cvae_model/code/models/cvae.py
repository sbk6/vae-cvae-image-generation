"""
Définition d'un Conditional Variational Autoencoder pour Fashion-MNIST.

Le CVAE est une version conditionnelle du VAE.

En plus de l'image, il reçoit une étiquette de classe. Cette étiquette
permet de contrôler la catégorie de vêtement que le modèle reconstruit
ou génère.

Pour Fashion-MNIST, les dix classes sont :

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

Le label est injecté à deux endroits :

1. dans l'encodeur, avec les caractéristiques extraites de l'image ;
2. dans le décodeur, avec le vecteur latent z.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder pour Fashion-MNIST.

    Parameters
    ----------
    latent_dim:
        Dimension du vecteur latent z.

    hidden_dim:
        Taille de la couche cachée entièrement connectée.

    num_classes:
        Nombre de classes du dataset.

        Fashion-MNIST possède dix classes, donc la valeur par défaut
        est égale à 10.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        """
        Initialise les différentes couches du CVAE.
        """

        # Initialise la classe de base nn.Module de PyTorch.
        super().__init__()

        # Vérification des paramètres reçus.
        if latent_dim <= 0:
            raise ValueError(
                "latent_dim doit être strictement supérieur à zéro."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim doit être strictement supérieur à zéro."
            )

        if num_classes <= 1:
            raise ValueError(
                "num_classes doit être supérieur à un."
            )

        # Conservation des paramètres pour les autres méthodes.
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # =========================================================
        # 1. EXTRACTION DES CARACTÉRISTIQUES DE L'IMAGE
        # =========================================================
        #
        # Entrée :
        # [batch_size, 1, 28, 28]
        #
        # Après la première convolution :
        # [batch_size, 32, 14, 14]
        #
        # Après la deuxième convolution :
        # [batch_size, 64, 7, 7]
        #
        self.encoder_convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # Transforme [64, 7, 7] en un vecteur de 3136 valeurs.
            nn.Flatten(),
        )

        # Dimension obtenue après les convolutions :
        #
        # 64 canaux × 7 pixels × 7 pixels = 3136.
        self.encoder_output_dim = 64 * 7 * 7

        # =========================================================
        # 2. ENCODEUR CONDITIONNEL
        # =========================================================
        #
        # Les caractéristiques de l'image ont une taille de 3136.
        #
        # Le label one-hot a une taille de 10.
        #
        # Ils sont concaténés :
        #
        # [caractéristiques de l'image, label one-hot]
        #
        # La taille totale est donc :
        #
        # 3136 + 10 = 3146
        #
        self.encoder_hidden = nn.Sequential(
            nn.Linear(
                in_features=self.encoder_output_dim + num_classes,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
        )

        # Comme dans le VAE classique, l'encodeur produit les paramètres
        # d'une distribution gaussienne :
        #
        # - mu : moyenne ;
        # - logvar : logarithme de la variance.
        self.fc_mu = nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
        )

        self.fc_logvar = nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
        )

        # =========================================================
        # 3. ENTRÉE DU DÉCODEUR CONDITIONNEL
        # =========================================================
        #
        # Le décodeur reçoit :
        #
        # - le vecteur latent z ;
        # - le label one-hot.
        #
        # Leur taille concaténée est :
        #
        # latent_dim + num_classes
        #
        self.decoder_input = nn.Sequential(
            nn.Linear(
                in_features=latent_dim + num_classes,
                out_features=hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=hidden_dim,
                out_features=self.encoder_output_dim,
            ),
            nn.ReLU(),
        )

        # =========================================================
        # 4. DÉCODEUR CONVOLUTIONNEL
        # =========================================================
        #
        # Le décodeur reconstruit progressivement l'image :
        #
        # [batch_size, 64, 7, 7]
        #              ↓
        # [batch_size, 32, 14, 14]
        #              ↓
        # [batch_size, 1, 28, 28]
        #
        self.decoder_convolutions = nn.Sequential(
            # Reforme le vecteur de 3136 valeurs en tenseur 3D.
            nn.Unflatten(
                dim=1,
                unflattened_size=(64, 7, 7),
            ),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            # Les pixels reconstruits doivent rester entre 0 et 1.
            nn.Sigmoid(),
        )

    def labels_to_one_hot(
        self,
        labels: Tensor,
        expected_batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """
        Convertit les labels numériques en vecteurs one-hot.

        Exemple
        -------
        Pour dix classes, le label 3 devient :

            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        Parameters
        ----------
        labels:
            Tenseur de labels de forme [batch_size].

        expected_batch_size:
            Nombre d'images ou de vecteurs latents dans le batch.

        device:
            Appareil sur lequel placer les labels : CPU ou GPU.

        Returns
        -------
        one_hot_labels:
            Tenseur de forme [batch_size, num_classes].
        """

        # Vérifie la forme, le type et les valeurs des labels.
        self._validate_labels(
            labels=labels,
            expected_batch_size=expected_batch_size,
        )

        # Place les labels sur le même appareil que les images ou z.
        labels = labels.to(device=device)

        # Convertit les indices de classe en vecteurs one-hot.
        one_hot_labels = F.one_hot(
            labels,
            num_classes=self.num_classes,
        )

        # F.one_hot produit des entiers.
        # Les couches linéaires ont besoin de nombres flottants.
        one_hot_labels = one_hot_labels.float()

        return one_hot_labels

    def encode(
        self,
        x: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Encode les images en utilisant également leurs labels.

        Parameters
        ----------
        x:
            Images de forme [batch_size, 1, 28, 28].

        labels:
            Labels de forme [batch_size].

        Returns
        -------
        mu:
            Moyennes des distributions latentes.

        logvar:
            Logarithmes des variances latentes.
        """

        # Vérifie le format des images.
        self._validate_images(x)

        # Transforme les labels en vecteurs one-hot.
        one_hot_labels = self.labels_to_one_hot(
            labels=labels,
            expected_batch_size=x.shape[0],
            device=x.device,
        )

        # Extrait les caractéristiques visuelles des images.
        image_features = self.encoder_convolutions(x)

        # Concatène les caractéristiques des images et les labels.
        #
        # dim=1 signifie que l'on concatène les colonnes de chaque exemple.
        conditional_features = torch.cat(
            tensors=(image_features, one_hot_labels),
            dim=1,
        )

        # Passage dans la couche cachée de l'encodeur.
        hidden = self.encoder_hidden(conditional_features)

        # Calcul de la moyenne et du logarithme de la variance.
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Applique l'astuce de reparamétrisation.

        La formule est :

            z = mu + sigma * epsilon

        avec :

            sigma = exp(0.5 * logvar)

        et :

            epsilon ~ N(0, I)
        """

        if mu.shape != logvar.shape:
            raise ValueError(
                "mu et logvar doivent avoir exactement la même forme."
            )

        # Transformation du logarithme de variance en écart-type.
        standard_deviation = torch.exp(0.5 * logvar)

        # Génération d'un bruit gaussien.
        epsilon = torch.randn_like(standard_deviation)

        # Construction du vecteur latent z.
        z = mu + standard_deviation * epsilon

        return z

    def decode(
        self,
        z: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Décode les vecteurs latents en utilisant les labels demandés.

        Parameters
        ----------
        z:
            Vecteurs latents de forme [batch_size, latent_dim].

        labels:
            Labels de forme [batch_size].

        Returns
        -------
        reconstruction:
            Images reconstruites de forme [batch_size, 1, 28, 28].
        """

        # Vérifie que z possède deux dimensions.
        if z.ndim != 2:
            raise ValueError(
                "z doit avoir la forme [batch_size, latent_dim]."
            )

        # Vérifie la dimension de l'espace latent.
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Dimension latente attendue : {self.latent_dim}. "
                f"Dimension reçue : {z.shape[1]}."
            )

        # Transforme les labels en vecteurs one-hot.
        one_hot_labels = self.labels_to_one_hot(
            labels=labels,
            expected_batch_size=z.shape[0],
            device=z.device,
        )

        # Concatène z avec le label conditionnel.
        conditional_latent = torch.cat(
            tensors=(z, one_hot_labels),
            dim=1,
        )

        # Transforme le vecteur conditionnel en caractéristiques 3136.
        decoder_features = self.decoder_input(conditional_latent)

        # Reconstruit l'image.
        reconstruction = self.decoder_convolutions(decoder_features)

        return reconstruction

    def forward(
        self,
        x: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Exécute le passage complet dans le CVAE.

        Le chemin suivi est :

            image + label
                  ↓
              encodeur
                  ↓
           mu et logvar
                  ↓
          reparamétrisation
                  ↓
                  z
                  ↓
              z + label
                  ↓
              décodeur
                  ↓
            reconstruction

        Returns
        -------
        reconstruction:
            Images reconstruites.

        mu:
            Moyennes des distributions latentes.

        logvar:
            Logarithmes des variances latentes.

        z:
            Vecteurs latents échantillonnés.
        """

        # L'encodeur reçoit l'image et son label.
        mu, logvar = self.encode(x, labels)

        # Échantillonne le vecteur latent.
        z = self.reparameterize(mu, logvar)

        # Le décodeur reçoit le vecteur latent et le label.
        reconstruction = self.decode(z, labels)

        return reconstruction, mu, logvar, z

    @torch.no_grad()
    def sample(
        self,
        labels: Tensor,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """
        Génère des images correspondant aux classes demandées.

        Une image est générée pour chaque label fourni.

        Exemple
        -------
        Pour générer cinq chaussures de classe 9 :

            labels = torch.tensor([9, 9, 9, 9, 9])
            images = model.sample(labels)

        Parameters
        ----------
        labels:
            Classes à générer, de forme [num_samples].

        device:
            Appareil de calcul utilisé.

        Returns
        -------
        generated_images:
            Images générées de forme [num_samples, 1, 28, 28].
        """

        # Vérifie les labels avant de calculer le nombre d'images.
        self._validate_labels(
            labels=labels,
            expected_batch_size=labels.shape[0]
            if labels.ndim == 1
            else 0,
        )

        # Utilise l'appareil du modèle lorsqu'aucun appareil n'est fourni.
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        # Place les labels sur l'appareil choisi.
        labels = labels.to(device=device)

        # Le nombre d'images générées correspond au nombre de labels.
        num_samples = labels.shape[0]

        # Tire des vecteurs latents selon une loi normale standard.
        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=device,
        )

        # Décode chaque z en utilisant le label associé.
        generated_images = self.decode(z, labels)

        return generated_images

    @staticmethod
    def _validate_images(x: Tensor) -> None:
        """
        Vérifie que les images ont le format de Fashion-MNIST.
        """

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

    def _validate_labels(
        self,
        labels: Tensor,
        expected_batch_size: int,
    ) -> None:
        """
        Vérifie que les labels sont valides pour Fashion-MNIST.
        """

        # Les labels doivent être fournis dans un tenseur.
        if not isinstance(labels, Tensor):
            raise TypeError(
                "Les labels doivent être fournis sous forme de Tensor."
            )

        # Un batch de labels doit avoir la forme [batch_size].
        if labels.ndim != 1:
            raise ValueError(
                "Les labels doivent avoir la forme [batch_size]."
            )

        # Le batch ne doit pas être vide.
        if labels.numel() == 0:
            raise ValueError(
                "Le tenseur de labels ne doit pas être vide."
            )

        # Fashion-MNIST fournit les labels sous forme d'entiers int64.
        if labels.dtype != torch.long:
            raise ValueError(
                "Les labels doivent avoir le type torch.int64 "
                "ou torch.long."
            )

        # Il doit exister un label pour chaque image ou vecteur latent.
        if labels.shape[0] != expected_batch_size:
            raise ValueError(
                "Le nombre de labels doit correspondre à la taille "
                "du batch. "
                f"Labels reçus : {labels.shape[0]}. "
                f"Taille attendue : {expected_batch_size}."
            )

        # Vérifie que les indices sont compris entre 0 et num_classes - 1.
        minimum_label = labels.min().item()
        maximum_label = labels.max().item()

        if minimum_label < 0 or maximum_label >= self.num_classes:
            raise ValueError(
                f"Les labels doivent être compris entre 0 et "
                f"{self.num_classes - 1}. "
                f"Valeurs observées : minimum={minimum_label}, "
                f"maximum={maximum_label}."
            )


def test_cvae() -> None:
    """
    Effectue un test rapide de l'architecture du CVAE.

    Ce test ne réalise aucun entraînement.

    Il vérifie :

    - les dimensions des entrées et sorties ;
    - la conversion des labels en one-hot ;
    - l'encodeur conditionnel ;
    - la reparamétrisation ;
    - le décodeur conditionnel ;
    - la génération pour une classe demandée.
    """

    # Fixe la graine aléatoire pour rendre le test reproductible.
    torch.manual_seed(42)

    batch_size = 8
    latent_dim = 16
    num_classes = 10

    # Création du modèle conditionnel.
    model = CVAE(
        latent_dim=latent_dim,
        hidden_dim=256,
        num_classes=num_classes,
    )

    # Faux batch ayant le format de Fashion-MNIST.
    fake_images = torch.rand(
        batch_size,
        1,
        28,
        28,
    )

    # Un label est associé à chaque image du batch.
    fake_labels = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 9],
        dtype=torch.long,
    )

    # Vérification explicite de la conversion one-hot.
    one_hot_labels = model.labels_to_one_hot(
        labels=fake_labels,
        expected_batch_size=batch_size,
        device=fake_images.device,
    )

    assert one_hot_labels.shape == (batch_size, num_classes)

    # Passage complet dans le CVAE.
    reconstruction, mu, logvar, z = model(
        fake_images,
        fake_labels,
    )

    # Vérification des dimensions.
    assert reconstruction.shape == (batch_size, 1, 28, 28)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)

    # Demande de génération de dix images de classe 9.
    generation_labels = torch.full(
        size=(10,),
        fill_value=9,
        dtype=torch.long,
    )

    generated_images = model.sample(generation_labels)

    assert generated_images.shape == (10, 1, 28, 28)

    # La Sigmoid doit maintenir les pixels entre 0 et 1.
    assert generated_images.min().item() >= 0.0
    assert generated_images.max().item() <= 1.0

    print("Test du CVAE réussi.")
    print(f"Images d'entrée       : {tuple(fake_images.shape)}")
    print(f"Labels                : {tuple(fake_labels.shape)}")
    print(f"Labels one-hot        : {tuple(one_hot_labels.shape)}")
    print(f"Reconstructions       : {tuple(reconstruction.shape)}")
    print(f"Mu                    : {tuple(mu.shape)}")
    print(f"Logvar                : {tuple(logvar.shape)}")
    print(f"Vecteurs latents      : {tuple(z.shape)}")
    print(f"Images générées       : {tuple(generated_images.shape)}")
    print("Classe demandée       : 9 - Ankle boot")
    print(
        "Intervalle des pixels : "
        f"[{generated_images.min().item():.4f}, "
        f"{generated_images.max().item():.4f}]"
    )


# Le test est exécuté uniquement lorsque ce fichier est lancé directement.
#
# Il ne sera pas exécuté lorsque CVAE sera importé dans un autre fichier.
if __name__ == "__main__":
    test_cvae()