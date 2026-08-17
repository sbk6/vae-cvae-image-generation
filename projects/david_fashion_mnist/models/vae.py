"""
Définition d'un Variational Autoencoder convolutionnel pour Fashion-MNIST.

Fashion-MNIST contient des images en niveaux de gris de taille 28 x 28.

Le VAE est composé de quatre grandes parties :

1. un encodeur convolutionnel ;
2. deux couches produisant mu et logvar ;
3. une étape de reparamétrisation ;
4. un décodeur convolutionnel.

Le modèle reçoit une image de forme :

    [batch_size, 1, 28, 28]

et produit une reconstruction ayant exactement la même forme.
"""

from typing import Optional

import torch
from torch import Tensor, nn


class VAE(nn.Module):
    """
    Variational Autoencoder convolutionnel pour Fashion-MNIST.

    Parameters
    ----------
    latent_dim:
        Dimension du vecteur latent z.

        Une valeur de 16 signifie que chaque image sera représentée
        dans un espace latent de 16 dimensions.

    hidden_dim:
        Taille de la couche entièrement connectée intermédiaire.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialise toutes les couches du VAE.
        """

        # Initialise la classe nn.Module de PyTorch.
        super().__init__()

        # Vérifications simples pour éviter des dimensions invalides.
        if latent_dim <= 0:
            raise ValueError(
                "latent_dim doit être strictement supérieur à zéro."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim doit être strictement supérieur à zéro."
            )

        # On conserve les dimensions dans l'objet pour les réutiliser
        # dans les autres méthodes.
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # =========================================================
        # 1. ENCODEUR CONVOLUTIONNEL
        # =========================================================
        #
        # Entrée :
        # [batch_size, 1, 28, 28]
        #
        # Première convolution :
        # [batch_size, 32, 14, 14]
        #
        # Deuxième convolution :
        # [batch_size, 64, 7, 7]
        #
        self.encoder = nn.Sequential(
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

            # Transforme le tenseur [64, 7, 7] en un vecteur.
            nn.Flatten(),
        )

        # Après les convolutions, chaque image est représentée par :
        #
        # 64 canaux × 7 pixels × 7 pixels = 3136 valeurs.
        self.encoder_output_dim = 64 * 7 * 7

        # Cette couche réduit le vecteur de 3136 valeurs vers hidden_dim.
        self.encoder_hidden = nn.Sequential(
            nn.Linear(
                in_features=self.encoder_output_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
        )

        # Le VAE ne produit pas directement un seul vecteur latent.
        #
        # Il produit les paramètres d'une distribution gaussienne :
        #
        # - mu : la moyenne ;
        # - logvar : le logarithme de la variance.
        self.fc_mu = nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
        )

        self.fc_logvar = nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
        )

        # =========================================================
        # 2. DÉBUT DU DÉCODEUR
        # =========================================================
        #
        # Le vecteur latent z est transformé en un vecteur de taille
        # 64 × 7 × 7 afin de pouvoir reconstruire une image.
        #
        self.decoder_input = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
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
        # 3. DÉCODEUR CONVOLUTIONNEL
        # =========================================================
        #
        # Le décodeur effectue l'opération inverse de l'encodeur.
        #
        # [batch_size, 64, 7, 7]
        #              ↓
        # [batch_size, 32, 14, 14]
        #              ↓
        # [batch_size, 1, 28, 28]
        #
        self.decoder = nn.Sequential(
            # Transforme le vecteur de taille 3136 en tenseur 3D.
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

            # ToTensor() transforme les pixels en valeurs entre 0 et 1.
            #
            # La fonction Sigmoid garantit donc que les pixels reconstruits
            # restent eux aussi entre 0 et 1.
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode un batch d'images.

        Parameters
        ----------
        x:
            Images de forme [batch_size, 1, 28, 28].

        Returns
        -------
        mu:
            Moyennes des distributions latentes.

        logvar:
            Logarithmes des variances latentes.
        """

        # Vérifie que le format des images correspond à Fashion-MNIST.
        self._validate_images(x)

        # Extraction des caractéristiques visuelles.
        encoded = self.encoder(x)

        # Transformation vers la couche cachée.
        hidden = self.encoder_hidden(encoded)

        # Calcul des paramètres de la distribution latente.
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Applique l'astuce de reparamétrisation.

        Le VAE veut échantillonner :

            z ~ N(mu, sigma²)

        Mais un échantillonnage direct empêcherait la rétropropagation.

        On écrit donc :

            z = mu + sigma * epsilon

        avec :

            epsilon ~ N(0, I)

        et :

            sigma = exp(0.5 * logvar)
        """

        if mu.shape != logvar.shape:
            raise ValueError(
                "mu et logvar doivent avoir exactement la même forme."
            )

        # Conversion du logarithme de la variance en écart-type.
        standard_deviation = torch.exp(0.5 * logvar)

        # Tirage d'un bruit gaussien de même forme que l'écart-type.
        epsilon = torch.randn_like(standard_deviation)

        # Construction du vecteur latent.
        z = mu + standard_deviation * epsilon

        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Décode des vecteurs latents pour produire des images.

        Parameters
        ----------
        z:
            Vecteurs latents de forme [batch_size, latent_dim].

        Returns
        -------
        reconstruction:
            Images reconstruites de forme [batch_size, 1, 28, 28].
        """

        # Un batch de vecteurs latents doit avoir deux dimensions :
        # [nombre d'exemples, dimension latente].
        if z.ndim != 2:
            raise ValueError(
                "z doit avoir la forme [batch_size, latent_dim]."
            )

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Dimension latente attendue : {self.latent_dim}. "
                f"Dimension reçue : {z.shape[1]}."
            )

        # Transformation du vecteur latent vers 64 × 7 × 7.
        decoder_features = self.decoder_input(z)

        # Reconstruction de l'image.
        reconstruction = self.decoder(decoder_features)

        return reconstruction

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Exécute le passage complet dans le VAE.

        Le chemin est :

            image
              ↓
            encodeur
              ↓
            mu et logvar
              ↓
            reparamétrisation
              ↓
            z
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

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar, z

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Génère de nouvelles images sans image d'entrée.

        Pour générer une image avec un VAE, on tire un vecteur aléatoire :

            z ~ N(0, I)

        puis on envoie ce vecteur dans le décodeur.

        Parameters
        ----------
        num_samples:
            Nombre d'images à générer.

        device:
            Appareil de calcul utilisé : CPU ou GPU.

        Returns
        -------
        generated_images:
            Images générées de forme [num_samples, 1, 28, 28].
        """

        if num_samples <= 0:
            raise ValueError(
                "num_samples doit être strictement supérieur à zéro."
            )

        # Si aucun appareil n'est fourni, on utilise celui du modèle.
        if device is None:
            device = next(self.parameters()).device

        # Génère des vecteurs aléatoires selon une loi normale standard.
        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=device,
        )

        # Transforme les vecteurs latents en images.
        generated_images = self.decode(z)

        return generated_images

    @staticmethod
    def _validate_images(x: Tensor) -> None:
        """
        Vérifie que les images respectent le format Fashion-MNIST.
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


def test_vae() -> None:
    """
    Effectue un test rapide de l'architecture.

    Ce test ne réalise aucun entraînement.

    Il vérifie seulement que :

    - les dimensions sont correctes ;
    - l'encodeur fonctionne ;
    - la reparamétrisation fonctionne ;
    - le décodeur fonctionne ;
    - la génération aléatoire fonctionne.
    """

    # Fixe la graine aléatoire afin de rendre le test reproductible.
    torch.manual_seed(42)

    batch_size = 8
    latent_dim = 16

    # Création du modèle.
    model = VAE(
        latent_dim=latent_dim,
        hidden_dim=256,
    )

    # Création d'un faux batch ayant le même format que Fashion-MNIST.
    fake_images = torch.rand(
        batch_size,
        1,
        28,
        28,
    )

    # Passage complet dans le modèle.
    reconstruction, mu, logvar, z = model(fake_images)

    # Vérification des dimensions obtenues.
    assert reconstruction.shape == (batch_size, 1, 28, 28)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)

    # Vérification de la méthode de génération.
    generated_images = model.sample(num_samples=10)

    assert generated_images.shape == (10, 1, 28, 28)

    # La Sigmoid doit produire des pixels entre 0 et 1.
    assert generated_images.min().item() >= 0.0
    assert generated_images.max().item() <= 1.0

    print("Test du VAE réussi.")
    print(f"Images d'entrée      : {tuple(fake_images.shape)}")
    print(f"Reconstructions      : {tuple(reconstruction.shape)}")
    print(f"Mu                   : {tuple(mu.shape)}")
    print(f"Logvar               : {tuple(logvar.shape)}")
    print(f"Vecteurs latents     : {tuple(z.shape)}")
    print(f"Images générées      : {tuple(generated_images.shape)}")
    print(
        "Intervalle des pixels : "
        f"[{generated_images.min().item():.4f}, "
        f"{generated_images.max().item():.4f}]"
    )


# Ce bloc est exécuté uniquement lorsque le fichier est lancé directement.
#
# Il ne sera pas exécuté lorsque VAE sera importé depuis un autre fichier.
if __name__ == "__main__":
    test_vae()