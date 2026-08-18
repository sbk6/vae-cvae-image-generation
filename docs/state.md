# État du projet — VAE / CVAE pour la génération d'images

> Document de référence, à tenir à jour. Objectif : pouvoir répondre en 2 minutes à "où en est-on ?" sans relire tout l'historique.
> Dernière mise à jour : 2026-08-17.

---

## 1. Le sujet, expliqué simplement

Le projet demandé (sujet n°2 de `projets_master.md`) consiste à construire un modèle qui **apprend à compresser puis régénérer des images**, et à comparer deux variantes :

- **VAE (Auto-Encodeur Variationnel)** : on prend une image, un réseau la compresse en un petit vecteur de nombres (l'**espace latent**), puis un autre réseau essaie de reconstruire l'image à partir de ce vecteur. La particularité du VAE (par rapport à un simple auto-encodeur) est qu'il force cet espace latent à ressembler à une distribution statistique connue (une gaussienne), ce qui permet ensuite de **générer de nouvelles images** en tirant un vecteur au hasard et en le donnant au décodeur.
- **CVAE (VAE Conditionnel)** : même principe, mais on donne en plus l'**étiquette de classe** (par exemple "c'est un 7") au modèle. Résultat : on peut **choisir** quelle classe générer, ce que le VAE simple ne permet pas.

Ce que l'énoncé exige concrètement :

| # | Exigence de l'énoncé | En clair |
|---|---|---|
| 1 | Implémenter le VAE (encodeur, reparamétrisation, décodeur, loss ELBO) | Coder le modèle soi-même, pas juste appeler une lib toute faite |
| 2 | Implémenter le CVAE (label injecté à l'encodeur et au décodeur) | Variante contrôlable du VAE |
| 3 | Étude d'ablation sur **β** (poids du terme KL) | Tester au moins 3 valeurs de β et observer l'effet sur la qualité |
| 4 | Visualiser l'espace latent (t-SNE/UMAP) + interpolation entre 2 exemples | Vérifier que l'espace latent est "propre" et continu |
| 5 | Évaluation quantitative (reconstruction, KL, FID si possible) | Chiffrer la qualité, pas juste "ça a l'air bien" |
| 6 | Application web de démo (choix de classe → génération, slider d'interpolation) | Un petit outil pour manipuler le modèle sans coder |
| 7 | Livrables : tableau d'ablation β (≥3 valeurs), comparaison qualitative VAE vs CVAE | Deux résultats précis attendus dans le rapport |

Il faut aussi garder en tête les règles générales du cours (`projets_master/projets_master.md`), qui s'appliquent à tous les sujets :
- Rapport style article scientifique, 8 à 20 pages, avec une **section ablation obligatoire**.
- Code sur GitHub, reproductible (seeds fixés, un script par résultat).
- **Une application de démo est obligatoire** — "modèle seul sans application = REFUSÉ".
- Présentation orale de 20 min, focus sur méthode + ablation + résultats (la démo est secondaire à l'oral, mais reste obligatoire pour la note).
- Répartition claire des rôles dans l'équipe, sinon pénalité.

---

## 2. Ce qui est déjà fait ✅

| Exigence | Statut | Où le voir |
|---|---|---|
| VAE (encodeur/reparamétrisation/décodeur/ELBO) | ✅ Fait, codé à la main | [`src/models/vae.py`](../src/models/vae.py), [`src/losses/elbo.py`](../src/losses/elbo.py) |
| CVAE (conditionnement one-hot, générique) | ✅ Fait | [`src/models/cvae.py`](../src/models/cvae.py) |
| Entraînement réel sur MNIST | ✅ Fait — VAE et CVAE, 10 epochs sur les 60 000 images, latent_dim=16 | `reports/experiments/vae_main/`, `reports/experiments/cvae_main/` |
| Ablation sur β | ✅ Fait — 3 valeurs (0.1, 1.0, 5.0), même seed/archi | [`docs/RESULTATS.md`](RESULTATS.md), `reports/experiments/ablation/` |
| Visualisation espace latent (t-SNE) | ✅ Fait, pour VAE et CVAE | `reports/figures/latent_tsne_vae.png`, `latent_tsne_cvae.png` |
| Interpolation entre 2 exemples | ✅ Fait (3→8 et 1→7) | `reports/figures/interpolation_vae_3_to_8.png`, `interpolation_vae_1_to_7.png` |
| Évaluation quantitative (reconstruction, KL) | ✅ Fait | `reports/experiments/comparison.json`, section 6.4 de [`presentation_seance_4.md`](presentation_seance_4.md) |
| Comparaison qualitative VAE vs CVAE (livrable demandé) | ✅ Fait, avec tableau et analyse | Section 8 de [`presentation_seance_4.md`](presentation_seance_4.md) |
| Tests unitaires | ✅ 8 tests, tous verts | `tests/` |
| Pipeline reproductible (config YAML, seeds, scripts séparés) | ✅ Fait | `configs/`, `Makefile` |
| Documentation pédagogique en français | ✅ Fait | [`explanations.md`](explanations.md), [`presentation_seance_4.md`](presentation_seance_4.md) |

**Points forts à noter** : un vrai bug a été trouvé et corrigé (VAE et CVAE écrasaient le même fichier de checkpoint), et un résultat contre-intuitif (le proxy de contrôlabilité du CVAE donnait un score bas et trompeur) a été investigué plutôt que caché — bon matériel pour la section "Discussion et limites" du rapport final.

---

## 3. Ce qui n'est PAS encore fait ❌

| Exigence | Statut | Pourquoi c'est important |
|---|---|---|
| **Application web de démo** | ❌ Pas commencée (rien dans `requirements.txt` : pas de Flask/FastAPI/Streamlit/Gradio) | **Obligatoire** — sans elle, le règlement du cours dit explicitement "modèle seul sans démo = REFUSÉ" |
| **Fashion-MNIST** | ❌ Configuré (`dataset.name: fashion_mnist`) mais charge en réalité encore MNIST | C'est un des 3 datasets suggérés par l'énoncé (pas obligatoire d'en faire plusieurs, mais renforce le rapport) |
| **CelebA** | ❌ Non implémenté, lève une `NotImplementedError` | Idem — optionnel mais mentionné dans l'énoncé |
| **Score FID** | ❌ Non calculé | L'énoncé le précise "si les ressources le permettent" → non bloquant, mais actuellement remplacé par un proxy maison peu fiable |
| **Vrai classifieur pour mesurer la contrôlabilité du CVAE** | ❌ Remplacé par un proxy "plus proche centroïde" dont les limites sont documentées | Rendrait le chiffre de contrôlabilité crédible |
| **Rapport final (PDF, style article scientifique)** | ❌ Pas encore rédigé (seuls des comptes rendus de séance existent) | **Obligatoire pour la soumission**, 8-20 pages, structure imposée (voir `projets_master.md`) |
| **Répartition des rôles réelle** | ⚠️ Placeholder générique dans `plan.md`/`presentation_seance_4.md` ("Membre 1, 2, 3, 4") | Le règlement pénalise l'absence de répartition claire avec les vrais noms |
| **UMAP** (en plus du t-SNE) | ⚠️ Prévu dans `plan.md` mais seul t-SNE est implémenté | Non bloquant, le t-SNE seul répond à l'exigence ("projection 2D/t-SNE") |
| Entraînement plus long / GPU | ⚠️ Limité à 10 epochs sur CPU, la loss n'est pas totalement stabilisée | Amélioration possible si une machine GPU devient disponible |

---

## 4. Priorités suggérées pour la suite

1. **L'application web de démo** — c'est le seul point qui peut faire "refuser" le projet selon le règlement, et rien n'est commencé. Cible minimale : sélection d'une classe → image générée par le CVAE, + un slider qui interpole entre deux z latents. Une petite app Streamlit ou Gradio suffit largement (l'exigence est "allégée" au niveau Master).
2. **Rédiger le rapport final** (8-20 pages) — beaucoup de contenu existe déjà dans `presentation_seance_4.md` et `RESULTATS.md`, il s'agit surtout de le remettre au format article scientifique imposé.
3. **Remplacer les noms génériques ("Membre 1"...) par les vrais prénoms** dans la répartition des tâches.
4. Optionnel si le temps le permet : Fashion-MNIST (le point d'extension existe déjà dans le code), puis un vrai classifieur CNN pour mesurer la contrôlabilité du CVAE plus proprement que le proxy actuel.

---

## 5. Documents liés

- [`presentation_seance_4.md`](presentation_seance_4.md) — compte rendu détaillé de la dernière séance (résultats, chiffres, interprétation).
- [`RESULTATS.md`](RESULTATS.md) — tableau d'ablation généré automatiquement.
- [`explanations.md`](explanations.md) — explications techniques et FAQ.
- `../projets_master/projets_master.md` — énoncé complet du sujet et règles de soumission (hors du dépôt du projet).
