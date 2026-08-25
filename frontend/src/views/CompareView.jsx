import { useEffect, useState } from 'react'
import { getFixtures, getMetrics, reconstructImage } from '../api.js'
import {
  DataTable,
  DigitImage,
  FixturePicker,
  Notice,
  NoWeightsNotice,
  formatNumber,
} from '../components.jsx'

export default function CompareView({ dataset, models }) {
  const [fixtures, setFixtures] = useState(null)
  const [index, setIndex] = useState(null)
  const [results, setResults] = useState({})
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)

  // Les identifiants different d'un dataset a l'autre (mnist/vae_main contre
  // fashion/vae_beta_1_seed42_final) : on les resout par nature, pas en dur.
  const mainModels = models.filter((model) => model.family === 'main')
  const vaeModel = mainModels.find((model) => !model.conditional)
  const cvaeModel = mainModels.find((model) => model.conditional)

  useEffect(() => {
    getFixtures(dataset.id)
      .then((payload) => {
        setFixtures(payload.by_class)
        const keys = Object.keys(payload.by_class)
        setIndex(payload.by_class[keys[Math.min(5, keys.length - 1)]]?.[0]?.index ?? 0)
      })
      .catch((err) => setError(err.message))
    getMetrics(dataset.id).then(setMetrics).catch(() => setMetrics(null))
  }, [dataset.id])

  useEffect(() => {
    if (index === null) return
    const requests = []
    if (vaeModel) requests.push(reconstructImage({ model_id: vaeModel.id, index }))
    if (cvaeModel) requests.push(reconstructImage({ model_id: cvaeModel.id, index }))
    if (requests.length === 0) return

    Promise.all(requests)
      .then((responses) => {
        const next = {}
        let cursor = 0
        if (vaeModel) next.vae = responses[cursor++]
        if (cvaeModel) next.cvae = responses[cursor++]
        setResults(next)
      })
      .catch((err) => setError(err.message))
  }, [index, vaeModel?.id, cvaeModel?.id])

  // Les tableaux de metriques restent affiches meme sans poids : ils viennent
  // des fichiers de resultats, pas des modeles.
  const noWeights = models.length === 0

  const comparison = metrics?.comparison
  const evaluation = metrics?.evaluation
  const original = results.vae?.original || results.cvae?.original

  return (
    <>
      <div className="card">
        <h2 className="mb-1 text-[17px] font-semibold">Reconstruction : VAE contre CVAE</h2>
        <p className="mb-5 text-[13.5px] text-dim">
          La même image réelle est encodée puis décodée par les deux modèles. Le CVAE reçoit en plus
          le label, il n'a donc pas besoin d'encoder l'identité de la classe dans z et peut consacrer
          sa capacité latente au style.
        </p>

        {error && <Notice kind="error">{error}</Notice>}

        {noWeights && <NoWeightsNotice dataset={dataset} />}

        {!noWeights && fixtures && (
          <div className="mb-5 flex flex-wrap items-end gap-5">
            <FixturePicker
              fixtures={fixtures}
              classNames={dataset.class_names}
              value={index}
              onChange={setIndex}
              label="Image de test à reconstruire"
            />
          </div>
        )}

        <div className={`mt-5 flex flex-wrap gap-10 ${noWeights ? 'hidden' : ''}`}>
          <div className="w-32 text-center">
            <DigitImage src={original} alt="Original" />
            <div className="caption">image réelle</div>
          </div>
          {vaeModel && (
            <div className="w-32 text-center">
              <DigitImage src={results.vae?.reconstruction} alt="Reconstruction VAE" />
              <div className="caption">{vaeModel.label}</div>
            </div>
          )}
          {cvaeModel && (
            <div className="w-32 text-center">
              <DigitImage src={results.cvae?.reconstruction} alt="Reconstruction CVAE" />
              <div className="caption">{cvaeModel.label}</div>
            </div>
          )}
        </div>

        {dataset.id === 'mnist' && (
          <div className="mt-5">
            <Notice kind="warn">
              <strong>Le fond gris n'est pas un artefact d'affichage.</strong> Le dernier bloc du
              décodeur applique un ReLU avant le Tanh (
              <code className="font-mono">src/models/vae.py</code>), donc la sortie est bornée dans
              [0, 1) alors que les données sont normalisées dans [-1, 1] : le modèle ne peut jamais
              produire le noir. Les modèles Fashion-MNIST, qui se terminent par une Sigmoid, n'ont
              pas ce problème — basculer entre les datasets le montre directement.
            </Notice>
          </div>
        )}
      </div>

      {comparison && (
        <div className="card">
          <h2 className="mb-1 text-[17px] font-semibold">Métriques sur le test set complet</h2>
          <p className="mb-5 text-[13.5px] text-dim">
            Issues des fichiers de résultats produits par les scripts d'évaluation, sur 10 000
            images.
          </p>
          <DataTable headers={['Modèle', 'Reconstruction', 'KL', 'Images']}>
            <tr>
              <td>VAE</td>
              <td className="num">{formatNumber(comparison.vae?.reconstruction)}</td>
              <td className="num">{formatNumber(comparison.vae?.kl)}</td>
              <td className="num">{comparison.vae?.n_test}</td>
            </tr>
            <tr>
              <td>CVAE</td>
              <td className="num">{formatNumber(comparison.cvae?.reconstruction)}</td>
              <td className="num">{formatNumber(comparison.cvae?.kl)}</td>
              <td className="num">{comparison.cvae?.n_test}</td>
            </tr>
          </DataTable>

          {comparison.cvae?.controllability && (
            <div className="mt-5">
              <Notice kind="info">
                Le score de contrôlabilité annoncé (
                {formatNumber(comparison.cvae.controllability.overall_accuracy * 100, 1)} %) est
                mesuré par un classifieur « plus proche centroïde » en espace pixel, qui s'effondre
                sur des images floues. L'onglet Génération montre que le conditionnement fonctionne
                bien en réalité — c'est la métrique qui est trop faible, pas le modèle.
              </Notice>
            </div>
          )}
        </div>
      )}

      {evaluation && (
        <div className="card">
          <h2 className="mb-1 text-[17px] font-semibold">Métriques sur le test set officiel</h2>
          <p className="mb-5 text-[13.5px] text-dim">
            Issues de{' '}
            <code className="font-mono">
              projects/david_fashion_mnist/results/evaluation_metrics.csv
            </code>{' '}
            — 10 000 images du test set officiel Fashion-MNIST.
          </p>
          <DataTable headers={['Modèle', 'β', 'Reconstruction', 'KL', 'Total']}>
            {evaluation.map((row) => (
              <tr key={row.checkpoint}>
                <td>{row.model_type}</td>
                <td className="num">{row.beta}</td>
                <td className="num">{formatNumber(row.test_reconstruction)}</td>
                <td className="num">{formatNumber(row.test_kl)}</td>
                <td className="num">{formatNumber(row.test_total)}</td>
              </tr>
            ))}
          </DataTable>
          <p className="mt-4 text-[13.5px] text-dim">
            À β égal, le CVAE obtient un KL plus faible que le VAE pour une reconstruction
            comparable : la classe lui étant fournie séparément, il a besoin de moins d'information
            dans z.
          </p>
        </div>
      )}
    </>
  )
}
