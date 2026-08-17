import { useEffect, useState } from 'react'
import { getFixtures, getMetrics, reconstructImage } from '../api.js'
import { DigitImage, FixturePicker, Notice, formatNumber } from '../components.jsx'

export default function CompareView({ dataset, models }) {
  const [fixtures, setFixtures] = useState(null)
  const [index, setIndex] = useState(null)
  const [results, setResults] = useState({})
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)

  // Les identifiants different d'un dataset a l'autre (mnist/vae_main contre
  // fashion/vae_beta_1) : on les resout par nature plutot qu'en dur.
  const mainModels = models.filter((model) => model.family === 'main')
  const vaeModel = mainModels.find((model) => !model.conditional)
  const cvaeModel = mainModels.find((model) => model.conditional)

  useEffect(() => {
    getFixtures(dataset.id)
      .then((payload) => {
        setFixtures(payload.by_class)
        const keys = Object.keys(payload.by_class)
        setIndex(payload.by_class[keys[5]]?.[0]?.index ?? 0)
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

  const comparison = metrics?.comparison
  const evaluation = metrics?.evaluation
  const original = results.vae?.original || results.cvae?.original

  return (
    <>
      <div className="card">
        <h2>Reconstruction : VAE contre CVAE</h2>
        <p className="subtitle">
          La même image réelle est encodée puis décodée par les deux modèles. Le CVAE reçoit en plus
          le label, il n'a donc pas besoin d'encoder l'identité de la classe dans z et peut consacrer
          sa capacité latente au style.
        </p>

        {error && <Notice kind="error">{error}</Notice>}

        {fixtures && (
          <div className="controls">
            <FixturePicker
              fixtures={fixtures}
              classNames={dataset.class_names}
              value={index}
              onChange={setIndex}
              label="Image de test à reconstruire"
            />
          </div>
        )}

        <div style={{ display: 'flex', gap: 40, flexWrap: 'wrap', marginTop: 20 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ width: 130 }}>
              <DigitImage src={original} alt="Original" />
            </div>
            <div className="caption">image réelle</div>
          </div>
          {vaeModel && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 130 }}>
                <DigitImage src={results.vae?.reconstruction} alt="Reconstruction VAE" />
              </div>
              <div className="caption">{vaeModel.label}</div>
            </div>
          )}
          {cvaeModel && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 130 }}>
                <DigitImage src={results.cvae?.reconstruction} alt="Reconstruction CVAE" />
              </div>
              <div className="caption">{cvaeModel.label}</div>
            </div>
          )}
        </div>

        {dataset.id === 'mnist' && (
          <Notice kind="warn">
            <strong>Le fond gris n'est pas un artefact d'affichage.</strong> Le dernier bloc du
            décodeur applique un ReLU avant le Tanh (
            <code className="mono">src/models/vae.py</code>), donc la sortie est bornée dans [0, 1)
            alors que les données sont normalisées dans [-1, 1] : le modèle ne peut jamais produire
            le noir. Les modèles Fashion-MNIST, qui se terminent par une Sigmoid, n'ont pas ce
            problème — la comparaison entre les deux onglets le montre directement.
          </Notice>
        )}
      </div>

      {comparison && (
        <div className="card">
          <h2>Métriques sur le test set complet</h2>
          <p className="subtitle">
            Issues de <code className="mono">reports/experiments/comparison.json</code>, produit par{' '}
            <code className="mono">scripts/evaluate.py</code> sur 10 000 images.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Modèle</th>
                  <th style={{ textAlign: 'right' }}>Reconstruction</th>
                  <th style={{ textAlign: 'right' }}>KL</th>
                  <th style={{ textAlign: 'right' }}>Images</th>
                </tr>
              </thead>
              <tbody>
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
              </tbody>
            </table>
          </div>

          {comparison.cvae?.controllability && (
            <Notice kind="info">
              Le score de contrôlabilité annoncé (
              {formatNumber(comparison.cvae.controllability.overall_accuracy * 100, 1)} %) est mesuré
              par un classifieur « plus proche centroïde » en espace pixel, qui s'effondre sur des
              images floues. L'onglet Génération montre que le conditionnement fonctionne bien en
              réalité — c'est la métrique qui est trop faible, pas le modèle.
            </Notice>
          )}
        </div>
      )}

      {evaluation && (
        <div className="card">
          <h2>Métriques sur le test set officiel</h2>
          <p className="subtitle">
            Issues de{' '}
            <code className="mono">
              projects/david_fashion_mnist/results/evaluation_metrics.csv
            </code>{' '}
            — 10 000 images du test set officiel Fashion-MNIST.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Modèle</th>
                  <th style={{ textAlign: 'right' }}>β</th>
                  <th style={{ textAlign: 'right' }}>Reconstruction</th>
                  <th style={{ textAlign: 'right' }}>KL</th>
                  <th style={{ textAlign: 'right' }}>Total</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.map((row) => (
                  <tr key={row.checkpoint}>
                    <td>{row.model_type}</td>
                    <td className="num">{row.beta}</td>
                    <td className="num">{formatNumber(row.test_reconstruction)}</td>
                    <td className="num">{formatNumber(row.test_kl)}</td>
                    <td className="num">{formatNumber(row.test_total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="subtitle" style={{ marginTop: 18 }}>
            À β égal, le CVAE obtient un KL plus faible que le VAE pour une reconstruction
            comparable : la classe lui étant fournie séparément, il a besoin de moins d'information
            dans z.
          </p>
        </div>
      )}
    </>
  )
}
