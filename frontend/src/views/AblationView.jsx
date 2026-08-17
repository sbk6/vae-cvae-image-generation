import { useCallback, useEffect, useState } from 'react'
import { compareAblation, getMetrics } from '../api.js'
import { ClassSelector, DigitImage, Notice, formatNumber } from '../components.jsx'

export default function AblationView({ dataset }) {
  const [results, setResults] = useState([])
  const [series, setSeries] = useState(null)
  const [availableSeries, setAvailableSeries] = useState([])
  const [classLabel, setClassLabel] = useState(0)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const draw = useCallback(
    (requestedSeries, requestedClass) => {
      setLoading(true)
      setError(null)
      // Pas de seed : chaque appel tire un nouveau z, mais le meme z est envoye
      // a tous les modeles de la serie — c'est ce qui rend la comparaison honnete.
      compareAblation({
        dataset: dataset.id,
        ...(requestedSeries ? { series: requestedSeries } : {}),
        ...(requestedClass !== undefined ? { class_label: requestedClass } : {}),
      })
        .then((payload) => {
          setResults(payload.results)
          setSeries(payload.series)
          setAvailableSeries(payload.available_series || [])
        })
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false))
    },
    [dataset.id],
  )

  useEffect(() => {
    draw()
    getMetrics(dataset.id).then(setMetrics).catch(() => setMetrics(null))
  }, [draw, dataset.id])

  const ablation = metrics?.ablation
  const evaluation = metrics?.evaluation
  const conditional = series === 'cvae'

  return (
    <>
      <div className="card">
        <h2>Effet de β, à vecteur latent identique</h2>
        <p className="subtitle">
          Un seul z est tiré, puis décodé par tous les modèles de la série. À latent constant, tout
          écart visible vient uniquement de β.
        </p>

        <div className="controls">
          <button className="btn" onClick={() => draw(series, conditional ? classLabel : undefined)} disabled={loading}>
            {loading ? 'Décodage…' : 'Tirer un nouveau z'}
          </button>

          {availableSeries.length > 1 && (
            <div className="field">
              <label htmlFor="series-select">Série</label>
              <select
                id="series-select"
                value={series || ''}
                onChange={(event) => draw(event.target.value, classLabel)}
              >
                {availableSeries.map((name) => (
                  <option key={name} value={name}>
                    {name.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {conditional && (
          <div className="field" style={{ marginBottom: 20 }}>
            <label>Classe conditionnée</label>
            <ClassSelector
              value={classLabel}
              onChange={(value) => {
                setClassLabel(value)
                draw(series, value)
              }}
              classNames={dataset.class_names}
              disabled={loading}
            />
          </div>
        )}

        {error && <Notice kind="error">{error}</Notice>}

        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
          {results.map((result) => (
            <div key={result.model_id} style={{ textAlign: 'center', flex: '0 0 160px' }}>
              <DigitImage src={result.image} alt={result.label} />
              <div className="caption" style={{ fontWeight: 600, color: 'var(--text)' }}>
                {result.label}
              </div>
            </div>
          ))}
        </div>

        {dataset.id === 'mnist' && (
          <Notice kind="info">
            <strong>β = 5.0 produit une tache informe, et c'est le résultat attendu.</strong> Le
            terme KL domine la loss, le modèle ferme son espace latent (KL ≈ 0.56) et le décodeur
            produit à peu près la même image quel que soit z : c'est l'effondrement du postérieur.
            β = 0.1 reconstruit plus finement mais régularise mal ; β = 1.0 est le compromis retenu.
          </Notice>
        )}
        {dataset.id === 'fashion_mnist' && (
          <Notice kind="info">
            L'effondrement est plus progressif ici : le KL passe de 41 à 6.8 entre β = 0.1 et β = 4
            (contre 39 → 0.56 sur MNIST). Fashion-MNIST étant plus texturé, le modèle a davantage
            intérêt à conserver de l'information dans z même sous forte pénalité.
          </Notice>
        )}
      </div>

      {ablation && (
        <div className="card">
          <h2>Résultats chiffrés de l'ablation</h2>
          <p className="subtitle">
            Issus de <code className="mono">reports/experiments/ablation/results.json</code> —
            6 epochs par valeur de β, même seed, même architecture.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>β</th>
                  <th style={{ textAlign: 'right' }}>Loss val.</th>
                  <th style={{ textAlign: 'right' }}>Reconstruction</th>
                  <th style={{ textAlign: 'right' }}>KL</th>
                </tr>
              </thead>
              <tbody>
                {ablation.map((row) => (
                  <tr key={row.beta}>
                    <td className="mono">{row.beta}</td>
                    <td className="num">{formatNumber(row.final_val_loss)}</td>
                    <td className="num">{formatNumber(row.final_val_reconstruction)}</td>
                    <td className="num">{formatNumber(row.final_val_kl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="subtitle" style={{ marginTop: 18 }}>
            La colonne KL raconte toute l'histoire : 39.5 → 15.4 → 0.56. Plus β augmente, plus le
            modèle paie cher l'utilisation de son espace latent, jusqu'à cesser de s'en servir.
          </p>
        </div>
      )}

      {evaluation && (
        <div className="card">
          <h2>Résultats chiffrés de l'ablation</h2>
          <p className="subtitle">
            Issus de{' '}
            <code className="mono">projects/david_fashion_mnist/results/evaluation_metrics.csv</code>{' '}
            — test set officiel, 10 000 images.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Modèle</th>
                  <th style={{ textAlign: 'right' }}>β</th>
                  <th style={{ textAlign: 'right' }}>Reconstruction</th>
                  <th style={{ textAlign: 'right' }}>KL</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.map((row) => (
                  <tr key={row.checkpoint}>
                    <td>{row.model_type}</td>
                    <td className="num">{row.beta}</td>
                    <td className="num">{formatNumber(row.test_reconstruction)}</td>
                    <td className="num">{formatNumber(row.test_kl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
