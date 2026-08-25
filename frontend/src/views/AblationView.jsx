import { useCallback, useEffect, useState } from 'react'
import { compareAblation, getMetrics } from '../api.js'
import {
  ClassSelector,
  DataTable,
  DigitImage,
  Field,
  Notice,
  NoWeightsNotice,
  formatNumber,
} from '../components.jsx'

export default function AblationView({ dataset, models }) {
  const [results, setResults] = useState([])
  const [series, setSeries] = useState(null)
  const [availableSeries, setAvailableSeries] = useState([])
  const [classLabel, setClassLabel] = useState(0)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [incomplete, setIncomplete] = useState(null)
  const [loading, setLoading] = useState(false)

  const draw = useCallback(
    (requestedSeries, requestedClass) => {
      setLoading(true)
      setError(null)
      setIncomplete(null)
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
        .catch((err) => {
          // Une serie incomplete n'est pas une panne : c'est un checkpoint
          // manquant, avec une action claire pour le recuperer.
          setResults([])
          if (/incomplète|incomplete|Aucun checkpoint/i.test(err.message)) {
            setIncomplete(err.message)
          } else {
            setError(err.message)
          }
        })
        .finally(() => setLoading(false))
    },
    [dataset.id],
  )

  useEffect(() => {
    draw()
    getMetrics(dataset.id).then(setMetrics).catch(() => setMetrics(null))
  }, [draw, dataset.id])

  const noWeights = models.length === 0

  const ablation = metrics?.ablation
  const evaluation = metrics?.evaluation
  const conditional = series === 'cvae'

  return (
    <>
      <div className="card">
        <h2 className="mb-1 text-[17px] font-semibold">Effet de β, à vecteur latent identique</h2>
        <p className="mb-5 text-[13.5px] text-dim">
          Un seul z est tiré, puis décodé par tous les modèles de la série. À latent constant, tout
          écart visible vient uniquement de β.
        </p>

        {noWeights && <NoWeightsNotice dataset={dataset} />}

        <div className={`mb-5 flex flex-wrap items-end gap-5 ${noWeights ? 'hidden' : ''}`}>
          <button
            className="btn"
            onClick={() => draw(series, conditional ? classLabel : undefined)}
            disabled={loading || Boolean(incomplete)}
          >
            {loading ? 'Décodage…' : 'Tirer un nouveau z'}
          </button>

          {availableSeries.length > 1 && (
            <Field label="Série" htmlFor="series-select">
              <select
                id="series-select"
                className="input-base"
                value={series || ''}
                onChange={(event) => draw(event.target.value, classLabel)}
              >
                {availableSeries.map((name) => (
                  <option key={name} value={name}>
                    {name.toUpperCase()}
                  </option>
                ))}
              </select>
            </Field>
          )}
        </div>

        {conditional && (
          <Field label="Classe conditionnée" className="mb-5">
            <ClassSelector
              value={classLabel}
              onChange={(value) => {
                setClassLabel(value)
                draw(series, value)
              }}
              classNames={dataset.class_names}
              disabled={loading}
            />
          </Field>
        )}

        {error && <Notice kind="error">{error}</Notice>}

        {incomplete && (
          <Notice kind="warn">
            <strong>Série d'ablation incomplète pour ce dataset.</strong> Un seul β est disponible,
            il en faut au moins deux pour que la comparaison ait un sens. Déposer les checkpoints des
            autres valeurs de β puis relancer{' '}
            <code className="font-mono">python scripts/register_models.py</code> suffira à activer
            cet écran. Les chiffres correspondants restent consultables ci-dessous.
          </Notice>
        )}

        <div className="flex flex-wrap gap-8">
          {results.map((result) => (
            <div key={result.model_id} className="w-40 text-center">
              <DigitImage src={result.image} alt={result.label} />
              <div className="mt-1.5 text-center text-xs font-semibold">{result.label}</div>
            </div>
          ))}
        </div>

        {results.length > 0 && dataset.id === 'mnist' && (
          <div className="mt-5">
            <Notice kind="info">
              <strong>β = 5.0 produit une tache informe, et c'est le résultat attendu.</strong> Le
              terme KL domine la loss, le modèle ferme son espace latent (KL ≈ 0.56) et le décodeur
              produit à peu près la même image quel que soit z : c'est l'effondrement du postérieur.
              β = 0.1 reconstruit plus finement mais régularise mal ; β = 1.0 est le compromis
              retenu.
            </Notice>
          </div>
        )}
      </div>

      {ablation && (
        <div className="card">
          <h2 className="mb-1 text-[17px] font-semibold">Résultats chiffrés de l'ablation</h2>
          <p className="mb-5 text-[13.5px] text-dim">
            Même seed, même architecture, seul β change d'une ligne à l'autre.
          </p>
          <DataTable headers={['β', 'Loss val.', 'Reconstruction', 'KL']}>
            {ablation.map((row) => (
              <tr key={row.beta}>
                <td className="font-mono">{row.beta}</td>
                <td className="num">{formatNumber(row.val_loss)}</td>
                <td className="num">{formatNumber(row.val_reconstruction)}</td>
                <td className="num">{formatNumber(row.val_kl)}</td>
              </tr>
            ))}
          </DataTable>
          <p className="mt-4 text-[13.5px] text-dim">
            La colonne KL raconte toute l'histoire : plus β augmente, plus le modèle paie cher
            l'utilisation de son espace latent, jusqu'à cesser de s'en servir.
          </p>
        </div>
      )}

      {evaluation && (
        <div className="card">
          <h2 className="mb-1 text-[17px] font-semibold">Résultats chiffrés de l'ablation</h2>
          <p className="mb-5 text-[13.5px] text-dim">
            Issus de{' '}
            <code className="font-mono">
              projects/david_fashion_mnist/results/evaluation_metrics.csv
            </code>{' '}
            — test set officiel, 10 000 images.
          </p>
          <DataTable headers={['Modèle', 'β', 'Reconstruction', 'KL']}>
            {evaluation.map((row) => (
              <tr key={row.checkpoint}>
                <td>{row.model_type}</td>
                <td className="num">{row.beta}</td>
                <td className="num">{formatNumber(row.test_reconstruction)}</td>
                <td className="num">{formatNumber(row.test_kl)}</td>
              </tr>
            ))}
          </DataTable>
        </div>
      )}
    </>
  )
}
