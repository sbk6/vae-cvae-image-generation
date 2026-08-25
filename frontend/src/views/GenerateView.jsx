import { useCallback, useEffect, useState } from 'react'
import { sampleImages } from '../api.js'
import {
  ClassSelector,
  DigitImage,
  Field,
  ModelSelect,
  Notice,
  NoWeightsNotice,
} from '../components.jsx'

export default function GenerateView({ dataset, models }) {
  const mainModels = models.filter((model) => model.family === 'main')
  const usable = mainModels.length > 0 ? mainModels : models
  const [modelId, setModelId] = useState(usable.find((m) => m.conditional)?.id || usable[0]?.id)

  // `models` arrive apres le montage et change avec le dataset. Sans ce
  // rattrapage, l'identifiant fige a l'initialisation resterait celui du
  // dataset precedent, et la vue interrogerait le mauvais modele.
  useEffect(() => {
    const main = models.filter((model) => model.family === 'main')
    const pool = main.length > 0 ? main : models
    if (pool.length > 0 && !pool.some((model) => model.id === modelId)) {
      setModelId(pool.find((model) => model.conditional)?.id || pool[0]?.id)
    }
  }, [models, modelId])
  const [classLabel, setClassLabel] = useState(0)
  const [count, setCount] = useState(16)
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const selected = models.find((model) => model.id === modelId)
  const conditional = selected?.conditional

  const run = useCallback(() => {
    if (!modelId) return
    setLoading(true)
    setError(null)
    sampleImages({
      model_id: modelId,
      n: count,
      ...(conditional ? { class_label: classLabel } : {}),
    })
      .then((payload) => setImages(payload.images))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [modelId, count, classLabel, conditional])

  useEffect(() => {
    run()
  }, [run])

  // Tous les hooks ci-dessus sont appeles inconditionnellement : React
  // interdit qu'un rendu en execute un nombre different du precedent, et
  // `models` passe de vide a rempli des que l'API repond.
  // Sans poids enregistres, cet ecran n'a rien a produire. On l'annonce
  // au lieu d'afficher des controles inertes.
  if (models.length === 0) {
    return (
      <div className="card">
        <h2 className="mb-1 text-[17px] font-semibold">Génération à partir de la prior</h2>
        <p className="mb-5 text-[13.5px] text-dim">
          Cet écran génère des images en direct : il lui faut au moins un modèle chargé.
        </p>
        <NoWeightsNotice dataset={dataset} />
      </div>
    )
  }

  const columns = Math.min(count, 8)

  return (
    <div className="card">
      <h2 className="mb-1 text-[17px] font-semibold">Génération à partir de la prior</h2>
      <p className="mb-5 text-[13.5px] text-dim">
        On tire z ~ N(0, I) puis on décode. Le CVAE reçoit en plus la classe demandée : c'est ce qui
        permet de <em>choisir</em> ce qui est généré, ce que le VAE ne sait pas faire.
      </p>

      <div className="mb-5 flex flex-wrap items-end gap-5">
        <ModelSelect models={usable} value={modelId} onChange={setModelId} />

        <Field label="Nombre d'images" htmlFor="count-select">
          <select
            id="count-select"
            className="input-base"
            value={count}
            onChange={(event) => setCount(Number(event.target.value))}
          >
            {[8, 16, 32, 64].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </Field>

        <button className="btn" onClick={run} disabled={loading}>
          {loading ? 'Génération…' : 'Nouveau tirage'}
        </button>
      </div>

      {conditional ? (
        <Field label="Classe conditionnée" className="mb-5">
          <ClassSelector
            value={classLabel}
            onChange={setClassLabel}
            classNames={dataset.class_names}
            disabled={loading}
          />
        </Field>
      ) : (
        <Notice kind="warn">
          Le VAE n'est pas conditionnel : on ne peut pas lui demander une classe précise. Le tirage
          suit la distribution apprise — c'est exactement la limite que le CVAE lève.
        </Notice>
      )}

      {error && <Notice kind="error">{error}</Notice>}

      <div
        className="grid gap-2"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      >
        {(loading && images.length === 0 ? Array.from({ length: count }) : images).map(
          (image, index) => (
            <DigitImage key={index} src={image} alt={`Échantillon ${index + 1}`} />
          ),
        )}
      </div>
    </div>
  )
}
