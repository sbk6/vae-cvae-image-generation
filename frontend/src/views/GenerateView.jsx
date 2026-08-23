import { useCallback, useEffect, useState } from 'react'
import { sampleImages } from '../api.js'
import { ClassSelector, DigitImage, Field, ModelSelect, Notice } from '../components.jsx'

export default function GenerateView({ dataset, models }) {
  const mainModels = models.filter((model) => model.family === 'main')
  const usable = mainModels.length > 0 ? mainModels : models
  const [modelId, setModelId] = useState(usable.find((m) => m.conditional)?.id || usable[0]?.id)
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
