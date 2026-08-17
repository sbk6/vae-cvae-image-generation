import { useCallback, useEffect, useState } from 'react'
import { sampleImages } from '../api.js'
import { ClassSelector, DigitImage, ModelSelect, Notice } from '../components.jsx'

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

  return (
    <div className="card">
      <h2>Génération à partir de la prior</h2>
      <p className="subtitle">
        On tire z ~ N(0, I) puis on décode. Le CVAE reçoit en plus la classe demandée : c'est ce qui
        permet de <em>choisir</em> ce qui est généré, ce que le VAE ne sait pas faire.
      </p>

      <div className="controls">
        <ModelSelect models={usable} value={modelId} onChange={setModelId} />

        <div className="field">
          <label htmlFor="count-select">Nombre d'images</label>
          <select
            id="count-select"
            value={count}
            onChange={(event) => setCount(Number(event.target.value))}
          >
            {[8, 16, 32, 64].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>

        <button className="btn" onClick={run} disabled={loading}>
          {loading ? 'Génération…' : 'Nouveau tirage'}
        </button>
      </div>

      {conditional ? (
        <div className="field" style={{ marginBottom: 22 }}>
          <label>Classe conditionnée</label>
          <ClassSelector
            value={classLabel}
            onChange={setClassLabel}
            classNames={dataset.class_names}
            disabled={loading}
          />
        </div>
      ) : (
        <Notice kind="warn">
          Le VAE n'est pas conditionnel : on ne peut pas lui demander une classe précise. Le tirage
          suit la distribution apprise — c'est exactement la limite que le CVAE lève.
        </Notice>
      )}

      {error && <Notice kind="error">{error}</Notice>}

      <div
        className="digit-grid"
        style={{ gridTemplateColumns: `repeat(${Math.min(count, 8)}, 1fr)` }}
      >
        {(loading && images.length === 0 ? Array.from({ length: count }) : images).map(
          (image, index) => (
            <div className="digit-cell" key={index}>
              <DigitImage src={image} alt={`Échantillon ${index + 1}`} />
            </div>
          ),
        )}
      </div>
    </div>
  )
}
