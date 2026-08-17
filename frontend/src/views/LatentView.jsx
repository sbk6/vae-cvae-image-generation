import { useCallback, useEffect, useRef, useState } from 'react'
import { decodeLatent, encodeImage, getFixtures } from '../api.js'
import { ClassSelector, DigitImage, FixturePicker, ModelSelect, Notice } from '../components.jsx'

const RANGE = 3.5 // ~3 ecarts-types : au-dela on sort du support de la prior

export default function LatentView({ dataset, models }) {
  const mainModels = models.filter((model) => model.family === 'main')
  const usable = mainModels.length > 0 ? mainModels : models
  const [modelId, setModelId] = useState(usable[0]?.id)
  const [z, setZ] = useState(null)
  const [image, setImage] = useState(null)
  const [fixtures, setFixtures] = useState(null)
  const [classLabel, setClassLabel] = useState(0)
  const [error, setError] = useState(null)

  const selected = models.find((model) => model.id === modelId)
  const conditional = selected?.conditional
  const latentDim = selected?.latent_dim || 16

  // Une seule requete de decodage en vol a la fois : pendant un drag de slider
  // on ne veut pas empiler 50 requetes ni afficher une reponse perimee.
  const inFlight = useRef(false)
  const pending = useRef(null)

  useEffect(() => {
    getFixtures(dataset.id)
      .then((payload) => setFixtures(payload.by_class))
      .catch(() => {})
  }, [dataset.id])

  useEffect(() => {
    setZ(Array.from({ length: latentDim }, () => 0))
  }, [latentDim, modelId])

  const decode = useCallback(
    (vector) => {
      if (!modelId || !vector) return
      if (inFlight.current) {
        pending.current = vector
        return
      }
      inFlight.current = true
      decodeLatent({
        model_id: modelId,
        z: vector,
        ...(conditional ? { class_label: classLabel } : {}),
      })
        .then((payload) => setImage(payload.image))
        .catch((err) => setError(err.message))
        .finally(() => {
          inFlight.current = false
          if (pending.current) {
            const next = pending.current
            pending.current = null
            decode(next)
          }
        })
    },
    [modelId, conditional, classLabel],
  )

  useEffect(() => {
    if (z) decode(z)
  }, [z, decode])

  const updateDimension = (index, value) => {
    setZ((previous) => {
      const next = [...previous]
      next[index] = value
      return next
    })
  }

  const randomize = () => setZ(Array.from({ length: latentDim }, () => gaussian()))
  const reset = () => setZ(Array.from({ length: latentDim }, () => 0))

  const loadFromImage = (index) => {
    encodeImage({ model_id: modelId, index })
      .then((payload) => {
        setZ(payload.z.map((value) => clamp(value, -RANGE, RANGE)))
        if (conditional) setClassLabel(payload.true_label)
      })
      .catch((err) => setError(err.message))
  }

  return (
    <div className="card">
      <h2>Exploration de l'espace latent</h2>
      <p className="subtitle">
        Chaque curseur contrôle une des {latentDim} dimensions de z. L'image est redécodée à chaque
        mouvement. C'est la manière la plus directe de voir que le latent encode du <em>style</em> —
        épaisseur, inclinaison, proportions.
      </p>

      <div className="controls">
        <ModelSelect models={usable} value={modelId} onChange={setModelId} />
        <button className="btn" onClick={randomize}>
          Tirage aléatoire
        </button>
        <button className="btn btn-ghost" onClick={reset}>
          Remettre à zéro
        </button>
      </div>

      {conditional && (
        <div className="field" style={{ marginBottom: 20 }}>
          <label>Classe conditionnée</label>
          <ClassSelector value={classLabel} onChange={setClassLabel} classNames={dataset.class_names} />
        </div>
      )}

      {fixtures && (
        <div className="controls">
          <FixturePicker
            fixtures={fixtures}
            classNames={dataset.class_names}
            value={null}
            onChange={loadFromImage}
            label="Partir d'une image réelle (encode puis remplit les curseurs)"
          />
        </div>
      )}

      {error && <Notice kind="error">{error}</Notice>}

      <div className="latent-layout">
        <div className="latent-preview">
          <DigitImage src={image} alt="Image décodée" />
          <div className="caption">sortie du décodeur</div>
        </div>

        <div className="sliders">
          {(z || []).map((value, index) => (
            <div className="slider-row" key={index}>
              <span className="dim">z{index}</span>
              <input
                type="range"
                min={-RANGE}
                max={RANGE}
                step={0.05}
                value={value}
                onChange={(event) => updateDimension(index, Number(event.target.value))}
              />
              <span className="val">{value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Box-Muller : tirage normal centre reduit, pour rester coherent avec la prior.
function gaussian() {
  const u = Math.random() || Number.EPSILON
  const v = Math.random()
  return clamp(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v), -RANGE, RANGE)
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}
