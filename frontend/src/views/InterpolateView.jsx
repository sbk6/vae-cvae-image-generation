import { useEffect, useState } from 'react'
import { getFixtures, interpolateLatent } from '../api.js'
import { DigitImage, FixturePicker, ModelSelect, Notice } from '../components.jsx'

const STEPS = 16

export default function InterpolateView({ dataset, models }) {
  const mainModels = models.filter((model) => model.family === 'main')
  const usable = mainModels.length > 0 ? mainModels : models
  const [modelId, setModelId] = useState(usable[0]?.id)
  const [fixtures, setFixtures] = useState(null)
  const [sourceIndex, setSourceIndex] = useState(null)
  const [targetIndex, setTargetIndex] = useState(null)
  const [result, setResult] = useState(null)
  const [position, setPosition] = useState(0)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const classNames = dataset.class_names
  const nameOf = (index) => classNames[index] ?? index

  useEffect(() => {
    getFixtures(dataset.id)
      .then((payload) => {
        setFixtures(payload.by_class)
        // Deux classes eloignees par defaut, pour que la transition soit nette.
        const keys = Object.keys(payload.by_class)
        setSourceIndex(payload.by_class[keys[3]]?.[0]?.index ?? 0)
        setTargetIndex(payload.by_class[keys[Math.min(8, keys.length - 1)]]?.[0]?.index ?? 1)
      })
      .catch((err) => setError(err.message))
  }, [dataset.id])

  useEffect(() => {
    if (!modelId || sourceIndex === null || targetIndex === null) return
    setLoading(true)
    setError(null)
    interpolateLatent({
      model_id: modelId,
      source_index: sourceIndex,
      target_index: targetIndex,
      steps: STEPS,
    })
      .then((payload) => {
        setResult(payload)
        setPosition(0)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [modelId, sourceIndex, targetIndex])

  const currentImage = result?.images?.[position]
  const currentAlpha = result?.alphas?.[position]

  return (
    <div className="card">
      <h2 className="mb-1 text-[17px] font-semibold">Interpolation dans l'espace latent</h2>
      <p className="mb-5 text-[13.5px] text-dim">
        On encode deux images réelles vers leurs vecteurs mu, on interpole linéairement entre les
        deux, puis on décode chaque point. Une transition progressive prouve que l'espace latent est
        continu — et pas un simple dictionnaire d'images mémorisées.
      </p>

      <div className="mb-5 flex flex-wrap items-end gap-5">
        <ModelSelect models={usable} value={modelId} onChange={setModelId} />
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      {fixtures && (
        <div className="mb-5 flex flex-wrap items-start gap-5">
          <FixturePicker
            fixtures={fixtures}
            classNames={classNames}
            value={sourceIndex}
            onChange={setSourceIndex}
            label="Image de départ"
          />
          <FixturePicker
            fixtures={fixtures}
            classNames={classNames}
            value={targetIndex}
            onChange={setTargetIndex}
            label="Image d'arrivée"
          />
        </div>
      )}

      {result && (
        <>
          <div className="my-7 flex flex-wrap items-center gap-7">
            <div className="text-center">
              <div className="w-24">
                <DigitImage src={result.source.image} alt="Départ" />
              </div>
              <div className="caption">départ — {nameOf(result.source.label)}</div>
            </div>

            <div className="max-w-52 flex-1">
              <DigitImage src={currentImage} alt="Interpolation" />
              <div className="caption">
                α = {currentAlpha?.toFixed(2)} — étape {position + 1} / {STEPS}
              </div>
            </div>

            <div className="text-center">
              <div className="w-24">
                <DigitImage src={result.target.image} alt="Arrivée" />
              </div>
              <div className="caption">arrivée — {nameOf(result.target.label)}</div>
            </div>
          </div>

          <input
            type="range"
            className="range-accent"
            min={0}
            max={STEPS - 1}
            value={position}
            onChange={(event) => setPosition(Number(event.target.value))}
            disabled={loading}
          />

          <div className="mt-4 flex gap-1.5 overflow-x-auto pb-2">
            {result.images.map((image, index) => (
              <button
                key={index}
                type="button"
                onClick={() => setPosition(index)}
                className={`w-16 shrink-0 cursor-pointer rounded border-2 bg-transparent p-0 ${
                  index === position ? 'border-accent' : 'border-transparent'
                }`}
              >
                <DigitImage src={image} alt={`α = ${result.alphas[index].toFixed(2)}`} />
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
