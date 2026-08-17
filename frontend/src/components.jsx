// Petits composants partages par les differentes vues.

export function Notice({ kind = 'info', children }) {
  if (!children) return null
  return <div className={`notice ${kind}`}>{children}</div>
}

export function DigitImage({ src, alt, className = '' }) {
  if (!src) return <div className="skeleton" />
  return <img className={`digit ${className}`} src={src} alt={alt} />
}

// Les noms de classes viennent de l'API : chiffres "0".."9" pour MNIST,
// libelles ("Basket", "Sac"...) pour Fashion-MNIST. Rien n'est code en dur.
export function ClassSelector({ value, onChange, classNames = [], disabled = false }) {
  // Des libelles courts tiennent dans les pastilles carrees ; au-dela on
  // bascule sur une liste deroulante pour rester lisible.
  const compact = classNames.every((name) => name.length <= 2)

  if (compact) {
    return (
      <div className="class-selector">
        {classNames.map((name, index) => (
          <button
            key={index}
            type="button"
            disabled={disabled}
            className={`class-btn ${value === index ? 'active' : ''}`}
            onClick={() => onChange(index)}
          >
            {name}
          </button>
        ))}
      </div>
    )
  }

  return (
    <div className="class-selector wide">
      {classNames.map((name, index) => (
        <button
          key={index}
          type="button"
          disabled={disabled}
          className={`class-chip ${value === index ? 'active' : ''}`}
          onClick={() => onChange(index)}
        >
          {name}
        </button>
      ))}
    </div>
  )
}

export function ModelSelect({ models, value, onChange, label = 'Modèle', id = 'model-select' }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value || ''} onChange={(event) => onChange(event.target.value)}>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label}
          </option>
        ))}
      </select>
    </div>
  )
}

export function DatasetSwitch({ datasets, value, onChange }) {
  return (
    <div className="dataset-switch">
      {datasets.map((dataset) => (
        <button
          key={dataset.id}
          type="button"
          className={`dataset-btn ${value === dataset.id ? 'active' : ''}`}
          onClick={() => onChange(dataset.id)}
          title={dataset.description}
        >
          {dataset.label}
          <span className="dataset-count">{dataset.model_count}</span>
        </button>
      ))}
    </div>
  )
}

export function FixturePicker({ fixtures, classNames = [], value, onChange, label }) {
  // fixtures : { "0": [{index, image}], ... }
  const entries = Object.entries(fixtures || {})
  return (
    <div className="field" style={{ flex: 1, minWidth: 260 }}>
      <label>{label}</label>
      <div className="digit-grid" style={{ gridTemplateColumns: 'repeat(10, 1fr)', gap: 4 }}>
        {entries.map(([classLabel, items]) => {
          const item = items[0]
          if (!item) return null
          const selected = value === item.index
          const name = classNames[Number(classLabel)] ?? classLabel
          return (
            <button
              key={classLabel}
              type="button"
              title={name}
              onClick={() => onChange(item.index)}
              className={`fixture-btn ${selected ? 'active' : ''}`}
            >
              <img className="digit" src={item.image} alt={name} />
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(digits)
}
