// Petits composants partages par les differentes vues.

const NOTICE_STYLES = {
  error: 'notice-error',
  info: 'notice-info',
  warn: 'notice-warn',
}

export function Notice({ kind = 'info', children }) {
  if (!children) return null
  return <div className={`notice ${NOTICE_STYLES[kind] ?? NOTICE_STYLES.info}`}>{children}</div>
}

export function DigitImage({ src, alt, className = '' }) {
  if (!src) return <div className="skeleton" />
  return <img className={`digit ${className}`} src={src} alt={alt} />
}

export function Field({ label, htmlFor, children, className = '' }) {
  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      <label className="field-label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
    </div>
  )
}

// Les noms de classes viennent de l'API : chiffres "0".."9" pour MNIST,
// libelles ("Basket", "Sac") pour Fashion-MNIST, combinaisons d'attributs
// pour CelebA. Rien n'est code en dur ici.
export function ClassSelector({ value, onChange, classNames = [], disabled = false }) {
  // Des libelles courts tiennent dans des pastilles carrees ; au-dela on
  // bascule sur des puces textuelles pour rester lisible.
  const compact = classNames.length > 0 && classNames.every((name) => name.length <= 2)

  return (
    <div className="flex flex-wrap gap-1.5">
      {classNames.map((name, index) => {
        const active = value === index
        const base = compact ? 'square-btn' : 'chip'
        const activeClass = compact ? 'square-btn-active' : 'chip-active'
        return (
          <button
            key={index}
            type="button"
            disabled={disabled}
            title={name}
            className={`${base} ${active ? activeClass : ''}`}
            onClick={() => onChange(index)}
          >
            {name}
          </button>
        )
      })}
    </div>
  )
}

export function ModelSelect({ models, value, onChange, label = 'Modèle', id = 'model-select' }) {
  return (
    <Field label={label} htmlFor={id}>
      <select
        id={id}
        className="input-base"
        value={value || ''}
        onChange={(event) => onChange(event.target.value)}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label}
          </option>
        ))}
      </select>
    </Field>
  )
}

export function DatasetSwitch({ datasets, value, onChange }) {
  return (
    <div className="inline-flex gap-1 rounded-xl border border-line bg-surface p-1">
      {datasets.map((dataset) => {
        const active = value === dataset.id
        return (
          <button
            key={dataset.id}
            type="button"
            title={dataset.description}
            onClick={() => onChange(dataset.id)}
            className={`flex cursor-pointer items-center gap-2 rounded-lg border-none px-4 py-2
              font-semibold transition-colors ${
                active ? 'bg-accent text-[#0b1020]' : 'bg-transparent text-dim hover:text-content'
              }`}
          >
            {dataset.label}
            <span
              className={`rounded-full px-1.5 py-px font-mono text-[11px] ${
                active ? 'bg-black/20' : 'bg-white/10'
              }`}
            >
              {dataset.model_count}
            </span>
          </button>
        )
      })}
    </div>
  )
}

export function FixturePicker({ fixtures, classNames = [], value, onChange, label }) {
  // fixtures : { "0": [{index, image}], ... }
  const entries = Object.entries(fixtures || {})
  return (
    <Field label={label} className="min-w-64 flex-1">
      <div className="grid grid-cols-10 gap-1">
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
              className={`cursor-pointer rounded-md border-2 bg-transparent p-0.5 transition-colors ${
                selected ? 'border-accent' : 'border-transparent hover:border-accent-dim'
              }`}
            >
              <img className="digit" src={item.image} alt={name} />
            </button>
          )
        })}
      </div>
    </Field>
  )
}

export function DataTable({ headers, children }) {
  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={header} className={index === 0 ? '' : 'text-right!'}>
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

// Affiché quand un dataset est annoncé par l'API mais qu'aucun poids n'est
// encore enregistré : son auteur a livré son code, ses images et ses
// résultats, seuls les checkpoints manquent.
export function NoWeightsNotice({ dataset, children }) {
  return (
    <Notice kind="warn">
      <strong>Aucun modèle {dataset.label} n'est encore chargé.</strong> Le code
      d'intégration, les images réelles et les résultats d'évaluation sont en place —
      il ne manque que les poids entraînés.
      <br />
      <br />
      Déposer un <code className="font-mono">best_checkpoint.pth</code> dans le
      sous-projet, puis lancer{' '}
      <code className="font-mono">python scripts/register_models.py</code>. Un seul
      fichier suffit à activer cet écran.
      {children}
    </Notice>
  )
}

export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(digits)
}
