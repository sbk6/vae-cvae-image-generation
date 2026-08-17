import { useEffect, useState } from 'react'
import { getDatasets, getHealth, getModels } from './api.js'
import { DatasetSwitch, Notice } from './components.jsx'
import GenerateView from './views/GenerateView.jsx'
import InterpolateView from './views/InterpolateView.jsx'
import LatentView from './views/LatentView.jsx'
import CompareView from './views/CompareView.jsx'
import AblationView from './views/AblationView.jsx'

const TABS = [
  { id: 'generate', label: 'Génération', component: GenerateView },
  { id: 'interpolate', label: 'Interpolation', component: InterpolateView },
  { id: 'latent', label: 'Espace latent', component: LatentView },
  { id: 'compare', label: 'Reconstruction', component: CompareView },
  { id: 'ablation', label: 'Ablation β', component: AblationView },
]

export default function App() {
  const [health, setHealth] = useState(null)
  const [datasets, setDatasets] = useState([])
  const [datasetId, setDatasetId] = useState(null)
  const [models, setModels] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('generate')

  useEffect(() => {
    Promise.all([getHealth(), getDatasets()])
      .then(([healthPayload, datasetsPayload]) => {
        setHealth(healthPayload)
        setDatasets(datasetsPayload.datasets)
        setDatasetId(datasetsPayload.datasets[0]?.id ?? null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  // Les modeles sont rechargés a chaque changement de dataset : leurs
  // identifiants sont namespacés (mnist/…, fashion/…) et ne se mélangent pas.
  useEffect(() => {
    if (!datasetId) return
    setModels([])
    getModels(datasetId)
      .then((payload) => setModels(payload.models))
      .catch((err) => setError(err.message))
  }, [datasetId])

  const dataset = datasets.find((item) => item.id === datasetId)
  const ActiveComponent = TABS.find((tab) => tab.id === activeTab)?.component
  const ready = !loading && !error && dataset && models.length > 0

  return (
    <div className="app">
      <header className="app-header">
        <h1>VAE / CVAE — démonstration interactive</h1>
        <p>
          Génération d'images à partir des modèles entraînés par l'équipe. Chaque image est produite
          en direct par le décodeur, pas rechargée depuis un PNG.
        </p>
        <div className="status-line">
          <span>
            <span className={`status-dot ${error ? 'down' : ''}`} />
            {error ? 'API injoignable' : loading ? 'Connexion…' : 'API en ligne'}
          </span>
          {health && (
            <>
              <span>torch {health.torch}</span>
              <span>device: {health.device}</span>
              <span>{health.models_available.length} modèles</span>
            </>
          )}
        </div>
      </header>

      {error && (
        <Notice kind="error">
          <strong>Impossible de joindre l'API.</strong> {error}
          <br />
          Vérifier que le backend tourne : <code className="mono">make dev-api</code> (ou{' '}
          <code className="mono">python -m backend.app</code>).
        </Notice>
      )}

      {datasets.length > 0 && (
        <div className="dataset-bar">
          <DatasetSwitch datasets={datasets} value={datasetId} onChange={setDatasetId} />
          {dataset && <p className="dataset-description">{dataset.description}</p>}
        </div>
      )}

      {datasets.length === 1 && (
        <Notice kind="warn">
          Un seul dataset est disponible. Pour activer Fashion-MNIST, déposer les checkpoints de
          David dans <code className="mono">projects/david_fashion_mnist/checkpoints/</code> — ils
          sont détectés automatiquement au démarrage.
        </Notice>
      )}

      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {ready && (
        <ActiveComponent
          // Remonter la vue a zero quand le dataset change : les index de
          // fixtures et les latents ne sont pas transposables d'un dataset a
          // l'autre, garder l'etat produirait des requetes incoherentes.
          key={`${activeTab}-${datasetId}`}
          dataset={dataset}
          models={models}
        />
      )}
      {loading && <div className="card muted">Chargement des modèles…</div>}
      {!loading && !error && models.length === 0 && (
        <div className="card muted">Aucun modèle disponible pour ce dataset.</div>
      )}
    </div>
  )
}
