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
  const unregistered = models.filter((model) => model.registered === false)

  return (
    <div className="mx-auto max-w-6xl px-6 pb-16">
      <header className="mb-6 border-b border-line pt-8 pb-5">
        <h1 className="mb-1.5 text-2xl font-bold tracking-tight">
          VAE / CVAE — démonstration interactive
        </h1>
        <p className="m-0 text-sm text-dim">
          Génération d'images à partir des modèles entraînés par l'équipe. Chaque image est produite
          en direct par le décodeur, servi depuis le MLflow Model Registry.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-4 font-mono text-xs text-dim">
          <span className="flex items-center gap-1.5">
            <span
              className={`inline-block size-2 rounded-full ${error ? 'bg-[#e5674f]' : 'bg-ok'}`}
            />
            {error ? 'API injoignable' : loading ? 'Connexion…' : 'API en ligne'}
          </span>
          {health && (
            <>
              <span>torch {health.torch}</span>
              <span>device : {health.device}</span>
              <span>inférence : {health.inference}</span>
              <span>{health.registered_models?.length ?? 0} modèles au registre</span>
            </>
          )}
        </div>
      </header>

      {error && (
        <Notice kind="error">
          <strong>Impossible de joindre l'API.</strong> {error}
          <br />
          Vérifier que le backend tourne : <code className="font-mono">make dev-api</code>.
        </Notice>
      )}

      {datasets.length > 0 && (
        <div className="mb-5">
          <DatasetSwitch datasets={datasets} value={datasetId} onChange={setDatasetId} />
          {dataset && <p className="mt-2.5 text-[12.5px] text-dim">{dataset.description}</p>}
        </div>
      )}

      {unregistered.length > 0 && (
        <Notice kind="warn">
          {unregistered.length} modèle(s) ont un checkpoint mais ne sont pas dans le Model Registry.
          Les enregistrer avec{' '}
          <code className="font-mono">python scripts/register_models.py</code>.
        </Notice>
      )}

      <nav className="mb-7 flex flex-wrap gap-1">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`tab ${activeTab === tab.id ? 'tab-active' : ''}`}
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
      {loading && <div className="card text-dim">Chargement des modèles…</div>}
      {!loading && !error && models.length === 0 && (
        <div className="card text-dim">Aucun modèle disponible pour ce dataset.</div>
      )}
    </div>
  )
}
