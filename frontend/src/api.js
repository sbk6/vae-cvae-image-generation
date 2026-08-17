// Client HTTP de l'API Flask.
// Tous les chemins sont relatifs : en dev Vite proxifie /api vers :8000,
// en demo Flask sert le build et l'API sur le meme port.

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  let payload
  try {
    payload = await response.json()
  } catch {
    throw new Error(`Réponse illisible du serveur (HTTP ${response.status})`)
  }

  if (!response.ok) {
    throw new Error(payload?.error || `Erreur HTTP ${response.status}`)
  }
  return payload
}

const post = (path, body) =>
  request(path, { method: 'POST', body: JSON.stringify(body) })

const withDataset = (path, dataset) =>
  dataset ? `${path}?dataset=${encodeURIComponent(dataset)}` : path

export const getHealth = () => request('/api/health')
export const getDatasets = () => request('/api/datasets')

export const getModels = (dataset) => request(withDataset('/api/models', dataset))
export const getMetrics = (dataset) => request(withDataset('/api/metrics', dataset))
export const getFixtures = (dataset) => request(withDataset('/api/fixtures', dataset))

export const sampleImages = (body) => post('/api/sample', body)
export const decodeLatent = (body) => post('/api/decode', body)
export const encodeImage = (body) => post('/api/encode', body)
export const reconstructImage = (body) => post('/api/reconstruct', body)
export const interpolateLatent = (body) => post('/api/interpolate', body)
export const compareAblation = (body) => post('/api/ablation/compare', body)
