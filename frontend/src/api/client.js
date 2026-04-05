const API_BASE = '';

async function request(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  // Config
  getConfig: () => request('/get_config'),
  toggleDetection: () => request('/toggle_detection', { method: 'POST' }),
  toggleBatchedMode: () => request('/toggle_batched_mode', { method: 'POST' }),
  toggleTracking: () => request('/toggle_tracking', { method: 'POST' }),
  toggleAutoFps: () => request('/toggle_auto_fps', { method: 'POST' }),
  setTargetFps: (fps) => request('/set_target_fps', { method: 'POST', body: JSON.stringify({ fps }) }),

  // Metrics
  getMetrics: () => request('/api/metrics'),
  getBatchStats: () => request('/api/batch_stats'),
  getComparisonMetrics: () => request('/api/comparison_metrics'),

  // Zones
  getZones: () => request('/api/zones'),
  createZone: (zone) => request('/api/zones', { method: 'POST', body: JSON.stringify(zone) }),
  updateZone: (id, zone) => request(`/api/zones/${id}`, { method: 'PUT', body: JSON.stringify(zone) }),
  deleteZone: (id) => request(`/api/zones/${id}`, { method: 'DELETE' }),

  // Classes
  getClasses: () => request('/api/classes'),
  setClassFilter: (classes) => request('/api/classes/filter', { method: 'POST', body: JSON.stringify({ classes }) }),

  // Events
  getEvents: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return request(`/api/events?${qs}`);
  },
  getEventStats: () => request('/api/events/stats'),

  // Database
  queryDb: (sql) => request('/api/db/query', { method: 'POST', body: JSON.stringify({ sql }) }),
  llmQuery: (question) => request('/api/db/llm-query', { method: 'POST', body: JSON.stringify({ question }) }),
  getSchema: () => request('/api/db/schema'),

  // Settings
  getSettings: () => request('/api/settings'),
  updateSettings: (settings) => request('/api/settings', { method: 'PUT', body: JSON.stringify(settings) }),

  // Video feed URL (not a fetch, used as img src)
  videoFeedUrl: (channel) => `/video_feed/${channel}`,
};
