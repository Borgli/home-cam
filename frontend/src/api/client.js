const API_BASE = '';

async function request(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// Like request(), but hands back the status and the JSON body (with its error message) instead of throwing
async function fetchJson(url, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${url}`, options);
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  } catch {
    return { ok: false, status: 0, data: { error: 'Backend unreachable' } };
  }
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

  // 4K view: the NVR's 4K main stream, at most 2 cameras at once (the NVR's limit)
  hdStart: (channel) => fetchJson(`/api/hd/${channel}`, { method: 'POST' }),
  hdStop: (channel) => fetch(`${API_BASE}/api/hd/${channel}`, { method: 'DELETE' }).catch(() => {}),
  // JPEG of the region x0..x1, y0..y1 (0-1) of the newest 4K frame, `width` pixels wide
  hdFrameUrl: (channel, { x0, y0, x1, y1, width }) => `${API_BASE}/api/hd/${channel}/frame?${new URLSearchParams({
    x0: x0.toFixed(5), y0: y0.toFixed(5), x1: x1.toFixed(5), y1: y1.toFixed(5), w: String(width),
  })}`,
  hdSnapshotUrl: (channel) => `${API_BASE}/api/hd/${channel}/snapshot`,
};
