import { useStore } from '../stores/store';
import { api } from '../api/client';
import ClassSelector from '../components/settings/ClassSelector';
import FPSControl from '../components/settings/FPSControl';
import { Eye, Crosshair, Zap, Gauge, Box, Boxes, Cpu, Fingerprint, Minus, Square, Trash2 } from 'lucide-react';

function InfoRow({ icon: Icon, label, value, children }) {
  return (
    <div className="flex items-start gap-3 p-4">
      <Icon className="w-4 h-4 text-gray-500 mt-0.5 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <span className="text-xs font-mono text-gray-500 uppercase tracking-wider">{label}</span>
          <span className="text-sm text-white text-right break-words">{value}</span>
        </div>
        {children}
      </div>
    </div>
  );
}

// Benchmark timings measured on this PC, chosen option highlighted
function BenchmarkBars({ rows, unit }) {
  if (!rows?.length) return null;
  const max = Math.max(...rows.map(r => r.ms));
  return (
    <div className="mt-3 space-y-1.5">
      {rows.map(r => (
        <div key={r.name} className="grid grid-cols-[minmax(0,1fr)_auto] sm:grid-cols-[minmax(0,20rem)_minmax(0,1fr)_auto] items-center gap-x-3 text-xs font-mono">
          <span className={`truncate ${r.chosen ? 'text-neon-cyan' : 'text-gray-500'}`} title={r.name}>{r.name}</span>
          <div className="hidden sm:block h-1.5 rounded-full bg-white/5 overflow-hidden">
            <div className={`h-full rounded-full ${r.chosen ? 'bg-neon-cyan' : 'bg-gray-600'}`} style={{ width: `${(r.ms / max) * 100}%` }} />
          </div>
          <span className={`text-right ${r.chosen ? 'text-neon-cyan' : 'text-gray-500'}`}>{r.ms} ms{r.chosen ? ' ✓' : ''}</span>
        </div>
      ))}
      <p className="text-[11px] text-gray-600 pt-1">{unit}</p>
    </div>
  );
}

function ToggleCard({ icon: Icon, title, description, active, onClick, color = 'green' }) {
  const colors = {
    green: active ? 'border-neon-green/40 glow-green' : 'border-border',
    cyan: active ? 'border-neon-cyan/40 glow-cyan' : 'border-border',
    magenta: active ? 'border-neon-magenta/40 glow-magenta' : 'border-border',
    amber: active ? 'border-neon-amber/40 glow-amber' : 'border-border',
  };
  const dots = {
    green: active ? 'bg-neon-green' : 'bg-gray-600',
    cyan: active ? 'bg-neon-cyan' : 'bg-gray-600',
    magenta: active ? 'bg-neon-magenta' : 'bg-gray-600',
    amber: active ? 'bg-neon-amber' : 'bg-gray-600',
  };

  return (
    <button
      onClick={onClick}
      className={`flex items-start gap-3 p-4 bg-surface rounded-xl border transition-all duration-200 text-left hover:bg-surface-light ${colors[color]}`}
    >
      <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${dots[color]}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        <p className="text-xs text-gray-500 mt-1">{description}</p>
      </div>
      <span className={`text-xs font-mono px-2 py-0.5 rounded ${active ? 'text-neon-green bg-neon-green/10' : 'text-gray-600 bg-white/5'}`}>
        {active ? 'ON' : 'OFF'}
      </span>
    </button>
  );
}

const zoneIcons = { privacy: Square, counting: Box, crossing_line: Minus };

export default function Settings() {
  const config = useStore(s => s.config);
  const setConfig = useStore(s => s.setConfig);
  const zones = useStore(s => s.zones);
  const removeZone = useStore(s => s.removeZone);

  const backend = config.inference_backend;
  const tracker = config.tracker;
  const model = config.model ? config.model.replace(/\.pt$/, '').replace(/^yolo/i, 'YOLO') : 'YOLO';
  const trackerName = tracker ? `${tracker.name}${tracker.reid ? ' with re-identification' : ''}` : 'Multi-object tracking';

  const toggle = async (fn) => {
    try {
      const data = await fn();
      setConfig(data);
    } catch {}
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-xl font-semibold text-white">Settings</h1>

      {/* Detection toggles */}
      <section className="space-y-3">
        <h2 className="text-sm font-mono text-gray-500 uppercase tracking-wider">Detection</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <ToggleCard
            icon={Eye}
            title="Object Detection"
            description={`Run ${model} on all camera feeds`}
            active={config.detection_enabled}
            onClick={() => toggle(api.toggleDetection)}
            color="green"
          />
          <ToggleCard
            icon={Crosshair}
            title="Object Tracking"
            description={trackerName}
            active={config.tracking_enabled}
            onClick={() => toggle(api.toggleTracking)}
            color="magenta"
          />
          <ToggleCard
            icon={Zap}
            title="Batched Mode"
            description="Process all cameras in one pass instead of one at a time"
            active={config.batched_mode}
            onClick={() => toggle(api.toggleBatchedMode)}
            color="amber"
          />
          <ToggleCard
            icon={Gauge}
            title="Auto FPS"
            description="Automatically detect stream FPS"
            active={config.auto_fps}
            onClick={() => toggle(api.toggleAutoFps)}
            color="cyan"
          />
        </div>
      </section>

      {/* Inference: what this PC runs and why */}
      {backend && (
        <section className="space-y-3">
          <h2 className="text-sm font-mono text-gray-500 uppercase tracking-wider">Inference on this PC</h2>
          <div className="bg-surface rounded-xl border border-border divide-y divide-border">
            <InfoRow icon={Boxes} label="Model" value={config.model} />
            <InfoRow icon={Cpu} label="Backend" value={backend.label}>
              {backend.benchmark?.length ? (
                <BenchmarkBars rows={backend.benchmark} unit="Time to detect objects in 4 frames, measured on this PC at first start." />
              ) : (
                <p className="text-xs text-gray-500 mt-1">{backend.reason}</p>
              )}
            </InfoRow>
            {tracker && (
              <InfoRow icon={Crosshair} label="Tracker" value={`${tracker.name}${tracker.reid ? ' + Re-ID' : ''}`} />
            )}
            {tracker?.reid && (
              <InfoRow icon={Fingerprint} label="Re-ID" value={tracker.reid_label}>
                <BenchmarkBars rows={backend.reid_benchmark} unit="Time to compare the appearance of 1 + 4 tracked objects." />
              </InfoRow>
            )}
          </div>
          <p className="text-xs text-gray-600">
            Each PC picks its own backend the first time it starts (YOLO_DEVICE=auto in .env). To measure again, set
            YOLO_REBENCHMARK=1 and restart the backend.
          </p>
        </section>
      )}

      {/* FPS */}
      <section className="bg-surface rounded-xl border border-border p-4">
        <FPSControl />
      </section>

      {/* Classes */}
      <section className="bg-surface rounded-xl border border-border p-4">
        <ClassSelector />
      </section>

      {/* Zone management */}
      <section className="space-y-3">
        <h2 className="text-sm font-mono text-gray-500 uppercase tracking-wider">
          Zones & Lines ({zones.length})
        </h2>
        {zones.length === 0 ? (
          <div className="bg-surface rounded-xl border border-border p-6 text-center">
            <p className="text-gray-600 text-sm">No zones defined. Use the drawing tools on the Dashboard to create zones.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {zones.map(zone => {
              const ZoneIcon = zoneIcons[zone.type] || Box;
              const typeColors = {
                privacy: 'text-neon-red border-neon-red/20',
                counting: 'text-neon-green border-neon-green/20',
                crossing_line: 'text-neon-cyan border-neon-cyan/20',
              };
              return (
                <div key={zone.id} className={`flex items-center gap-3 bg-surface rounded-lg border p-3 ${typeColors[zone.type] || 'border-border'}`}>
                  <ZoneIcon className="w-4 h-4 flex-shrink-0" />
                  <span className="text-sm text-white flex-1">{zone.label}</span>
                  <span className="text-xs font-mono text-gray-500">Camera {zone.camera + 1}</span>
                  <span className="text-xs font-mono text-gray-500 capitalize">{zone.type.replace('_', ' ')}</span>
                  <button
                    onClick={() => {
                      removeZone(zone.id);
                      api.deleteZone(zone.id).catch(() => {});
                    }}
                    className="p-1 rounded hover:bg-neon-red/10 text-gray-500 hover:text-neon-red transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
