import { useStore } from '../stores/store';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { Activity, Clock, Zap, Cpu } from 'lucide-react';

function StatCard({ icon: Icon, label, value, color = '#00ff88' }) {
  return (
    <div className="bg-surface rounded-xl border border-border p-4">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4" style={{ color }} />
        <span className="text-xs text-gray-500 font-mono uppercase">{label}</span>
      </div>
      <div className="text-2xl font-bold font-mono" style={{ color }}>{value}</div>
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div className="bg-surface rounded-xl border border-border p-4">
      <h3 className="text-sm font-medium text-white mb-3">{title}</h3>
      <div className="h-48">
        {children}
      </div>
    </div>
  );
}

const neonTheme = {
  bg: '#12121a',
  grid: '#1e1e2e',
  text: '#666',
  green: '#00ff88',
  cyan: '#00ccff',
  magenta: '#ff00ff',
  amber: '#ffaa00',
};

export default function Analytics() {
  const metrics = useStore(s => s.metrics);

  if (!metrics) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-2">
          <Activity className="w-12 h-12 text-gray-600 mx-auto" />
          <p className="text-gray-500 text-sm">Waiting for metrics data...</p>
          <p className="text-gray-600 text-xs">Enable detection and wait a few seconds.</p>
        </div>
      </div>
    );
  }

  // Build FPS chart data
  const fpsData = [];
  const maxLen = Math.max(
    ...Object.values(metrics.fps || {}).map(arr => arr?.length || 0), 0
  );
  for (let i = 0; i < Math.min(maxLen, 100); i++) {
    const point = { idx: i };
    for (let ch = 0; ch < 4; ch++) {
      const arr = metrics.fps?.[ch] || [];
      point[`ch${ch}`] = arr[i]?.fps ?? null;
    }
    fpsData.push(point);
  }

  // Batch time breakdown
  const batchBreakdown = [
    { name: 'Preprocess', time: metrics.avg_preprocessing_time || 0, fill: neonTheme.amber },
    { name: 'Inference', time: metrics.avg_inference_time || 0, fill: neonTheme.cyan },
    { name: 'Postprocess', time: metrics.avg_postprocessing_time || 0, fill: neonTheme.magenta },
  ];

  // Efficiency over time
  const efficiencyData = (metrics.batch_efficiency || []).map((d, i) => ({
    idx: i, efficiency: d.efficiency,
  }));

  // Memory over time
  const memoryData = (metrics.memory_usage || []).map((d, i) => ({
    idx: i, memory: d.memory,
  }));

  // Confidence distribution
  const confidenceBins = {};
  (metrics.confidence || []).forEach(c => {
    const bin = Math.floor(c * 10) / 10;
    confidenceBins[bin.toFixed(1)] = (confidenceBins[bin.toFixed(1)] || 0) + 1;
  });
  const confidenceData = Object.entries(confidenceBins).map(([range, count]) => ({ range, count }));

  const chColors = [neonTheme.green, neonTheme.cyan, neonTheme.magenta, neonTheme.amber];

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      <h1 className="text-xl font-semibold text-white">Analytics</h1>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard icon={Activity} label="Total Detections" value={metrics.total_detections} color={neonTheme.green} />
        <StatCard icon={Clock} label="Batch Time" value={`${(metrics.avg_batch_time || 0).toFixed(1)}ms`} color={neonTheme.amber} />
        <StatCard icon={Zap} label="Efficiency" value={`${(metrics.avg_efficiency || 0).toFixed(2)}x`} color={neonTheme.cyan} />
        <StatCard icon={Cpu} label="Uptime" value={metrics.uptime || '0s'} color={neonTheme.magenta} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* FPS per camera */}
        <ChartCard title="FPS Per Camera">
          <ResponsiveContainer>
            <LineChart data={fpsData}>
              <CartesianGrid stroke={neonTheme.grid} />
              <XAxis dataKey="idx" stroke={neonTheme.text} tick={{ fontSize: 10 }} />
              <YAxis stroke={neonTheme.text} tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ background: neonTheme.bg, border: `1px solid ${neonTheme.grid}`, borderRadius: 8 }} />
              {[0, 1, 2, 3].map(ch => (
                <Line key={ch} type="monotone" dataKey={`ch${ch}`} stroke={chColors[ch]} dot={false} strokeWidth={1.5} name={`Camera ${ch + 1}`} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Batch breakdown */}
        <ChartCard title="Batch Time Breakdown">
          <ResponsiveContainer>
            <BarChart data={batchBreakdown}>
              <CartesianGrid stroke={neonTheme.grid} />
              <XAxis dataKey="name" stroke={neonTheme.text} tick={{ fontSize: 10 }} />
              <YAxis stroke={neonTheme.text} tick={{ fontSize: 10 }} unit="ms" />
              <Tooltip contentStyle={{ background: neonTheme.bg, border: `1px solid ${neonTheme.grid}`, borderRadius: 8 }} />
              <Bar dataKey="time" radius={[4, 4, 0, 0]}>
                {batchBreakdown.map((entry, i) => (
                  <rect key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Efficiency over time */}
        <ChartCard title="Batch Efficiency Over Time">
          <ResponsiveContainer>
            <AreaChart data={efficiencyData}>
              <CartesianGrid stroke={neonTheme.grid} />
              <XAxis dataKey="idx" stroke={neonTheme.text} tick={{ fontSize: 10 }} />
              <YAxis stroke={neonTheme.text} tick={{ fontSize: 10 }} unit="x" />
              <Tooltip contentStyle={{ background: neonTheme.bg, border: `1px solid ${neonTheme.grid}`, borderRadius: 8 }} />
              <Area type="monotone" dataKey="efficiency" stroke={neonTheme.amber} fill={`${neonTheme.amber}20`} strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* GPU Memory */}
        <ChartCard title="GPU Memory Usage">
          {memoryData.length > 0 ? (
            <ResponsiveContainer>
              <LineChart data={memoryData}>
                <CartesianGrid stroke={neonTheme.grid} />
                <XAxis dataKey="idx" stroke={neonTheme.text} tick={{ fontSize: 10 }} />
                <YAxis stroke={neonTheme.text} tick={{ fontSize: 10 }} unit="MB" />
                <Tooltip contentStyle={{ background: neonTheme.bg, border: `1px solid ${neonTheme.grid}`, borderRadius: 8 }} />
                <Line type="monotone" dataKey="memory" stroke={neonTheme.magenta} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-600 text-xs font-mono">
              No GPU data available (CPU mode)
            </div>
          )}
        </ChartCard>

        {/* Confidence distribution */}
        <ChartCard title="Detection Confidence Distribution">
          <ResponsiveContainer>
            <BarChart data={confidenceData}>
              <CartesianGrid stroke={neonTheme.grid} />
              <XAxis dataKey="range" stroke={neonTheme.text} tick={{ fontSize: 10 }} />
              <YAxis stroke={neonTheme.text} tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ background: neonTheme.bg, border: `1px solid ${neonTheme.grid}`, borderRadius: 8 }} />
              <Bar dataKey="count" fill={neonTheme.green} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}
