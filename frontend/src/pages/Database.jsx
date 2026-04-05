import { useState, useEffect } from 'react';
import { Database as DbIcon, Search, Sparkles, Table } from 'lucide-react';
import { api } from '../api/client';
import QueryEditor from '../components/database/QueryEditor';
import LLMQuery from '../components/database/LLMQuery';
import TableViewer from '../components/database/TableViewer';

const tabs = [
  { id: 'browse', icon: Table, label: 'Browse' },
  { id: 'query', icon: Search, label: 'SQL Query' },
  { id: 'llm', icon: Sparkles, label: 'AI Query' },
];

export default function Database() {
  const [activeTab, setActiveTab] = useState('browse');
  const [recentDetections, setRecentDetections] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [events, eventStats] = await Promise.all([
        api.getEvents({ page: 1, per_page: 200 }).catch(() => ({ events: [] })),
        api.getEventStats().catch(() => null),
      ]);
      setRecentDetections(events.events || []);
      setStats(eventStats);
    } catch {
      // Backend may not have these endpoints yet
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-white flex items-center gap-2">
          <DbIcon className="w-5 h-5 text-neon-cyan" />
          Database
        </h1>

        {/* Stats */}
        {stats && (
          <div className="flex items-center gap-4 text-xs font-mono text-gray-500">
            <span>Total: <span className="text-neon-green">{stats.total_detections || 0}</span></span>
            <span>Classes: <span className="text-neon-cyan">{stats.unique_classes || 0}</span></span>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 bg-surface rounded-lg border border-border p-1">
        {tabs.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === id
                ? 'bg-white/10 text-white'
                : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="bg-surface rounded-xl border border-border p-4">
        {activeTab === 'browse' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-white">Recent Detections</h2>
              <button
                onClick={loadData}
                className="text-xs text-neon-cyan hover:text-white transition-colors"
              >
                Refresh
              </button>
            </div>
            {recentDetections.length > 0 ? (
              <TableViewer data={recentDetections} title="Detections" />
            ) : (
              <div className="text-center py-8">
                <DbIcon className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-gray-600 text-sm">No detections recorded yet.</p>
                <p className="text-gray-700 text-xs mt-1">
                  Enable detection to start populating the database.
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'query' && (
          <div className="space-y-3">
            <h2 className="text-sm font-medium text-white">SQL Query Editor</h2>
            <p className="text-xs text-gray-600">
              Write SQL queries to explore detection data. Only SELECT queries are allowed.
            </p>
            <QueryEditor />
          </div>
        )}

        {activeTab === 'llm' && (
          <div className="space-y-3">
            <h2 className="text-sm font-medium text-white">AI-Powered Query</h2>
            <p className="text-xs text-gray-600">
              Ask questions in natural language. Powered by Gemma 4 E2B via Ollama (local, private).
            </p>
            <LLMQuery />
          </div>
        )}
      </div>
    </div>
  );
}
