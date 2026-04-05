import { useState } from 'react';
import { Play, AlertCircle } from 'lucide-react';
import { api } from '../../api/client';
import TableViewer from './TableViewer';

export default function QueryEditor() {
  const [sql, setSql] = useState('SELECT * FROM detections ORDER BY timestamp DESC LIMIT 100');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const execute = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.queryDb(sql);
      if (data.error) {
        setError(data.error);
        setResults(null);
      } else {
        setResults(data.rows || []);
      }
    } catch (e) {
      setError(e.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="relative">
        <textarea
          value={sql}
          onChange={e => setSql(e.target.value)}
          rows={4}
          placeholder="SELECT * FROM detections WHERE class_name = 'person' LIMIT 100"
          className="w-full bg-surface-dark border border-border rounded-lg p-3 text-sm font-mono text-neon-green placeholder-gray-700 focus:outline-none focus:border-neon-green/50 resize-y"
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              execute();
            }
          }}
        />
        <button
          onClick={execute}
          disabled={loading || !sql.trim()}
          className="absolute top-2 right-2 flex items-center gap-1 px-3 py-1.5 bg-neon-green/10 text-neon-green rounded-md text-xs font-medium hover:bg-neon-green/20 transition-colors disabled:opacity-30"
        >
          <Play className="w-3 h-3" />
          {loading ? 'Running...' : 'Run (Ctrl+Enter)'}
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 bg-neon-red/10 border border-neon-red/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-neon-red flex-shrink-0 mt-0.5" />
          <p className="text-xs font-mono text-neon-red">{error}</p>
        </div>
      )}

      {results && <TableViewer data={results} title="Query Results" />}
    </div>
  );
}
