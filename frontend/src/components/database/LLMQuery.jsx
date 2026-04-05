import { useState } from 'react';
import { Sparkles, AlertCircle, Code, Loader2 } from 'lucide-react';
import { api } from '../../api/client';
import TableViewer from './TableViewer';

export default function LLMQuery() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const examples = [
    'How many people were detected today?',
    'Show me the top 5 most common detected objects',
    'What time was the last car detected on camera 1?',
    'Count detections per camera in the last hour',
    'Show all unique classes detected with their average confidence',
  ];

  const ask = async (q) => {
    const query = q || question;
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const data = await api.llmQuery(query);
      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Input */}
      <div className="relative">
        <Sparkles className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neon-magenta" />
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="Ask a question about your surveillance data..."
          className="w-full pl-10 pr-24 py-3 bg-surface-dark border border-border rounded-lg text-sm text-white placeholder-gray-600 focus:outline-none focus:border-neon-magenta/50"
          onKeyDown={e => {
            if (e.key === 'Enter') {
              e.preventDefault();
              ask();
            }
          }}
        />
        <button
          onClick={() => ask()}
          disabled={loading || !question.trim()}
          className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5 px-3 py-1.5 bg-neon-magenta/10 text-neon-magenta rounded-md text-xs font-medium hover:bg-neon-magenta/20 transition-colors disabled:opacity-30"
        >
          {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3" />}
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </div>

      {/* Example queries */}
      <div className="flex flex-wrap gap-1.5">
        {examples.map(ex => (
          <button
            key={ex}
            onClick={() => { setQuestion(ex); ask(ex); }}
            className="px-2.5 py-1 bg-white/5 border border-border rounded-full text-xs text-gray-500 hover:text-neon-magenta hover:border-neon-magenta/30 transition-colors"
          >
            {ex}
          </button>
        ))}
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 bg-neon-red/10 border border-neon-red/20 rounded-lg">
          <AlertCircle className="w-4 h-4 text-neon-red flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-mono text-neon-red">{error}</p>
            {error.includes('Ollama') && (
              <div className="mt-2 text-xs text-gray-500 space-y-1">
                <p>To use natural language queries, install Ollama:</p>
                <ol className="list-decimal ml-4 space-y-0.5">
                  <li>Install Ollama from <span className="text-neon-cyan">ollama.com</span></li>
                  <li>Run: <code className="text-neon-green">ollama pull gemma4:e2b</code></li>
                  <li>Ollama runs at localhost:11434 automatically</li>
                </ol>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-3">
          {/* Generated SQL */}
          {result.sql && (
            <div className="bg-surface-dark border border-border rounded-lg p-3">
              <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-2">
                <Code className="w-3 h-3" /> Generated SQL
              </div>
              <pre className="text-xs font-mono text-neon-green whitespace-pre-wrap">{result.sql}</pre>
            </div>
          )}

          {/* Data */}
          {result.rows && <TableViewer data={result.rows} title="LLM Query Results" />}
        </div>
      )}
    </div>
  );
}
