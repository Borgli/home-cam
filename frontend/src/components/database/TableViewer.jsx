import { useState } from 'react';
import { ChevronLeft, ChevronRight, Download } from 'lucide-react';

export default function TableViewer({ data, columns, title = 'Results' }) {
  const [page, setPage] = useState(0);
  const perPage = 50;

  if (!data || data.length === 0) {
    return (
      <div className="bg-surface rounded-xl border border-border p-6 text-center text-gray-600 text-sm font-mono">
        No data to display.
      </div>
    );
  }

  const cols = columns || Object.keys(data[0]);
  const totalPages = Math.ceil(data.length / perPage);
  const pageData = data.slice(page * perPage, (page + 1) * perPage);

  const exportCsv = () => {
    const header = cols.join(',');
    const rows = data.map(row => cols.map(c => JSON.stringify(row[c] ?? '')).join(','));
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-gray-500">{data.length} rows</span>
        <div className="flex items-center gap-2">
          <button
            onClick={exportCsv}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs text-neon-cyan hover:bg-neon-cyan/10 transition-colors"
          >
            <Download className="w-3 h-3" /> CSV
          </button>
          {totalPages > 1 && (
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="p-1 rounded hover:bg-white/10 text-gray-500 disabled:opacity-30"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-xs font-mono text-gray-500">
                {page + 1}/{totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="p-1 rounded hover:bg-white/10 text-gray-500 disabled:opacity-30"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto max-h-96 rounded-lg border border-border">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-surface-dark">
            <tr>
              {cols.map(col => (
                <th key={col} className="px-3 py-2 text-left text-gray-500 font-medium border-b border-border">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.map((row, i) => (
              <tr key={i} className="hover:bg-white/5 transition-colors">
                {cols.map(col => (
                  <td key={col} className="px-3 py-1.5 text-gray-300 border-b border-border/50 max-w-xs truncate">
                    {typeof row[col] === 'number' ? (
                      Number.isInteger(row[col]) ? row[col] : row[col].toFixed(3)
                    ) : (
                      String(row[col] ?? '')
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
