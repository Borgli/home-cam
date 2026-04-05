import { useState } from 'react';
import { Search, CheckSquare, Square } from 'lucide-react';
import { useStore, COCO_CLASSES } from '../../stores/store';

export default function ClassSelector() {
  const enabledClasses = useStore(s => s.enabledClasses);
  const toggleClass = useStore(s => s.toggleClass);
  const setEnabledClasses = useStore(s => s.setEnabledClasses);
  const [search, setSearch] = useState('');

  const filtered = COCO_CLASSES
    .map((name, id) => ({ name, id }))
    .filter(c => c.name.toLowerCase().includes(search.toLowerCase()));

  const allEnabled = enabledClasses.size === COCO_CLASSES.length;
  const noneEnabled = enabledClasses.size === 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">Detection Classes</h3>
        <span className="text-xs font-mono text-gray-500">{enabledClasses.size}/{COCO_CLASSES.length}</span>
      </div>

      {/* Search + bulk actions */}
      <div className="flex items-center gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
          <input
            type="text"
            placeholder="Search classes..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 bg-surface-dark border border-border rounded-lg text-xs text-white placeholder-gray-600 focus:outline-none focus:border-neon-green/50"
          />
        </div>
        <button
          onClick={() => setEnabledClasses(allEnabled ? [] : Array.from({ length: COCO_CLASSES.length }, (_, i) => i))}
          className="text-xs font-mono text-neon-cyan hover:text-white transition-colors px-2 py-1"
        >
          {allEnabled ? 'None' : 'All'}
        </button>
      </div>

      {/* Class grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1 max-h-80 overflow-y-auto pr-1">
        {filtered.map(({ name, id }) => {
          const enabled = enabledClasses.has(id);
          return (
            <button
              key={id}
              onClick={() => toggleClass(id)}
              className={`flex items-center gap-1.5 px-2 py-1.5 rounded text-xs transition-all ${
                enabled
                  ? 'bg-neon-green/10 text-neon-green border border-neon-green/20'
                  : 'bg-white/5 text-gray-500 border border-transparent hover:border-border'
              }`}
            >
              {enabled
                ? <CheckSquare className="w-3 h-3 flex-shrink-0" />
                : <Square className="w-3 h-3 flex-shrink-0" />
              }
              <span className="truncate">{name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
