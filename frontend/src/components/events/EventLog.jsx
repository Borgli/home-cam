import { useStore } from '../../stores/store';
import { ChevronDown, ChevronUp, Activity } from 'lucide-react';

function formatTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

const classColors = {
  person: 'text-neon-green',
  car: 'text-neon-cyan',
  truck: 'text-neon-amber',
  bicycle: 'text-neon-magenta',
  dog: 'text-yellow-400',
  cat: 'text-orange-400',
};

export default function EventLog() {
  const events = useStore(s => s.events);
  const eventLogOpen = useStore(s => s.eventLogOpen);
  const setEventLogOpen = useStore(s => s.setEventLogOpen);

  return (
    <div className={`bg-surface border-t border-border transition-all duration-300 ${eventLogOpen ? 'h-48' : 'h-9'}`}>
      {/* Toggle bar */}
      <button
        onClick={() => setEventLogOpen(!eventLogOpen)}
        className="w-full flex items-center justify-between px-4 py-2 text-xs font-mono text-gray-400 hover:text-white transition-colors"
      >
        <span className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-neon-green" />
          Event Log
          <span className="text-gray-600">({events.length})</span>
        </span>
        {eventLogOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
      </button>

      {/* Event list */}
      {eventLogOpen && (
        <div className="overflow-y-auto h-[calc(100%-36px)] px-4 pb-2 space-y-0.5">
          {events.length === 0 ? (
            <p className="text-gray-600 text-xs font-mono py-4 text-center">
              No events yet. Enable detection to start logging.
            </p>
          ) : (
            events.map((event, i) => (
              <div key={`${event.timestamp}-${i}`} className="flex items-center gap-3 text-xs font-mono py-0.5 hover:bg-white/5 rounded px-1">
                <span className="text-gray-600 w-16 flex-shrink-0">{formatTime(event.timestamp)}</span>
                <span className="text-gray-500 w-8 flex-shrink-0">Ch{event.channel}</span>
                <span className={`font-medium w-20 flex-shrink-0 ${classColors[event.class] || 'text-gray-300'}`}>
                  {event.class}
                </span>
                <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-neon-green"
                    style={{ width: `${(event.confidence * 100).toFixed(0)}%`, opacity: event.confidence }}
                  />
                </div>
                <span className="text-gray-500 w-10 text-right flex-shrink-0">
                  {(event.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
