import CameraCard from '../components/camera/CameraCard';
import DrawingToolbar from '../components/drawing/DrawingToolbar';
import EventLog from '../components/events/EventLog';
import { useStore } from '../stores/store';

const CHANNELS = [0, 1, 2, 3];

export default function Dashboard() {
  const expandedCamera = useStore(s => s.expandedCamera);

  return (
    <div className="flex flex-col h-full -m-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <DrawingToolbar />
        <span className="text-xs font-mono text-gray-600">
          Double-click a zone to delete it
        </span>
      </div>

      {/* Camera grid */}
      <div className="flex-1 p-4 overflow-hidden">
        {expandedCamera !== null ? (
          <div className="h-full">
            <CameraCard channel={expandedCamera} />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 h-full">
            {CHANNELS.map(ch => (
              <CameraCard key={ch} channel={ch} />
            ))}
          </div>
        )}
      </div>

      {/* Event log */}
      <EventLog />
    </div>
  );
}
