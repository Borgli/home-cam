import { useState, useRef, useCallback } from 'react';
import { Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import { useStore } from '../../stores/store';
import { api } from '../../api/client';
import CameraOverlay from './CameraOverlay';

export default function CameraCard({ channel }) {
  const expandedCamera = useStore(s => s.expandedCamera);
  const setExpandedCamera = useStore(s => s.setExpandedCamera);
  const activeTool = useStore(s => s.activeTool);
  const isExpanded = expandedCamera === channel;

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(z => Math.max(1, Math.min(5, z + delta)));
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (activeTool !== 'pointer' || zoom <= 1) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [activeTool, zoom, pan]);

  const handleMouseMove = useCallback((e) => {
    if (!isPanning) return;
    setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
  }, [isPanning, panStart]);

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  const resetZoom = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  return (
    <div
      className={`relative bg-surface rounded-xl border border-border overflow-hidden group transition-all duration-300
        ${isExpanded ? 'fixed inset-4 z-50' : ''}
        hover:border-neon-green/30 hover:glow-green`}
    >
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-3 py-2 bg-gradient-to-b from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="text-xs font-mono text-neon-cyan font-medium">
          Camera {channel + 1}
        </span>
        <div className="flex items-center gap-1">
          {zoom > 1 && (
            <>
              <span className="text-xs font-mono text-neon-amber mr-1">{zoom.toFixed(1)}x</span>
              <button onClick={resetZoom} className="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white">
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </>
          )}
          <button
            onClick={() => setZoom(z => Math.min(5, z + 0.5))}
            className="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => { setZoom(z => Math.max(1, z - 0.5)); if (zoom <= 1.5) setPan({ x: 0, y: 0 }); }}
            className="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setExpandedCamera(isExpanded ? null : channel)}
            className="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white"
          >
            {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Video + Overlay container */}
      <div
        ref={containerRef}
        className="relative w-full aspect-video overflow-hidden cursor-crosshair"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <img
          src={api.videoFeedUrl(channel)}
          alt={`Camera ${channel + 1}`}
          className="w-full h-full object-cover select-none"
          draggable={false}
          style={{
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
            transition: isPanning ? 'none' : 'transform 0.2s ease',
          }}
          onError={(e) => {
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'flex';
          }}
        />
        <div className="hidden items-center justify-center w-full h-full bg-surface text-gray-600 text-sm font-mono">
          Camera {channel + 1} - Offline
        </div>

        {/* Drawing overlay */}
        <CameraOverlay channel={channel} containerRef={containerRef} zoom={zoom} pan={pan} />
      </div>
    </div>
  );
}
