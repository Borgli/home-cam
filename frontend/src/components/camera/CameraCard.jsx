import { useState, useRef, useCallback, useEffect } from 'react';
import { Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw, Camera, Loader2 } from 'lucide-react';
import { useStore } from '../../stores/store';
import { api } from '../../api/client';
import CameraOverlay from './CameraOverlay';

const HD_MIN_INTERVAL_MS = 80; // fetch the 4K view at most ~12 times a second
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Part of the frame (0-1 coordinates) that is visible at this zoom/pan: the inverse
// of the CSS transform `scale(zoom) translate(pan / zoom)` applied to the video.
function visibleRegion(zoom, pan, width, height) {
  return {
    x0: 0.5 - (0.5 + pan.x / width) / zoom,
    x1: 0.5 + (0.5 - pan.x / width) / zoom,
    y0: 0.5 - (0.5 + pan.y / height) / zoom,
    y1: 0.5 + (0.5 - pan.y / height) / zoom,
  };
}

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

  // 4K view: the backend crops the visible region out of the 4K frame, so zooming in shows real detail
  const hdOn = useStore(s => !!s.hdChannels[channel]);
  const setHdChannel = useStore(s => s.setHdChannel);
  const hdInfo = useStore(s => s.config.hd?.streams?.[channel]);
  const [hdBusy, setHdBusy] = useState(false);
  const [hdFrame, setHdFrame] = useState(null); // { url, zoom, pan } of the 4K image on screen
  const [hdMessage, setHdMessage] = useState(null);
  const viewRef = useRef({ zoom, pan });
  viewRef.current = { zoom, pan };
  const hdUrls = useRef(new Set());

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

  const toggleHd = async () => {
    if (hdBusy) return;
    if (hdOn) {
      setHdChannel(channel, false);
      api.hdStop(channel);
      return;
    }
    setHdBusy(true);
    const res = await api.hdStart(channel);
    setHdBusy(false);
    if (res.ok) setHdChannel(channel, true);
    else setHdMessage(res.data.error || `4K unavailable (HTTP ${res.status})`);
  };

  // While 4K is on, fetch the visible region of the newest 4K frame, one request at a time
  useEffect(() => {
    if (!hdOn) return undefined;
    let cancelled = false;
    (async () => {
      let restarted = false;
      while (!cancelled) {
        const el = containerRef.current;
        if (!el || !el.clientWidth || document.hidden) { await sleep(400); continue; }
        const { zoom: z, pan: p } = viewRef.current;
        const width = Math.min(2560, Math.round(el.clientWidth * (window.devicePixelRatio || 1)));
        const region = visibleRegion(z, p, el.clientWidth, el.clientHeight);
        const started = performance.now();
        try {
          const res = await fetch(api.hdFrameUrl(channel, { ...region, width }), { cache: 'no-store' });
          if (cancelled) break;
          if (res.ok) {
            const url = URL.createObjectURL(await res.blob());
            hdUrls.current.add(url);
            if (cancelled) break;
            setHdFrame({ url, zoom: z, pan: { ...p } });
            setHdMessage(null);
            restarted = false;
          } else if (res.status === 404 && !restarted) {
            // The backend stopped the stream (idle, or restarted): start it again once
            restarted = true;
            const start = await api.hdStart(channel);
            if (!start.ok && !cancelled) {
              setHdMessage(start.data.error || '4K unavailable');
              setHdChannel(channel, false);
            }
          } else if (res.status === 404) {
            setHdChannel(channel, false);
          } else {
            const data = await res.json().catch(() => ({}));
            setHdMessage(data.error || `4K view unavailable (HTTP ${res.status})`);
            await sleep(1000);
          }
        } catch {
          await sleep(1000); // backend unreachable; keep trying while 4K is on
        }
        const elapsed = performance.now() - started;
        if (elapsed < HD_MIN_INTERVAL_MS) await sleep(HD_MIN_INTERVAL_MS - elapsed);
      }
    })();
    return () => { cancelled = true; };
  }, [hdOn, channel, setHdChannel]);

  // Free image memory: keep only the image on screen, drop everything when 4K goes off or the card unmounts
  const revokeHdUrls = (keep) => {
    for (const url of hdUrls.current) {
      if (url !== keep) { URL.revokeObjectURL(url); hdUrls.current.delete(url); }
    }
  };
  useEffect(() => {
    if (!hdOn) { setHdFrame(null); revokeHdUrls(null); }
  }, [hdOn]);
  useEffect(() => () => revokeHdUrls(null), []);

  // Messages disappear after a while
  useEffect(() => {
    if (!hdMessage) return undefined;
    const t = setTimeout(() => setHdMessage(null), 6000);
    return () => clearTimeout(t);
  }, [hdMessage]);

  // The 4K image shows the view it was fetched for; map it onto the current view until the next one arrives
  const hdRatio = hdFrame ? zoom / hdFrame.zoom : 1;
  const hdTransform = hdFrame
    ? `translate(${pan.x - hdRatio * hdFrame.pan.x}px, ${pan.y - hdRatio * hdFrame.pan.y}px) scale(${hdRatio})`
    : undefined;

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
            onClick={toggleHd}
            title={hdOn ? 'Turn the 4K view off' : 'Show the 4K stream: zoom in for full detail (the NVR allows 2 cameras at once)'}
            className={`min-w-[2rem] h-5 px-1.5 flex items-center justify-center rounded border text-[10px] font-mono font-bold transition-colors
              ${hdOn
                ? 'text-neon-green border-neon-green/50 bg-neon-green/10'
                : 'text-gray-400 border-white/20 hover:text-white hover:border-white/40'}`}
          >
            {hdBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : '4K'}
          </button>
          <a
            href={api.hdSnapshotUrl(channel)}
            target="_blank"
            rel="noreferrer"
            title="Open a full-resolution still"
            className="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white"
          >
            <Camera className="w-3.5 h-3.5" />
          </a>
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
            transition: isPanning || hdOn ? 'none' : 'transform 0.2s ease',
          }}
          onError={(e) => {
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'flex';
          }}
        />
        <div className="hidden items-center justify-center w-full h-full bg-surface text-gray-600 text-sm font-mono">
          Camera {channel + 1} - Offline
        </div>

        {/* 4K view on top of the 720p stream, which still shows wherever the 4K image doesn't reach yet */}
        {hdOn && hdFrame && (
          <img
            src={hdFrame.url}
            alt=""
            draggable={false}
            onLoad={(e) => revokeHdUrls(e.currentTarget.src)}
            className="absolute inset-0 w-full h-full select-none pointer-events-none"
            style={{ objectFit: 'fill', transform: hdTransform }}
          />
        )}

        {/* Drawing overlay */}
        <CameraOverlay channel={channel} containerRef={containerRef} zoom={zoom} pan={pan} />

        {hdOn && (
          <div className="absolute bottom-2 left-2 z-20 px-1.5 py-0.5 rounded bg-black/60 text-[10px] font-mono text-neon-green pointer-events-none">
            4K{hdInfo?.fps ? ` · ${hdInfo.fps} fps` : ''}{hdInfo?.decoder ? ` · ${hdInfo.decoder}` : ''}
          </div>
        )}
        {hdMessage && (
          <div className="absolute bottom-2 right-2 z-30 max-w-[75%] px-2 py-1 rounded bg-black/80 border border-neon-amber/40 text-[11px] leading-snug text-neon-amber font-mono">
            {hdMessage}
          </div>
        )}
      </div>
    </div>
  );
}
