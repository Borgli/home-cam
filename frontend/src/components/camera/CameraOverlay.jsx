import { useState, useRef, useEffect, useCallback } from 'react';
import { Stage, Layer, Rect, Line, Circle, Text, Group, Transformer } from 'react-konva';
import { useStore } from '../../stores/store';
import { api } from '../../api/client';

const ZONE_COLORS = {
  privacy: { fill: 'rgba(0,0,0,0.7)', stroke: '#ff3355' },
  counting: { fill: 'rgba(0,255,136,0.1)', stroke: '#00ff88' },
  crossing_line: { fill: 'transparent', stroke: '#00ccff' },
};

export default function CameraOverlay({ channel, containerRef }) {
  const activeTool = useStore(s => s.activeTool);
  const zones = useStore(s => s.zones);
  const addZone = useStore(s => s.addZone);
  const updateZone = useStore(s => s.updateZone);
  const removeZone = useStore(s => s.removeZone);

  const [stageSize, setStageSize] = useState({ width: 640, height: 360 });
  const [drawing, setDrawing] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const stageRef = useRef(null);
  const trRef = useRef(null);

  // Resize observer
  useEffect(() => {
    const el = containerRef?.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setStageSize({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [containerRef]);

  // Filter zones for this camera
  const cameraZones = zones.filter(z => z.camera === channel);

  const handleMouseDown = useCallback((e) => {
    if (activeTool === 'pointer') {
      // Deselect on stage click
      if (e.target === e.target.getStage()) setSelectedId(null);
      return;
    }

    const pos = e.target.getStage().getPointerPosition();
    const id = `zone_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    if (activeTool === 'crossing_line') {
      setDrawing({ id, type: 'crossing_line', camera: channel, x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
    } else {
      setDrawing({ id, type: activeTool, camera: channel, x: pos.x, y: pos.y, width: 0, height: 0 });
    }
  }, [activeTool, channel]);

  const handleMouseMove = useCallback((e) => {
    if (!drawing) return;
    const pos = e.target.getStage().getPointerPosition();

    if (drawing.type === 'crossing_line') {
      setDrawing(d => ({ ...d, x2: pos.x, y2: pos.y }));
    } else {
      setDrawing(d => ({ ...d, width: pos.x - d.x, height: pos.y - d.y }));
    }
  }, [drawing]);

  const handleMouseUp = useCallback(() => {
    if (!drawing) return;

    // Normalize coords and store as percentages
    const w = stageSize.width;
    const h = stageSize.height;

    let coords;
    if (drawing.type === 'crossing_line') {
      const dx = Math.abs(drawing.x2 - drawing.x1);
      const dy = Math.abs(drawing.y2 - drawing.y1);
      if (dx < 5 && dy < 5) { setDrawing(null); return; } // Too small
      coords = [
        { x: drawing.x1 / w, y: drawing.y1 / h },
        { x: drawing.x2 / w, y: drawing.y2 / h },
      ];
    } else {
      if (Math.abs(drawing.width) < 10 || Math.abs(drawing.height) < 10) { setDrawing(null); return; }
      const x = Math.min(drawing.x, drawing.x + drawing.width);
      const y = Math.min(drawing.y, drawing.y + drawing.height);
      const width = Math.abs(drawing.width);
      const height = Math.abs(drawing.height);
      coords = [
        { x: x / w, y: y / h },
        { x: (x + width) / w, y: (y + height) / h },
      ];
    }

    const zone = {
      id: drawing.id,
      camera: channel,
      type: drawing.type,
      label: `${drawing.type} ${cameraZones.length + 1}`,
      coords,
      color: ZONE_COLORS[drawing.type].stroke,
      enabled: true,
    };

    addZone(zone);
    setDrawing(null);

    // Try to persist to backend
    api.createZone(zone).catch(() => {});
  }, [drawing, stageSize, channel, cameraZones.length, addZone]);

  // Select transformer
  useEffect(() => {
    if (selectedId && trRef.current) {
      const stage = stageRef.current;
      const node = stage?.findOne(`#${selectedId}`);
      if (node) {
        trRef.current.nodes([node]);
        trRef.current.getLayer().batchDraw();
      }
    }
  }, [selectedId]);

  const isDrawMode = activeTool !== 'pointer';

  return (
    <div className="absolute inset-0" style={{ pointerEvents: isDrawMode ? 'all' : 'none' }}>
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{ cursor: isDrawMode ? 'crosshair' : 'default' }}
      >
        <Layer>
          {/* Existing zones */}
          {cameraZones.map(zone => {
            const w = stageSize.width;
            const h = stageSize.height;
            const colors = ZONE_COLORS[zone.type] || ZONE_COLORS.counting;

            if (zone.type === 'crossing_line') {
              const [p1, p2] = zone.coords;
              return (
                <Group key={zone.id} id={zone.id}>
                  <Line
                    points={[p1.x * w, p1.y * h, p2.x * w, p2.y * h]}
                    stroke={colors.stroke}
                    strokeWidth={3}
                    lineCap="round"
                    dash={[10, 5]}
                    onClick={() => { if (activeTool === 'pointer') setSelectedId(zone.id); }}
                    onDblClick={() => {
                      if (confirm('Delete this crossing line?')) {
                        removeZone(zone.id);
                        api.deleteZone(zone.id).catch(() => {});
                      }
                    }}
                    style={{ pointerEvents: 'all' }}
                  />
                  <Circle x={p1.x * w} y={p1.y * h} radius={6} fill={colors.stroke} opacity={0.8} />
                  <Circle x={p2.x * w} y={p2.y * h} radius={6} fill={colors.stroke} opacity={0.8} />
                  <Text
                    x={(p1.x * w + p2.x * w) / 2 - 30}
                    y={(p1.y * h + p2.y * h) / 2 - 20}
                    text={zone.label || 'Line'}
                    fill={colors.stroke}
                    fontSize={11}
                    fontFamily="JetBrains Mono, monospace"
                  />
                </Group>
              );
            }

            const [tl, br] = zone.coords;
            const rx = tl.x * w;
            const ry = tl.y * h;
            const rw = (br.x - tl.x) * w;
            const rh = (br.y - tl.y) * h;

            return (
              <Group key={zone.id} id={zone.id}>
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth={2}
                  dash={zone.type === 'counting' ? [8, 4] : undefined}
                  cornerRadius={zone.type === 'privacy' ? 0 : 4}
                  onClick={() => { if (activeTool === 'pointer') setSelectedId(zone.id); }}
                  onDblClick={() => {
                    if (confirm(`Delete this ${zone.type} zone?`)) {
                      removeZone(zone.id);
                      api.deleteZone(zone.id).catch(() => {});
                    }
                  }}
                  draggable={activeTool === 'pointer'}
                  style={{ pointerEvents: 'all' }}
                />
                <Text
                  x={rx + 4}
                  y={ry + 4}
                  text={zone.label || zone.type}
                  fill={colors.stroke}
                  fontSize={11}
                  fontFamily="JetBrains Mono, monospace"
                  listening={false}
                />
              </Group>
            );
          })}

          {/* Currently drawing shape */}
          {drawing && drawing.type === 'crossing_line' && (
            <Line
              points={[drawing.x1, drawing.y1, drawing.x2, drawing.y2]}
              stroke={ZONE_COLORS.crossing_line.stroke}
              strokeWidth={3}
              lineCap="round"
              dash={[10, 5]}
              opacity={0.7}
            />
          )}
          {drawing && drawing.type !== 'crossing_line' && (
            <Rect
              x={Math.min(drawing.x, drawing.x + drawing.width)}
              y={Math.min(drawing.y, drawing.y + drawing.height)}
              width={Math.abs(drawing.width)}
              height={Math.abs(drawing.height)}
              fill={ZONE_COLORS[drawing.type]?.fill}
              stroke={ZONE_COLORS[drawing.type]?.stroke}
              strokeWidth={2}
              dash={[8, 4]}
              opacity={0.7}
            />
          )}

          {/* Transformer for selection */}
          {selectedId && (
            <Transformer
              ref={trRef}
              boundBoxFunc={(oldBox, newBox) => newBox}
              enabledAnchors={['top-left', 'top-right', 'bottom-left', 'bottom-right']}
              borderStroke="#00ff88"
              anchorFill="#00ff88"
              anchorSize={8}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
