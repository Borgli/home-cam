import { create } from 'zustand';

export const useStore = create((set, get) => ({
  // Config state
  config: {
    detection_enabled: false,
    batched_mode: true,
    auto_fps: true,
    target_fps: 12,
    tracking_enabled: false,
  },
  setConfig: (config) => set({ config: { ...get().config, ...config } }),

  // Zones (privacy, counting, crossing lines)
  zones: [],
  setZones: (zones) => set({ zones }),
  addZone: (zone) => set({ zones: [...get().zones, zone] }),
  removeZone: (id) => set({ zones: get().zones.filter(z => z.id !== id) }),
  updateZone: (id, updates) => set({
    zones: get().zones.map(z => z.id === id ? { ...z, ...updates } : z)
  }),

  // Drawing state
  activeTool: 'pointer', // 'pointer' | 'privacy' | 'counting' | 'crossing_line'
  setActiveTool: (tool) => set({ activeTool: tool }),

  // Selected camera for full-screen
  expandedCamera: null,
  setExpandedCamera: (channel) => set({ expandedCamera: channel }),

  // Events
  events: [],
  setEvents: (events) => set({ events }),
  addEvents: (newEvents) => {
    const current = get().events;
    const merged = [...newEvents, ...current].slice(0, 500);
    set({ events: merged });
  },

  // Metrics
  metrics: null,
  setMetrics: (metrics) => set({ metrics }),
  batchStats: null,
  setBatchStats: (batchStats) => set({ batchStats }),

  // Class filter
  enabledClasses: new Set(Array.from({ length: 80 }, (_, i) => i)),
  setEnabledClasses: (classes) => set({ enabledClasses: new Set(classes) }),
  toggleClass: (classId) => {
    const classes = new Set(get().enabledClasses);
    if (classes.has(classId)) classes.delete(classId);
    else classes.add(classId);
    set({ enabledClasses: classes });
  },

  // Event log panel
  eventLogOpen: true,
  setEventLogOpen: (open) => set({ eventLogOpen: open }),

  // Sidebar collapsed
  sidebarCollapsed: false,
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
}));

// COCO class names
export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];
