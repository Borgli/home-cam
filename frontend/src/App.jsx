import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Database from './pages/Database';
import Settings from './pages/Settings';
import { useStore } from './stores/store';
import { api } from './api/client';
import { usePolling } from './hooks/usePolling';
import { useCallback } from 'react';

export default function App() {
  const sidebarCollapsed = useStore(s => s.sidebarCollapsed);
  const setConfig = useStore(s => s.setConfig);
  const setMetrics = useStore(s => s.setMetrics);
  const setBatchStats = useStore(s => s.setBatchStats);
  const addEvents = useStore(s => s.addEvents);

  const pollData = useCallback(async () => {
    try {
      const [config, metrics, batchStats] = await Promise.all([
        api.getConfig(),
        api.getMetrics().catch(() => null),
        api.getBatchStats().catch(() => null),
      ]);
      setConfig(config);
      if (metrics) {
        setMetrics(metrics);
        if (metrics.detections?.length) {
          addEvents(metrics.detections.slice(-50));
        }
      }
      if (batchStats) setBatchStats(batchStats);
    } catch {
      // Backend not available
    }
  }, [setConfig, setMetrics, setBatchStats, addEvents]);

  usePolling(pollData, 2000);

  return (
    <div className="flex h-screen bg-surface-dark">
      <Sidebar />
      <div className={`flex-1 flex flex-col min-w-0 transition-all duration-200 ${sidebarCollapsed ? 'ml-16' : 'ml-56'}`}>
        <Header />
        <main className="flex-1 overflow-auto p-4">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/database" element={<Database />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
