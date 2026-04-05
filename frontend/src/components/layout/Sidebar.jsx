import { NavLink } from 'react-router-dom';
import { Camera, BarChart3, Database, Settings, ChevronLeft, ChevronRight, Shield } from 'lucide-react';
import { useStore } from '../../stores/store';

const navItems = [
  { to: '/', icon: Camera, label: 'Dashboard' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/database', icon: Database, label: 'Database' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar() {
  const collapsed = useStore(s => s.sidebarCollapsed);
  const setCollapsed = useStore(s => s.setSidebarCollapsed);

  return (
    <aside className={`fixed left-0 top-0 h-screen bg-surface border-r border-border flex flex-col z-50 transition-all duration-200 ${collapsed ? 'w-16' : 'w-56'}`}>
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-14 border-b border-border">
        <Shield className="w-7 h-7 text-neon-green flex-shrink-0" />
        {!collapsed && (
          <span className="text-neon-green font-bold text-lg font-mono text-glow-green tracking-tight">
            HomeCam
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 space-y-1 px-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group ${
                isActive
                  ? 'bg-neon-green/10 text-neon-green glow-green'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`
            }
          >
            <Icon className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center h-12 border-t border-border text-gray-500 hover:text-white transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  );
}
