import React from 'react';
import type { ActiveTab, HealthResponse } from '../../lib/types';

interface NavbarProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  health: HealthResponse | null;
  activeJobCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onTabChange,
  health,
}) => {
  const isOnline = health?.status === 'ok';

  return (
    <header style={{
      backgroundColor: '#ffffff',
      borderBottom: '1px solid var(--border-subtle)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        {/* Brand identity */}
        <div
          onClick={() => onTabChange('start')}
          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <span style={{
            fontSize: '1.25rem',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            color: 'var(--text-primary)',
          }}>
            Voice Studio
          </span>
        </div>

        {/* Center navigation */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '28px' }}>
          <button
            onClick={() => onTabChange('start')}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: '0.9375rem',
              fontWeight: 600,
              color: (activeTab === 'start' || activeTab === 'tts' || activeTab === 'vc') ? 'var(--text-primary)' : 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px 0',
              fontFamily: 'inherit',
              position: 'relative',
            }}
          >
            Create
            {(activeTab === 'start' || activeTab === 'tts' || activeTab === 'vc') && (
              <span style={{
                position: 'absolute',
                bottom: '-16px',
                left: 0,
                right: 0,
                height: '2px',
                backgroundColor: 'var(--accent-green)',
                borderRadius: '2px',
              }} />
            )}
          </button>

          <button
            onClick={() => onTabChange('voices')}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: '0.9375rem',
              fontWeight: 600,
              color: activeTab === 'voices' ? 'var(--text-primary)' : 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px 0',
              fontFamily: 'inherit',
              position: 'relative',
            }}
          >
            Voices
            {activeTab === 'voices' && (
              <span style={{
                position: 'absolute',
                bottom: '-16px',
                left: 0,
                right: 0,
                height: '2px',
                backgroundColor: 'var(--accent-green)',
                borderRadius: '2px',
              }} />
            )}
          </button>

          <button
            onClick={() => onTabChange('history')}
            style={{
              border: 'none',
              background: 'transparent',
              fontSize: '0.9375rem',
              fontWeight: 600,
              color: activeTab === 'history' ? 'var(--text-primary)' : 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px 0',
              fontFamily: 'inherit',
              position: 'relative',
            }}
          >
            History
            {activeTab === 'history' && (
              <span style={{
                position: 'absolute',
                bottom: '-16px',
                left: 0,
                right: 0,
                height: '2px',
                backgroundColor: 'var(--accent-green)',
                borderRadius: '2px',
              }} />
            )}
          </button>
        </nav>

        {/* Right status badge */}
        <div>
          <span className="status-pill status-ready" style={{ fontSize: '0.8125rem', padding: '5px 12px' }}>
            <span
              className="pulse-dot"
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                backgroundColor: isOnline ? '#16a34a' : '#dc2626',
              }}
            />
            <span>{isOnline ? 'GPU ready' : 'Offline'}</span>
          </span>
        </div>
      </div>
    </header>
  );
};
