import React from 'react';
import { Mic, Waves, Library, History, Activity, Sparkles } from 'lucide-react';
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
  activeJobCount,
}) => {
  const isOnline = health?.status === 'ok';

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 50,
      backgroundColor: 'rgba(9, 10, 15, 0.85)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    }}>
      <div style={{
        maxWidth: '1320px',
        margin: '0 auto',
        padding: '14px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
      }}>
        {/* Brand identity */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(139, 92, 246, 0.4)',
          }}>
            <Waves size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em' }} className="gradient-text">
                AI Voice Studio
              </h1>
              <span className="badge badge-ai" style={{ fontSize: '0.65rem', padding: '2px 6px' }}>v{health?.version || '0.1'}</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Multilingual Neural TTS & RVC Conversion
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          background: 'rgba(15, 17, 26, 0.9)',
          padding: '4px',
          borderRadius: '14px',
          border: '1px solid rgba(255, 255, 255, 0.06)',
        }}>
          <button
            onClick={() => onTabChange('tts')}
            className="btn"
            style={{
              padding: '8px 16px',
              fontSize: '0.875rem',
              borderRadius: '10px',
              backgroundColor: activeTab === 'tts' ? 'var(--accent-violet)' : 'transparent',
              color: activeTab === 'tts' ? '#fff' : 'var(--text-secondary)',
              boxShadow: activeTab === 'tts' ? '0 4px 14px rgba(139, 92, 246, 0.4)' : 'none',
            }}
          >
            <Mic size={16} />
            <span>Text-to-Speech</span>
          </button>

          <button
            onClick={() => onTabChange('vc')}
            className="btn"
            style={{
              padding: '8px 16px',
              fontSize: '0.875rem',
              borderRadius: '10px',
              backgroundColor: activeTab === 'vc' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'vc' ? '#fff' : 'var(--text-secondary)',
              boxShadow: activeTab === 'vc' ? '0 4px 14px rgba(6, 182, 212, 0.4)' : 'none',
            }}
          >
            <Sparkles size={16} />
            <span>Voice Conversion</span>
          </button>

          <button
            onClick={() => onTabChange('catalog')}
            className="btn"
            style={{
              padding: '8px 16px',
              fontSize: '0.875rem',
              borderRadius: '10px',
              backgroundColor: activeTab === 'catalog' ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
              color: activeTab === 'catalog' ? '#fff' : 'var(--text-secondary)',
            }}
          >
            <Library size={16} />
            <span>Voices</span>
          </button>

          <button
            onClick={() => onTabChange('history')}
            className="btn"
            style={{
              padding: '8px 16px',
              fontSize: '0.875rem',
              borderRadius: '10px',
              backgroundColor: activeTab === 'history' ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
              color: activeTab === 'history' ? '#fff' : 'var(--text-secondary)',
              position: 'relative',
            }}
          >
            <History size={16} />
            <span>History</span>
            {activeJobCount > 0 && (
              <span style={{
                position: 'absolute',
                top: '-4px',
                right: '-4px',
                width: '18px',
                height: '18px',
                borderRadius: '50%',
                background: 'var(--accent-violet)',
                color: '#fff',
                fontSize: '0.65rem',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                {activeJobCount}
              </span>
            )}
          </button>
        </nav>

        {/* Backend health status pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 14px',
          borderRadius: '9999px',
          background: isOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          border: `1px solid ${isOnline ? 'rgba(16, 185, 129, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
          fontSize: '0.8rem',
        }}>
          <Activity size={14} color={isOnline ? '#10b981' : '#ef4444'} />
          <span style={{
            color: isOnline ? '#34d399' : '#f87171',
            fontWeight: 600,
          }}>
            {isOnline ? 'GPU / Core Ready' : 'Backend Connecting...'}
          </span>
        </div>
      </div>
    </header>
  );
};
