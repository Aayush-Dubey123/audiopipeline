import React, { useState, useEffect } from 'react';
import type { ActiveTab, Voice, HealthResponse, PublicConfigResponse, JobStatusResponse } from './lib/types';
import { api } from './lib/api';
import { Button } from './components/common/Button';
import { Toast } from './components/feedback/Toast';
import { StartScreen } from './components/catalog/StartScreen';
import { HistoryDrawer } from './components/catalog/HistoryDrawer';
import { TTSWizard } from './components/studio/TTSWizard';
import { VoiceToVoice } from './components/studio/VoiceToVoice';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>(() => {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get('tab') as ActiveTab | null;
    if (params.get('job_id')) return 'tts';
    return tabParam && ['start', 'tts', 'vc'].includes(tabParam) ? tabParam : 'start';
  });

  const [voices, setVoices] = useState<Voice[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastType, setToastType] = useState<'info' | 'success' | 'error'>('info');

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      try {
        const [healthData, voicesData, configData] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getVoices().catch(() => ({ voices: [], total: 0 })),
          api.getPublicConfig().catch(() => null),
        ]);
        if (isMounted) {
          setHealth(healthData);
          setVoices(voicesData.voices || []);
          setConfig(configData);
        }
      } catch {
        if (isMounted) {
          setToastMessage('Could not connect to backend service.');
          setToastType('error');
        }
      }
    };
    init();

    // Health check polling
    const timer = setInterval(async () => {
      try {
        const h = await api.getHealth();
        if (isMounted) setHealth(h);
      } catch {
        if (isMounted) setHealth(null);
      }
    }, 15000);

    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, []);

  const handleTabChange = (tab: ActiveTab) => {
    setActiveTab(tab);
    const params = new URLSearchParams(window.location.search);
    params.set('tab', tab);
    if (tab === 'start') {
      params.delete('step');
      params.delete('job_id');
    }
    window.history.replaceState({}, '', `?${params.toString()}`);
  };

  const handleJobCreated = (newJob: JobStatusResponse) => {
    setJobs((prev) => {
      const filtered = prev.filter((j) => j.job_id !== newJob.job_id);
      return [newJob, ...filtered];
    });
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      await api.deleteJob(jobId);
      setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
      setToastMessage('Job deleted from history.');
      setToastType('info');
    } catch {
      setToastMessage('Failed deleting job.');
      setToastType('error');
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100dvh',
        maxHeight: '100dvh',
        width: '100vw',
        overflow: 'hidden',
        backgroundColor: 'var(--color-bg)',
      }}
    >
      {/* GLOBAL SLIM HEADER (44px) */}
      <header
        role="banner"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 var(--space-2)',
          height: '44px',
          minHeight: '44px',
          backgroundColor: 'var(--color-surface)',
          borderBottom: '1px solid var(--color-border)',
          flexShrink: 0,
          zIndex: 10,
        }}
      >
        {/* Brand / Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
          <button
            type="button"
            onClick={() => handleTabChange('start')}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 'var(--space-1)',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
              padding: 0,
            }}
          >
            <span
              style={{
                width: '24px',
                height: '24px',
                borderRadius: 'var(--radius-control)',
                backgroundColor: 'var(--color-accent)',
                color: 'var(--color-text-on-accent)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '12px',
                fontWeight: 700,
              }}
            >
              V
            </span>
            <h1
              style={{
                fontSize: 'var(--font-base)',
                fontWeight: 700,
                color: 'var(--color-text-primary)',
                letterSpacing: '-0.01em',
              }}
            >
              Voice Studio
            </h1>
          </button>

          {/* Health Indicator Dot */}
          <span
            tabIndex={0}
            role="status"
            aria-label={health?.status === 'ok' ? 'Backend online' : 'Backend offline'}
            title={health?.status === 'ok' ? 'Backend healthy and online' : 'Backend unreachable'}
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: health?.status === 'ok' ? 'var(--color-success)' : 'var(--color-error)',
              marginLeft: 'var(--space-0-5)',
              display: 'inline-block',
            }}
          />
        </div>

        {/* History Action Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setIsHistoryOpen(true)}
            aria-label="Open generation history"
          >
            <span>History</span>
            {jobs.length > 0 && (
              <span
                style={{
                  fontSize: 'var(--font-sm)',
                  fontWeight: 700,
                  backgroundColor: 'var(--color-accent-light)',
                  color: 'var(--color-accent)',
                  padding: '1px 6px',
                  borderRadius: 'var(--radius-control)',
                }}
              >
                {jobs.length}
              </span>
            )}
          </Button>
        </div>
      </header>

      {/* MAIN VIEWPORT (Fills available space, strictly no page scroll) */}
      <main
        role="main"
        style={{
          flex: 1,
          height: 'calc(100dvh - 44px)',
          maxHeight: 'calc(100dvh - 44px)',
          overflow: 'hidden',
          padding: 'var(--space-1) var(--space-2) var(--space-2) var(--space-2)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {activeTab === 'start' && (
          <StartScreen
            voices={voices}
            config={config}
            onSelectTextToVoice={() => handleTabChange('tts')}
            onSelectVoiceToVoice={() => handleTabChange('vc')}
          />
        )}

        {activeTab === 'tts' && (
          <TTSWizard
            voices={voices}
            config={config}
            onJobCreated={handleJobCreated}
            onNavigateHome={() => handleTabChange('start')}
          />
        )}

        {activeTab === 'vc' && (
          <VoiceToVoice
            voices={voices}
            config={config}
            onNavigateHome={() => handleTabChange('start')}
            onJobCreated={handleJobCreated}
          />
        )}
      </main>

      {/* HISTORY DRAWER (Right-side drawer) */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        jobs={jobs}
        onDeleteJob={handleDeleteJob}
      />

      {/* GLOBAL TOAST NOTIFICATIONS */}
      <Toast
        message={toastMessage}
        type={toastType}
        onClose={() => setToastMessage(null)}
      />
    </div>
  );
};
export default App;
