import React, { useState, useEffect } from 'react';
import { Sparkles, Cpu, AlertCircle, RefreshCw, ArrowRight } from 'lucide-react';
import type { ActiveTab, Voice, HealthResponse, PublicConfigResponse, JobStatusResponse } from './lib/types';
import { api } from './lib/api';
import { Navbar } from './components/common/Navbar';
import { TTSStudio } from './components/studio/TTSStudio';
import { VCStudio } from './components/studio/VCStudio';
import { VoiceCatalog } from './components/catalog/VoiceCatalog';
import { JobHistory } from './components/catalog/JobHistory';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('start');
  const [voices, setVoices] = useState<Voice[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>('my_voice');
  const [appError, setAppError] = useState<string | null>(null);

  // Initial data loading
  const loadInitialData = async () => {
    setAppError(null);
    try {
      const [healthData, voicesData, configData] = await Promise.all([
        api.getHealth().catch(() => null),
        api.getVoices().catch(() => ({ voices: [], total: 0 })),
        api.getPublicConfig().catch(() => null),
      ]);

      setHealth(healthData);
      setVoices(voicesData.voices || []);
      setConfig(configData);

      if (voicesData.voices && voicesData.voices.length > 0) {
        setSelectedVoiceId(voicesData.voices[0].id);
      }
    } catch {
      setAppError('Failed to initialize communication with the Voice Studio engine.');
    }
  };

  useEffect(() => {
    loadInitialData();

    // Check URL search parameters on mount for active job or tab
    const params = new URLSearchParams(window.location.search);
    const urlJobId = params.get('job_id');
    const urlTab = params.get('tab') as ActiveTab | null;
    if (urlJobId) {
      setActiveTab('tts');
    } else if (urlTab && ['start', 'tts', 'vc', 'voices', 'history'].includes(urlTab)) {
      setActiveTab(urlTab);
    }

    // Heartbeat poll for backend health
    const interval = setInterval(async () => {
      try {
        const h = await api.getHealth();
        setHealth(h);
      } catch {
        setHealth(null);
      }
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (tab: ActiveTab) => {
    setActiveTab(tab);
    const params = new URLSearchParams(window.location.search);
    params.set('tab', tab);
    window.history.replaceState({}, '', `?${params.toString()}`);
  };

  const handleActiveJobChange = (jobId: string | null) => {
    const params = new URLSearchParams(window.location.search);
    if (jobId) {
      params.set('tab', 'tts');
      params.set('job_id', jobId);
    } else {
      params.delete('job_id');
    }
    const newQuery = params.toString() ? `?${params.toString()}` : window.location.pathname;
    window.history.replaceState({}, '', newQuery);
  };

  const handleDeleteJob = (jobId: string) => {
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
  };

  const handlePlayJob = (_job: JobStatusResponse) => {
    // Navigate to history or show active player
    handleTabChange('history');
  };

  // Compute "What works today" statuses dynamically from voices & config
  const textLibraryStatus = voices.some((v) => v.model_type === 'openvoice_reference')
    ? 'Ready'
    : 'Needs more voices';

  const textCustomStatus = config?.features?.openvoice !== false ? 'Ready' : 'Unavailable';

  const audioLibraryStatus = voices.some((v) => v.model_type === 'rvc_model')
    ? 'Ready'
    : 'Needs more voices';

  const audioCustomStatus = 'Coming soon';

  const renderStatusPill = (status: string) => {
    if (status === 'Ready') {
      return <span className="status-pill status-ready">Ready</span>;
    }
    if (status === 'Needs more voices') {
      return <span className="status-pill status-needs-voices">Needs more voices</span>;
    }
    return <span className="status-pill status-coming-soon">{status}</span>;
  };

  const activeJobCount = jobs.filter((j) => j.status === 'queued' || j.status === 'processing').length;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-primary)' }}>
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        health={health}
        activeJobCount={activeJobCount}
      />

      <main style={{ flex: 1 }}>
        {/* Global Connection Error Banner */}
        {appError && (
          <div style={{
            maxWidth: '1200px',
            margin: '20px auto 0 auto',
            padding: '14px 20px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: '#fef2f2',
            border: '1px solid #fee2e2',
            color: '#991b1b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.875rem' }}>
              <AlertCircle size={18} />
              <span>{appError}</span>
            </div>
            <button onClick={loadInitialData} className="btn-secondary" style={{ fontSize: '0.8125rem', padding: '4px 10px' }}>
              <RefreshCw size={13} />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* 1. Start Screen matching Mockup Page 1 */}
        {activeTab === 'start' && (
          <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '56px 24px 72px 24px' }}>
            {/* Hero Section */}
            <div style={{ marginBottom: '40px' }}>
              <h1 style={{
                fontSize: '2.5rem',
                fontWeight: 800,
                letterSpacing: '-0.03em',
                color: 'var(--text-primary)',
                marginBottom: '10px',
              }}>
                What do you want to do?
              </h1>
              <p style={{
                fontSize: '1.05rem',
                color: 'var(--text-secondary)',
              }}>
                Pick a goal. We will show which voices work for it, and what to expect.
              </p>
            </div>

            {/* Two Goal Cards side by side */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '24px',
              marginBottom: '56px',
            }}>
              {/* Card 1: Text to Voice */}
              <div
                className="studio-card studio-card-interactive"
                style={{
                  padding: '32px 28px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  borderRadius: 'var(--radius-xl)',
                }}
              >
                <div>
                  <span style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    letterSpacing: '0.08em',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    display: 'block',
                    marginBottom: '12px',
                  }}>
                    TEXT TO VOICE
                  </span>

                  <h2 style={{
                    fontSize: '1.35rem',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    marginBottom: '12px',
                    lineHeight: 1.3,
                  }}>
                    Type a script, hear it in a voice
                  </h2>

                  <p style={{
                    fontSize: '0.9375rem',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.5,
                    marginBottom: '28px',
                  }}>
                    Choose a library voice, or upload a 10 to 30 second clip of a voice you are allowed to use.
                  </p>
                </div>

                <div>
                  <button
                    onClick={() => setActiveTab('tts')}
                    className="btn-primary"
                    style={{ padding: '12px 24px', fontSize: '0.9375rem' }}
                  >
                    <span>Start with text</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              </div>

              {/* Card 2: Voice to Voice */}
              <div
                className="studio-card studio-card-interactive"
                style={{
                  padding: '32px 28px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  borderRadius: 'var(--radius-xl)',
                }}
              >
                <div>
                  <span style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    letterSpacing: '0.08em',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    display: 'block',
                    marginBottom: '12px',
                  }}>
                    VOICE TO VOICE
                  </span>

                  <h2 style={{
                    fontSize: '1.35rem',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    marginBottom: '12px',
                    lineHeight: 1.3,
                  }}>
                    Upload or record, change the voice
                  </h2>

                  <p style={{
                    fontSize: '0.9375rem',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.5,
                    marginBottom: '28px',
                  }}>
                    Turn an existing recording into another voice. Best with clean speech from one person.
                  </p>
                </div>

                <div>
                  <button
                    onClick={() => setActiveTab('vc')}
                    className="btn-primary"
                    style={{ padding: '12px 24px', fontSize: '0.9375rem' }}
                  >
                    <span>Start with audio</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              </div>
            </div>

            {/* "What works today" Table */}
            <div>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                marginBottom: '4px',
              }}>
                What works today
              </h3>
              <p style={{
                fontSize: '0.875rem',
                color: 'var(--text-muted)',
                marginBottom: '18px',
              }}>
                No guessing: each combination and its status.
              </p>

              <div className="studio-card" style={{ overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-primary)' }}>
                      <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)' }}>You give</th>
                      <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)' }}>Target voice</th>
                      <th style={{ padding: '12px 20px', fontWeight: 600, color: 'var(--text-secondary)' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '14px 20px', color: 'var(--text-primary)', fontWeight: 500 }}>Text</td>
                      <td style={{ padding: '14px 20px', color: 'var(--text-secondary)' }}>Any library voice</td>
                      <td style={{ padding: '14px 20px' }}>{renderStatusPill(textLibraryStatus)}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '14px 20px', color: 'var(--text-primary)', fontWeight: 500 }}>Text</td>
                      <td style={{ padding: '14px 20px', color: 'var(--text-secondary)' }}>My own clip (consent needed)</td>
                      <td style={{ padding: '14px 20px' }}>{renderStatusPill(textCustomStatus)}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '14px 20px', color: 'var(--text-primary)', fontWeight: 500 }}>Audio</td>
                      <td style={{ padding: '14px 20px', color: 'var(--text-secondary)' }}>A trained library voice (best quality)</td>
                      <td style={{ padding: '14px 20px' }}>{renderStatusPill(audioLibraryStatus)}</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '14px 20px', color: 'var(--text-primary)', fontWeight: 500 }}>Audio</td>
                      <td style={{ padding: '14px 20px', color: 'var(--text-secondary)' }}>My own clip (lower quality)</td>
                      <td style={{ padding: '14px 20px' }}>{renderStatusPill(audioCustomStatus)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 2. Text to Voice Studio matching Mockup Page 2 */}
        {activeTab === 'tts' && (
          <TTSStudio
            voices={voices}
            onNavigateBack={() => handleTabChange('start')}
            selectedVoiceId={selectedVoiceId}
            config={config}
            initialJobId={new URLSearchParams(window.location.search).get('job_id')}
            onActiveJobChange={handleActiveJobChange}
          />
        )}

        {/* 3. Voice to Voice Studio */}
        {activeTab === 'vc' && (
          <VCStudio
            voices={voices}
            onNavigateBack={() => setActiveTab('start')}
          />
        )}

        {/* 4. Voice Catalog Browser */}
        {activeTab === 'voices' && (
          <VoiceCatalog
            voices={voices}
            onSelectVoice={(voiceId) => {
              setSelectedVoiceId(voiceId);
              setActiveTab('tts');
            }}
          />
        )}

        {/* 5. Generation History Page */}
        {activeTab === 'history' && (
          <JobHistory
            jobs={jobs}
            onDeleteJob={handleDeleteJob}
            onPlayJob={handlePlayJob}
          />
        )}
      </main>

      {/* Production Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        backgroundColor: '#ffffff',
        padding: '28px 24px',
        marginTop: 'auto',
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem' }}>
            <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Voice Studio</span>
            <span style={{ color: 'var(--text-muted)' }}>— Free AI Voice Cloning & Conversion</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Cpu size={14} color="var(--accent-green-text)" />
              CUDA Enabled
            </span>
            <span>•</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Sparkles size={14} color="#d97706" />
              AI-generated
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
