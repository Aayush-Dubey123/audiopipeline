import React, { useState, useEffect } from 'react';
import { Waves, Shield, Cpu, RefreshCw, AlertCircle } from 'lucide-react';
import type { ActiveTab, Voice, HealthResponse, JobStatusResponse } from './lib/types';
import { api } from './lib/api';
import { Navbar } from './components/common/Navbar';
import { AudioPlayer } from './components/common/AudioPlayer';
import { TTSStudio } from './components/studio/TTSStudio';
import { VCStudio } from './components/studio/VCStudio';
import { VoiceCatalog } from './components/catalog/VoiceCatalog';
import { JobHistory } from './components/catalog/JobHistory';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('tts');
  const [voices, setVoices] = useState<Voice[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [activePlaybackJob, setActivePlaybackJob] = useState<JobStatusResponse | null>(null);
  const [appError, setAppError] = useState<string | null>(null);

  // Initial load
  const loadInitialData = async () => {
    setAppError(null);
    try {
      const [healthData, voicesData] = await Promise.all([
        api.getHealth().catch(() => null),
        api.getVoices().catch(() => ({ voices: [], total: 0 })),
      ]);

      setHealth(healthData);
      setVoices(voicesData.voices || []);
    } catch {
      setAppError('Failed to initialize communication with the AI Voice Studio engine.');
    }
  };

  useEffect(() => {
    loadInitialData();

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

  // Job tracking
  const handleJobStarted = async (jobId: string) => {
    try {
      const status = await api.getJobStatus(jobId);
      setJobs((prev) => [status, ...prev.filter((j) => j.job_id !== jobId)]);
    } catch {
      // Ignore initial poll failure
    }
  };

  const handleDeleteJob = (jobId: string) => {
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
    if (activePlaybackJob?.job_id === jobId) {
      setActivePlaybackJob(null);
    }
  };

  const handlePlayJob = (job: JobStatusResponse) => {
    setActivePlaybackJob(job);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleSelectVoiceFromCatalog = (_voiceId: string, targetTab: ActiveTab) => {
    setActiveTab(targetTab);
  };

  const activeJobCount = jobs.filter((j) => j.status === 'queued' || j.status === 'processing').length;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        activeJobCount={activeJobCount}
      />

      <main className="app-container" style={{ flex: 1 }}>
        {/* Global Error Banner */}
        {appError && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '14px 20px',
            borderRadius: '12px',
            backgroundColor: 'var(--danger-bg)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#f87171',
            marginBottom: '24px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertCircle size={20} />
              <span>{appError}</span>
            </div>
            <button onClick={loadInitialData} className="btn btn-secondary btn-sm">
              <RefreshCw size={14} /> Retry Connection
            </button>
          </div>
        )}

        {/* Global Active Playback Overlay / Bar */}
        {activePlaybackJob && activePlaybackJob.output_url && (
          <div style={{ marginBottom: '24px' }}>
            <AudioPlayer
              src={activePlaybackJob.output_url}
              title={`Playing: Job ${activePlaybackJob.job_id.slice(0, 8)}`}
              subtitle={`Kind: ${activePlaybackJob.kind.toUpperCase()} | Voice: ${activePlaybackJob.voice_id || 'Custom Voice'}`}
              wavDownloadUrl={activePlaybackJob.output_url}
              mp3DownloadUrl={`${activePlaybackJob.output_url}?format=mp3`}
              aiAttribution={true}
            />
          </div>
        )}

        {/* Active Tab View */}
        {activeTab === 'tts' && (
          <TTSStudio voices={voices} onJobStarted={handleJobStarted} />
        )}

        {activeTab === 'vc' && (
          <VCStudio voices={voices} onJobStarted={handleJobStarted} />
        )}

        {activeTab === 'catalog' && (
          <VoiceCatalog voices={voices} onSelectVoice={handleSelectVoiceFromCatalog} />
        )}

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
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        backgroundColor: 'rgba(9, 10, 15, 0.95)',
        padding: '32px 20px',
        marginTop: '60px',
      }}>
        <div style={{
          maxWidth: '1320px',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '16px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Waves size={20} color="var(--accent-violet)" />
              <span style={{ fontWeight: 700, fontSize: '0.95rem' }} className="gradient-text">
                AI Voice Studio
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>— Zero-Budget Neural Speech & Voice Cloning</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Cpu size={14} color="var(--accent-cyan)" />
                NVIDIA CUDA Accelerated
              </span>
              <span>•</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Shield size={14} color="var(--success)" />
                C2PA Watermarked
              </span>
            </div>
          </div>

          <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            <strong>Responsible AI Disclosure:</strong> Audio generated and converted by AI Voice Studio is synthesized using OpenVoice V2 and RVC neural models. All generated audio contains watermarked synthetic attribution tags. Only authorized reference recordings with confirmed ownership or explicit permission may be synthesized.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
