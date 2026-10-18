import React from 'react';
import {
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Trash2,
  Download,
  Volume2,
} from 'lucide-react';
import type { JobStatusResponse } from '../../lib/types';
import { api } from '../../lib/api';

interface JobHistoryProps {
  jobs: JobStatusResponse[];
  onDeleteJob: (jobId: string) => void;
  onPlayJob: (job: JobStatusResponse) => void;
}

export const JobHistory: React.FC<JobHistoryProps> = ({ jobs, onDeleteJob, onPlayJob }) => {
  const handleDelete = async (jobId: string) => {
    try {
      await api.deleteJob(jobId);
      onDeleteJob(jobId);
    } catch {
      onDeleteJob(jobId);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <span className="status-pill status-ready" style={{ fontSize: '0.75rem', padding: '3px 8px' }}>
            <CheckCircle2 size={12} /> Completed
          </span>
        );
      case 'processing':
        return (
          <span className="status-pill status-needs-voices" style={{ fontSize: '0.75rem', padding: '3px 8px' }}>
            <Loader2 size={12} className="pulse-dot" /> Processing
          </span>
        );
      case 'queued':
        return (
          <span className="status-pill status-coming-soon" style={{ fontSize: '0.75rem', padding: '3px 8px' }}>
            <Clock size={12} /> Queued
          </span>
        );
      case 'failed':
        return (
          <span className="status-pill" style={{ backgroundColor: '#fee2e2', color: '#b91c1c', border: '1px solid #fecaca', fontSize: '0.75rem', padding: '3px 8px' }}>
            <AlertCircle size={12} /> Failed
          </span>
        );
      default:
        return <span className="status-pill status-coming-soon">{status}</span>;
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)', marginBottom: '8px' }}>
          Generation History
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>
          Real-time tracking of jobs processed during this session. Output audio files are retained on the server for 60 minutes.
        </p>
      </div>

      {jobs.length === 0 ? (
        <div className="studio-card" style={{ padding: '64px 24px', textAlign: 'center', color: 'var(--text-muted)' }}>
          <Clock size={40} style={{ margin: '0 auto 16px auto', opacity: 0.4 }} />
          <h3 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
            No generation jobs yet
          </h3>
          <p style={{ fontSize: '0.875rem', maxWidth: '420px', margin: '0 auto' }}>
            Submit a Text-to-Speech synthesis or Voice Conversion request to monitor its status and download audio results here.
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {jobs.map((job) => {
            const isCompleted = job.status === 'completed';
            const isFailed = job.status === 'failed';

            return (
              <div
                key={job.job_id}
                className="studio-card"
                style={{
                  padding: '18px 24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'var(--bg-tertiary)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid var(--border-subtle)',
                  }}>
                    <Volume2 size={18} color="var(--accent-green)" />
                  </div>

                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.9375rem', color: 'var(--text-primary)' }}>
                        {job.kind === 'tts'
                          ? 'Text to voice'
                          : job.kind === 'tts_custom'
                          ? 'Custom voice clone'
                          : 'Voice to voice'}
                      </span>
                      {getStatusBadge(job.status)}
                    </div>

                    <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {job.progress_message || `Job ID: ${job.job_id}`}
                      {isFailed && job.error_message && (
                        <span style={{ color: '#b91c1c', marginLeft: '6px' }}>— {job.error_message}</span>
                      )}
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {isCompleted && job.output_url && (
                    <>
                      <button
                        onClick={() => onPlayJob(job)}
                        className="btn-primary"
                        style={{ padding: '6px 12px', fontSize: '0.8125rem' }}
                      >
                        <Volume2 size={14} />
                        <span>Play</span>
                      </button>

                      <a
                        href={job.output_url}
                        download={`audio_${job.job_id}.wav`}
                        className="btn-secondary"
                        style={{ padding: '6px 12px', fontSize: '0.8125rem', textDecoration: 'none' }}
                      >
                        <Download size={14} />
                        <span>WAV</span>
                      </a>
                    </>
                  )}

                  <button
                    onClick={() => handleDelete(job.job_id)}
                    className="btn-secondary"
                    title="Remove job"
                    style={{ padding: '6px 10px', color: 'var(--text-muted)' }}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
