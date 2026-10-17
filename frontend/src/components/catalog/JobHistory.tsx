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
      // Optimistic delete
      onDeleteJob(jobId);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <span className="badge badge-success">
            <CheckCircle2 size={12} /> Completed
          </span>
        );
      case 'processing':
        return (
          <span className="badge badge-violet animate-pulse-glow">
            <Loader2 size={12} className="animate-spin" /> Processing
          </span>
        );
      case 'queued':
        return (
          <span className="badge badge-cyan">
            <Clock size={12} /> Queued
          </span>
        );
      case 'failed':
        return (
          <span className="badge badge-danger">
            <AlertCircle size={12} /> Failed
          </span>
        );
      default:
        return <span className="badge">{status}</span>;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }} className="gradient-text">
          Session Generation History
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          Real-time tracking of jobs processed during this session. Output audio files are retained on the server for 60 minutes.
        </p>
      </div>

      {jobs.length === 0 ? (
        <div className="glass-panel" style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
          <Clock size={40} style={{ margin: '0 auto 16px auto', opacity: 0.4 }} />
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
            No Active Generations Yet
          </h3>
          <p style={{ fontSize: '0.9rem', maxWidth: '420px', margin: '0 auto' }}>
            Submit a Text-to-Speech synthesis or Voice Conversion request in the studio to monitor its inference status and download results here.
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
                className="glass-panel"
                style={{
                  padding: '16px 20px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '14px',
                  borderRadius: '14px',
                }}
              >
                {/* Left Info */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                  <div style={{
                    width: '42px',
                    height: '42px',
                    borderRadius: '10px',
                    background: job.kind === 'vc' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(139, 92, 246, 0.15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: `1px solid ${job.kind === 'vc' ? 'rgba(6, 182, 212, 0.3)' : 'rgba(139, 92, 246, 0.3)'}`,
                  }}>
                    <Volume2 size={20} color={job.kind === 'vc' ? 'var(--accent-cyan)' : 'var(--accent-violet)'} />
                  </div>

                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>
                        {job.kind === 'tts'
                          ? 'Text-to-Speech'
                          : job.kind === 'tts_custom'
                          ? 'Custom Voice Clone'
                          : 'Voice Conversion'}
                      </span>
                      {getStatusBadge(job.status)}
                    </div>

                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {job.progress_message || `Job ID: ${job.job_id}`}
                      {isFailed && job.error_message && (
                        <span style={{ color: 'var(--danger)', marginLeft: '6px' }}>— {job.error_message}</span>
                      )}
                    </p>
                  </div>
                </div>

                {/* Right Actions */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {isCompleted && job.output_url && (
                    <>
                      <button
                        onClick={() => onPlayJob(job)}
                        className="btn btn-primary btn-sm"
                        title="Load in Audio Player"
                      >
                        <Volume2 size={14} />
                        <span>Play</span>
                      </button>

                      <a
                        href={job.output_url}
                        download={`audio_${job.job_id}.wav`}
                        className="btn btn-secondary btn-sm"
                        title="Download WAV"
                      >
                        <Download size={14} />
                        <span>WAV</span>
                      </a>
                    </>
                  )}

                  <button
                    onClick={() => handleDelete(job.job_id)}
                    className="btn btn-secondary btn-sm"
                    title="Remove from history & delete server artifacts"
                    style={{ color: 'var(--text-muted)' }}
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
