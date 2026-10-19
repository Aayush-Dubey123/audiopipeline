import React from 'react';
import type { JobStatusResponse } from '../../lib/types';
import { Drawer } from '../navigation/Drawer';
import { Button } from '../common/Button';
import { AudioPlayer } from '../audio/AudioPlayer';

export interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  jobs: JobStatusResponse[];
  onDeleteJob: (jobId: string) => void;
  onResumeJob?: (jobId: string) => void;
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  jobs,
  onDeleteJob,
  onResumeJob,
}) => {
  return (
    <Drawer isOpen={isOpen} onClose={onClose} title="Generation History">
      {jobs.length === 0 ? (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '240px',
            textAlign: 'center',
            gap: 'var(--space-1)',
            color: 'var(--color-text-muted)',
          }}
        >
          <span style={{ fontSize: '24px' }}>📜</span>
          <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>No history records</span>
          <span style={{ fontSize: 'var(--font-sm)' }}>
            Completed and active generation jobs will be preserved and accessible here.
          </span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          {jobs.map((job) => {
            const isCompleted = job.status === 'completed';
            const isFailed = job.status === 'failed';
            const formattedDate = new Date(job.created_at).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            });

            return (
              <div
                key={job.job_id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 'var(--space-1)',
                  backgroundColor: 'var(--color-surface)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-control)',
                  padding: 'var(--space-1) var(--space-2)',
                }}
              >
                {/* Header row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                    <span
                      style={{
                        fontSize: 'var(--font-sm)',
                        fontWeight: 600,
                        color: 'var(--color-text-primary)',
                      }}
                    >
                      {job.voice_id || 'Voice'}
                    </span>
                    <span
                      style={{
                        fontSize: 'var(--font-sm)',
                        padding: '1px 6px',
                        borderRadius: 'var(--radius-control)',
                        backgroundColor: isCompleted
                          ? 'var(--color-success-bg)'
                          : isFailed
                          ? 'var(--color-error-bg)'
                          : 'var(--color-accent-light)',
                        color: isCompleted
                          ? 'var(--color-success)'
                          : isFailed
                          ? 'var(--color-error)'
                          : 'var(--color-accent)',
                        fontWeight: 600,
                      }}
                    >
                      {job.status}
                    </span>
                  </div>

                  <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                    {formattedDate}
                  </span>
                </div>

                {/* Audio player if completed */}
                {isCompleted && (
                  <AudioPlayer
                    src={job.output_url || `/v1/audio/${job.job_id}/output`}
                    compact
                  />
                )}

                {/* Progress / Failure text */}
                {!isCompleted && (
                  <p style={{ fontSize: 'var(--font-sm)', color: isFailed ? 'var(--color-error)' : 'var(--color-text-secondary)' }}>
                    {job.progress_message || job.error_message || 'In progress...'}
                  </p>
                )}

                {/* Actions row */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginTop: 'var(--space-0-5)',
                  }}
                >
                  <div style={{ display: 'flex', gap: 'var(--space-1)' }}>
                    {isCompleted && (
                      <a
                        href={job.output_url || `/v1/audio/${job.job_id}/output?format=mp3`}
                        download={`audio_${job.job_id}.mp3`}
                        style={{
                          fontSize: 'var(--font-sm)',
                          color: 'var(--color-accent)',
                          fontWeight: 600,
                          textDecoration: 'none',
                        }}
                      >
                        Download MP3
                      </a>
                    )}
                    {isFailed && onResumeJob && (
                      <Button size="sm" onClick={() => onResumeJob(job.job_id)}>
                        Resume
                      </Button>
                    )}
                  </div>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onDeleteJob(job.job_id)}
                    aria-label={`Delete job ${job.job_id}`}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Drawer>
  );
};
