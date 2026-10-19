import React, { useEffect, useState } from 'react';
import { Button } from '../common/Button';

export interface ProgressStage {
  id: string;
  name: string;
  order: number;
}

const PREVIEW_STAGES: ProgressStage[] = [
  { id: 'queued', name: 'Queued', order: 1 },
  { id: 'loading_voice', name: 'Loading voice', order: 2 },
  { id: 'generating_speech', name: 'Generating speech', order: 3 },
  { id: 'polishing_audio', name: 'Polishing audio', order: 4 },
];

export interface ProgressPanelProps {
  currentStageName?: string;
  stageIndex?: number;
  stageTotal?: number;
  chunkDone?: number | null;
  chunkTotal?: number | null;
  etaSeconds?: number | null;
  estimatedTotalSeconds?: number;
  isColdModel?: boolean;
  isLongForm?: boolean;
  onCancel?: () => void;
  onRetry?: () => void;
  lastUpdatedAt?: Date | number;
  errorMessage?: string | null;
}

export const ProgressPanel: React.FC<ProgressPanelProps> = ({
  currentStageName = 'Queued',
  stageIndex = 1,
  stageTotal = 4,
  chunkDone,
  chunkTotal,
  etaSeconds,
  estimatedTotalSeconds = 5,
  isColdModel = false,
  isLongForm = false,
  onCancel,
  onRetry,
  lastUpdatedAt,
  errorMessage,
}) => {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [secondsSinceLastUpdate, setSecondsSinceLastUpdate] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - start) / 1000));
      if (lastUpdatedAt) {
        const updateTime = typeof lastUpdatedAt === 'number' ? lastUpdatedAt : lastUpdatedAt.getTime();
        setSecondsSinceLastUpdate(Math.max(0, Math.floor((Date.now() - updateTime) / 1000)));
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [lastUpdatedAt]);

  const isStalled = secondsSinceLastUpdate >= 45;

  // Compute active stage index based on currentStageName or stageIndex
  const activeOrder = (() => {
    const normalized = (currentStageName || '').toLowerCase().trim();
    if (normalized.includes('loading')) return 2;
    if (normalized.includes('speech') || normalized.includes('synthes')) return 3;
    if (normalized.includes('polish') || normalized.includes('export')) return 4;
    if (normalized.includes('complete')) return 4;
    return stageIndex || 1;
  })();

  const percentComplete = isLongForm && chunkTotal && chunkTotal > 0
    ? Math.min(100, Math.round(((chunkDone || 0) / chunkTotal) * 100))
    : Math.min(100, Math.round((activeOrder / stageTotal) * 100));

  return (
    <div
      role="region"
      aria-live="polite"
      style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-surface)',
        padding: 'var(--space-2)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-2)',
      }}
    >
      {/* Stage Progression Bar */}
      <div style={{ display: 'flex', gap: 'var(--space-1)', alignItems: 'center' }}>
        {PREVIEW_STAGES.map((s) => {
          const isDone = s.order < activeOrder;
          const isCurrent = s.order === activeOrder;
          return (
            <div
              key={s.id}
              style={{
                flex: 1,
                padding: 'var(--space-1)',
                borderRadius: 'var(--radius-control)',
                backgroundColor: isCurrent
                  ? 'var(--color-accent-light)'
                  : isDone
                  ? 'var(--color-surface-muted)'
                  : 'transparent',
                border: `1px solid ${
                  isCurrent
                    ? 'var(--color-accent)'
                    : isDone
                    ? 'var(--color-border)'
                    : 'var(--color-border)'
                }`,
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-0-5)',
              }}
            >
              <div
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: isCurrent
                    ? 'var(--color-accent)'
                    : isDone
                    ? 'var(--color-success)'
                    : 'var(--color-border)',
                  flexShrink: 0,
                }}
              />
              <span
                style={{
                  fontSize: 'var(--font-sm)',
                  fontWeight: isCurrent ? 600 : 500,
                  color: isCurrent ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {s.name}
              </span>
            </div>
          );
        })}
      </div>

      {/* Primary Details Row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 'var(--space-1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            {isLongForm && chunkTotal
              ? `Chunk ${chunkDone || 0} of ${chunkTotal} (${percentComplete}%)`
              : currentStageName}
          </span>
          <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
            Elapsed: {elapsedSeconds}s
          </span>
          {(etaSeconds !== null && etaSeconds !== undefined && etaSeconds > 0) ? (
            <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
              ETA: ~{etaSeconds}s
            </span>
          ) : (
            <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
              Usually about {Math.max(2, Math.round(estimatedTotalSeconds))}s
            </span>
          )}
        </div>

        <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
          Last update: {secondsSinceLastUpdate}s ago
        </span>
      </div>

      {/* Progress Track */}
      <div
        style={{
          height: '6px',
          width: '100%',
          backgroundColor: 'var(--color-surface-muted)',
          borderRadius: 'var(--radius-control)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${percentComplete}%`,
            backgroundColor: 'var(--color-accent)',
            transition: 'width 250ms ease',
          }}
        />
      </div>

      {/* Cold Model Note */}
      {isColdModel && (
        <div
          style={{
            fontSize: 'var(--font-sm)',
            color: 'var(--color-warning)',
            backgroundColor: 'var(--color-warning-bg)',
            border: '1px solid var(--color-warning-border)',
            borderRadius: 'var(--radius-control)',
            padding: 'var(--space-0-5) var(--space-1)',
          }}
        >
          First run is slower while the voice loads.
        </div>
      )}

      {/* Working & Stalled States */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 'var(--space-1)',
        }}
      >
        {isStalled ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
            <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-warning)', fontWeight: 600 }}>
              Taking longer than usual.
            </span>
            {onRetry && (
              <Button size="sm" onClick={onRetry}>
                Retry
              </Button>
            )}
          </div>
        ) : (
          <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)' }}>
            {isLongForm
              ? 'Working. You can leave and come back.'
              : 'Working. It is safe to wait, please keep this tab open.'}
          </span>
        )}

        {onCancel && (
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>

      {/* Error Message if failed */}
      {errorMessage && (
        <div
          style={{
            fontSize: 'var(--font-sm)',
            color: 'var(--color-error)',
            backgroundColor: 'var(--color-error-bg)',
            border: '1px solid var(--color-error-border)',
            borderRadius: 'var(--radius-control)',
            padding: 'var(--space-1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span>{errorMessage}</span>
          {onRetry && (
            <Button size="sm" variant="danger" onClick={onRetry}>
              Retry
            </Button>
          )}
        </div>
      )}
    </div>
  );
};
