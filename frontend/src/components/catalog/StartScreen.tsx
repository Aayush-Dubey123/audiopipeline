import React from 'react';
import type { Voice, PublicConfigResponse } from '../../lib/types';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

export interface StartScreenProps {
  voices: Voice[];
  config: PublicConfigResponse | null;
  onSelectTextToVoice: () => void;
  onSelectVoiceToVoice: () => void;
}

export const StartScreen: React.FC<StartScreenProps> = ({
  voices,
  config,
  onSelectTextToVoice,
  onSelectVoiceToVoice,
}) => {
  // Dynamically compute feature statuses from /v1/voices and /v1/config/public (never hardcoded)
  const hasLibraryVoices = voices.some(
    (v) => (v.status === 'ready' || v.status === 'available')
  );

  const isCustomTTSEnabled = config?.features?.chatterbox !== false;

  const statusPills = [
    {
      id: 'text-library',
      title: 'Text to library voice',
      ready: hasLibraryVoices,
      detail: hasLibraryVoices
        ? `${voices.filter((v) => v.status === 'ready' || v.status === 'available').length} voice profiles loaded and verified.`
        : 'Catalog voice models are initializing.',
    },
    {
      id: 'text-reference',
      title: 'Text to my reference',
      ready: isCustomTTSEnabled,
      detail: isCustomTTSEnabled
        ? 'Instant zero-shot cloning enabled via Chatterbox pipeline.'
        : 'Custom voice cloning is currently disabled by administrator configuration.',
    },
    {
      id: 'voice-to-voice',
      title: 'Voice to voice',
      ready: false,
      detail: 'Voice to voice audio conversion is coming soon.',
    },
  ];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%',
        width: '100%',
        padding: 'var(--space-2)',
        gap: 'var(--space-3)',
      }}
    >
      {/* Two Equal Cards Side by Side */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 'var(--space-3)',
          width: '100%',
          maxWidth: '820px',
        }}
      >
        {/* Card 1: Text to Voice (Primary) */}
        <Card
          interactive
          onClick={onSelectTextToVoice}
          padding="lg"
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            minHeight: '180px',
            backgroundColor: 'var(--color-surface)',
            border: '2px solid var(--color-accent)',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 'var(--font-lg)', fontWeight: 700, color: 'var(--color-accent)' }}>
                Text to voice
              </span>
              <span
                style={{
                  fontSize: 'var(--font-sm)',
                  backgroundColor: 'var(--color-accent-light)',
                  color: 'var(--color-accent)',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-control)',
                  fontWeight: 600,
                }}
              >
                Primary
              </span>
            </div>
            <p style={{ fontSize: 'var(--font-base)', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
              Generate studio-quality speech in English, Hindi, and Marathi from script text with verified voices or your own reference.
            </p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 'var(--space-2)' }}>
            <Button variant="primary" size="base" onClick={onSelectTextToVoice}>
              Open Studio →
            </Button>
          </div>
        </Card>

        {/* Card 2: Voice to Voice */}
        <Card
          interactive
          onClick={onSelectVoiceToVoice}
          padding="lg"
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            minHeight: '180px',
            backgroundColor: 'var(--color-surface)',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 'var(--font-lg)', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                Voice to voice
              </span>
              <span
                style={{
                  fontSize: 'var(--font-sm)',
                  backgroundColor: 'var(--color-surface-muted)',
                  color: 'var(--color-text-muted)',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-control)',
                  fontWeight: 600,
                }}
              >
                Coming soon
              </span>
            </div>
            <p style={{ fontSize: 'var(--font-base)', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
              Transform spoken audio into another voice with pitch preservation. Voice to voice synthesis is coming soon.
            </p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 'var(--space-2)' }}>
            <Button variant="secondary" size="base" onClick={onSelectVoiceToVoice}>
              Preview Feature →
            </Button>
          </div>
        </Card>
      </div>

      {/* ONE Compact Row: What works today */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 'var(--space-1)',
          width: '100%',
          maxWidth: '820px',
        }}
      >
        <span
          style={{
            fontSize: 'var(--font-sm)',
            fontWeight: 600,
            color: 'var(--color-text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}
        >
          What works today
        </span>

        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: 'var(--space-1)',
            width: '100%',
          }}
        >
          {statusPills.map((pill) => (
            <div
              key={pill.id}
              tabIndex={0}
              title={pill.detail}
              role="status"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 'var(--space-0-5)',
                padding: '4px 10px',
                borderRadius: 'var(--radius-control)',
                backgroundColor: pill.ready ? 'var(--color-success-bg)' : 'var(--color-neutral-pill-bg)',
                color: pill.ready ? 'var(--color-success)' : 'var(--color-neutral-pill-text)',
                border: `1px solid ${pill.ready ? 'var(--color-success-border)' : 'var(--color-border)'}`,
                fontSize: 'var(--font-sm)',
                fontWeight: 500,
                cursor: 'help',
                userSelect: 'none',
              }}
            >
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  backgroundColor: pill.ready ? 'var(--color-success)' : 'var(--color-text-muted)',
                }}
              />
              <span>{pill.title}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
