import React from 'react';
import type { Voice, PublicConfigResponse, JobStatusResponse } from '../../lib/types';
import { Button } from '../common/Button';
import { Card } from '../common/Card';

export interface VoiceToVoiceProps {
  voices?: Voice[];
  config?: PublicConfigResponse | null;
  onNavigateHome: () => void;
  onJobCreated?: (job: JobStatusResponse) => void;
}

export const VoiceToVoice: React.FC<VoiceToVoiceProps> = ({
  onNavigateHome,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        width: '100%',
        padding: 'var(--space-2)',
      }}
    >
      <Card
        padding="lg"
        style={{
          maxWidth: '460px',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
          gap: 'var(--space-2)',
          border: '1px solid var(--color-border)',
        }}
      >
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            backgroundColor: 'var(--color-accent-light)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '22px',
          }}
        >
          🎙
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-0-5)' }}>
          <span style={{ fontSize: 'var(--font-lg)', fontWeight: 700, color: 'var(--color-text-primary)' }}>
            Voice to voice: coming soon
          </span>
          <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
            Voice to voice conversion is coming in a future release. Text to speech with reference voice cloning is currently active.
          </p>
        </div>

        <Button variant="primary" size="base" onClick={onNavigateHome} style={{ width: '100%' }}>
          ← Back to Studio
        </Button>
      </Card>
    </div>
  );
};
