import React, { useState } from 'react';
import { Play, Pause, ArrowRight, Globe, Shield } from 'lucide-react';
import type { Voice } from '../../lib/types';
import { api } from '../../lib/api';

interface VoiceCatalogProps {
  voices: Voice[];
  onSelectVoice: (voiceId: string) => void;
}

export const VoiceCatalog: React.FC<VoiceCatalogProps> = ({ voices, onSelectVoice }) => {
  const [playingVoiceId, setPlayingVoiceId] = useState<string | null>(null);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

  const togglePreview = (voiceId: string) => {
    if (playingVoiceId === voiceId && audioElement) {
      audioElement.pause();
      setPlayingVoiceId(null);
    } else {
      if (audioElement) {
        audioElement.pause();
      }
      const audio = new Audio(api.getVoicePreviewUrl(voiceId));
      audio.onended = () => setPlayingVoiceId(null);
      audio.onerror = () => setPlayingVoiceId(null);
      audio.play().catch(() => setPlayingVoiceId(null));
      setAudioElement(audio);
      setPlayingVoiceId(voiceId);
    }
  };

  const languageLabels: Record<string, string> = {
    en: 'English',
    hi: 'Hindi (हिन्दी)',
    mr: 'Marathi (मराठी)',
    ja: 'Japanese (日本語)',
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)', marginBottom: '8px' }}>
          Voice Library
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>
          Explore available reference voices and neural conversion models. Only verified and permitted voices are exposed.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '24px' }}>
        {voices.map((voice) => {
          const isPlaying = playingVoiceId === voice.id;

          return (
            <div
              key={voice.id}
              className="studio-card studio-card-interactive"
              style={{
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '16px',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '8px' }}>
                  <div>
                    <h3 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                      {voice.name}
                    </h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      ID: {voice.id}
                    </span>
                  </div>

                  <span className={`status-pill ${voice.model_type === 'rvc_model' ? 'status-needs-voices' : 'status-ready'}`}>
                    {voice.model_type === 'rvc_model' ? 'RVC Model' : 'OpenVoice Ref'}
                  </span>
                </div>

                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.5 }}>
                  {voice.description}
                </p>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap', marginBottom: '14px' }}>
                  <Globe size={14} color="var(--text-muted)" />
                  {voice.languages.map((lang) => (
                    <span
                      key={lang}
                      style={{
                        fontSize: '0.75rem',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        backgroundColor: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-secondary)',
                      }}
                    >
                      {languageLabels[lang] || lang.toUpperCase()}
                    </span>
                  ))}
                </div>

                <div style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-green-text)', fontWeight: 600 }}>
                    <Shield size={13} />
                    <span>Usage Permission & Provenance</span>
                  </div>
                  <div>{voice.usage_permission}</div>
                  <div><strong>License:</strong> {voice.license}</div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)' }}>
                {voice.has_preview ? (
                  <button
                    onClick={() => togglePreview(voice.id)}
                    className="btn-secondary"
                    style={{ fontSize: '0.8125rem', padding: '6px 12px' }}
                  >
                    {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                    <span>{isPlaying ? 'Pause' : 'Play Sample'}</span>
                  </button>
                ) : (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    No sample preview
                  </span>
                )}

                <button
                  onClick={() => onSelectVoice(voice.id)}
                  className="btn-primary"
                  style={{ fontSize: '0.8125rem', padding: '6px 14px' }}
                >
                  <span>Use Voice</span>
                  <ArrowRight size={14} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
