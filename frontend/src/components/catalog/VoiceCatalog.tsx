import React, { useState } from 'react';
import { Play, Pause, ShieldCheck, ArrowRight, Globe } from 'lucide-react';
import type { Voice, ActiveTab } from '../../lib/types';
import { api } from '../../lib/api';

interface VoiceCatalogProps {
  voices: Voice[];
  onSelectVoice: (voiceId: string, targetTab: ActiveTab) => void;
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }} className="gradient-text">
          Voice Model Catalog
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          Explore verified reference voices and neural conversion models. All publicly exposed voices are verified for usage permissions.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '20px' }}>
        {voices.map((voice) => {
          const isPlaying = playingVoiceId === voice.id;

          return (
            <div
              key={voice.id}
              className="glass-panel glass-panel-interactive"
              style={{
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '16px',
                borderRadius: '16px',
              }}
            >
              <div>
                {/* Header: Name + Badges */}
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '8px' }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                      {voice.name}
                    </h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      ID: {voice.id}
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px' }}>
                    <span className={`badge ${voice.model_type === 'rvc_model' ? 'badge-cyan' : 'badge-violet'}`}>
                      {voice.model_type === 'rvc_model' ? 'RVC Model' : 'OpenVoice Ref'}
                    </span>
                    {voice.status === 'experimental' && (
                      <span className="badge badge-warning" style={{ fontSize: '0.65rem' }}>
                        Experimental
                      </span>
                    )}
                  </div>
                </div>

                {/* Description */}
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.5 }}>
                  {voice.description}
                </p>

                {/* Supported Languages */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                  <Globe size={14} color="var(--text-muted)" />
                  {voice.languages.map((lang) => (
                    <span
                      key={lang}
                      style={{
                        fontSize: '0.75rem',
                        padding: '2px 8px',
                        borderRadius: '6px',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-primary)',
                      }}
                    >
                      {languageLabels[lang] || lang.toUpperCase()}
                    </span>
                  ))}
                </div>

                {/* License & Permission */}
                <div style={{
                  padding: '10px 12px',
                  borderRadius: '10px',
                  backgroundColor: 'rgba(10, 12, 18, 0.6)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.775rem',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#a78bfa' }}>
                    <ShieldCheck size={13} />
                    <strong>Permission & Provenance</strong>
                  </div>
                  <div>{voice.usage_permission}</div>
                  <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>
                    <strong>License:</strong> {voice.license}
                  </div>
                </div>
              </div>

              {/* Bottom Actions: Preview Playback + Use In Studio */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', paddingTop: '12px', borderTop: '1px solid var(--border-subtle)' }}>
                {voice.has_preview ? (
                  <button
                    onClick={() => togglePreview(voice.id)}
                    className="btn btn-secondary btn-sm"
                  >
                    {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                    <span>{isPlaying ? 'Pause Preview' : 'Play Sample'}</span>
                  </button>
                ) : (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    No sample preview
                  </span>
                )}

                <button
                  onClick={() => onSelectVoice(voice.id, voice.model_type === 'rvc_model' ? 'vc' : 'tts')}
                  className="btn btn-primary btn-sm"
                >
                  <span>Use in {voice.model_type === 'rvc_model' ? 'VC Studio' : 'TTS Studio'}</span>
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
