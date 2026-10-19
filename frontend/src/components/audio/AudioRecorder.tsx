import React, { useState, useRef } from 'react';
import type { ReferenceCheckResponse } from '../../lib/types';
import { Button } from '../common/Button';
import { AudioPlayer } from './AudioPlayer';

export interface AudioRecorderProps {
  onReferenceReady: (file: File) => void;
  referenceAudioUrl: string | null;
  consent: boolean;
  onConsentChange: (consent: boolean) => void;
  checkResult: ReferenceCheckResponse | null;
  isChecking: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onReferenceReady,
  referenceAudioUrl,
  consent,
  onConsentChange,
  checkResult,
  isChecking,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const file = new File([audioBlob], 'recorded_reference.wav', { type: 'audio/wav' });
        onReferenceReady(file);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordSeconds(0);
      timerRef.current = window.setInterval(() => {
        setRecordSeconds((prev) => prev + 1);
      }, 1000);
    } catch {
      alert('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onReferenceReady(file);
    }
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: 'var(--space-2)',
        height: '100%',
      }}
    >
      {/* Column 1: Audio Capture (Upload or Record) */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-1)',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
          <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            Provide reference audio
          </span>
          <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)' }}>
            Upload or record 5–20 seconds of clear speech with minimal background noise.
          </p>

          <div style={{ display: 'flex', gap: 'var(--space-1)', marginTop: 'var(--space-0-5)' }}>
            <Button
              size="sm"
              variant={isRecording ? 'danger' : 'secondary'}
              onClick={isRecording ? stopRecording : startRecording}
            >
              {isRecording ? `⏹ Stop (${recordSeconds}s)` : '🎙 Record microphone'}
            </Button>

            <Button
              size="sm"
              variant="secondary"
              onClick={() => fileInputRef.current?.click()}
            >
              📁 Upload audio
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.wav,.mp3,.m4a"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>
        </div>

        {/* Consent Checkbox */}
        <label
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 'var(--space-1)',
            cursor: 'pointer',
            fontSize: 'var(--font-sm)',
            color: 'var(--color-text-primary)',
            padding: 'var(--space-1)',
            backgroundColor: 'var(--color-surface-muted)',
            borderRadius: 'var(--radius-control)',
            border: '1px solid var(--color-border)',
          }}
        >
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => onConsentChange(e.target.checked)}
            style={{ accentColor: 'var(--color-accent)', width: '16px', height: '16px', marginTop: '2px' }}
          />
          <span>I confirm I own or have permission to clone this voice.</span>
        </label>
      </div>

      {/* Column 2: Reference Player & Quality Check */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-1)',
          justifyContent: 'space-between',
        }}
      >
        {referenceAudioUrl ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
            <AudioPlayer src={referenceAudioUrl} title="Reference Preview" compact />

            {/* Quality Score & Diagnostic Advice */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-0-5)',
                padding: 'var(--space-1)',
                borderRadius: 'var(--radius-control)',
                backgroundColor: checkResult?.is_usable ? 'var(--color-success-bg)' : 'var(--color-surface-muted)',
                border: `1px solid ${checkResult?.is_usable ? 'var(--color-success-border)' : 'var(--color-border)'}`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600 }}>
                  Quality score: {checkResult ? `${checkResult.score}/100` : isChecking ? 'Evaluating...' : '—'}
                </span>
                {checkResult && (
                  <span
                    style={{
                      fontSize: 'var(--font-sm)',
                      fontWeight: 600,
                      color: checkResult.is_usable ? 'var(--color-success)' : 'var(--color-error)',
                    }}
                  >
                    {checkResult.is_usable ? 'Usable' : 'Attention required'}
                  </span>
                )}
              </div>
              <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)' }}>
                {checkResult?.advice?.[0] || 'Ready for reference safety evaluation.'}
              </span>
            </div>
          </div>
        ) : (
          <div
            style={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'var(--color-surface-muted)',
              border: '1px dashed var(--color-border)',
              borderRadius: 'var(--radius-control)',
              padding: 'var(--space-2)',
              textAlign: 'center',
            }}
          >
            <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
              No reference audio selected yet. Record or upload an audio sample to continue.
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
