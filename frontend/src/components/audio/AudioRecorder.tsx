import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { ReferenceCheckResponse } from '../../lib/types';
import { Button } from '../common/Button';
import { AudioPlayer } from './AudioPlayer';

export interface AudioRecorderProps {
  onReferenceReady: (file: File) => void;
  referenceAudioUrl: string | null;
  referenceFile?: File | null;
  consent: boolean;
  onConsentChange: (consent: boolean) => void;
  checkResult: ReferenceCheckResponse | null;
  isChecking: boolean;
  checkError?: string | null;
  onRetryCheck?: () => void;
  onClearReference?: () => void;
}

/**
 * Encode an AudioBuffer into standard 16-bit PCM RIFF WAV format.
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = Math.min(buffer.numberOfChannels, 2);
  const sampleRate = buffer.sampleRate;
  const numSamples = buffer.length;
  const byteRate = sampleRate * numChannels * 2;
  const blockAlign = numChannels * 2;
  const bufferLength = 44 + numSamples * blockAlign;

  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // RIFF Chunk Descriptor
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * blockAlign, true);
  writeString(8, 'WAVE');

  // fmt subchunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // BitsPerSample (16-bit)

  // data subchunk
  writeString(36, 'data');
  view.setUint32(40, numSamples * blockAlign, true);

  // Write interleaved PCM samples
  const channels: Float32Array[] = [];
  for (let c = 0; c < numChannels; c++) {
    channels.push(buffer.getChannelData(c));
  }

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let c = 0; c < numChannels; c++) {
      const sample = Math.max(-1, Math.min(1, channels[c][i]));
      const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, int16, true);
      offset += 2;
    }
  }

  return new Blob([view], { type: 'audio/wav' });
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onReferenceReady,
  referenceAudioUrl,
  referenceFile,
  consent,
  onConsentChange,
  checkResult,
  isChecking,
  checkError,
  onRetryCheck,
  onClearReference,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [recorderError, setRecorderError] = useState<string | null>(null);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Stop stream tracks
  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  // Clean up recording tracks on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      stopStream();
    };
  }, [stopStream]);

  // Find best supported MIME type
  const getPreferredMimeType = (): string => {
    if (typeof MediaRecorder === 'undefined') return '';
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
      'audio/wav',
    ];
    for (const c of candidates) {
      if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) {
        return c;
      }
    }
    return '';
  };

  const startRecording = async () => {
    setRecorderError(null);

    if (!navigator?.mediaDevices?.getUserMedia) {
      setRecorderError('Microphone recording is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;
      audioChunksRef.current = [];

      const preferredMime = getPreferredMimeType();
      const options: MediaRecorderOptions = preferredMime ? { mimeType: preferredMime } : {};
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        stopStream();
        setIsRecording(false);
        if (timerRef.current) clearInterval(timerRef.current);

        const rawMime = recorder.mimeType || preferredMime || 'audio/webm';
        const rawBlob = new Blob(audioChunksRef.current, { type: rawMime });

        if (rawBlob.size === 0) {
          setRecorderError('Recording produced no audio data. Please try again.');
          return;
        }

        setIsProcessingAudio(true);
        try {
          // Attempt transcode to genuine 16-bit PCM WAV using AudioContext
          const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
          if (AudioCtxClass) {
            const ctx = new AudioCtxClass();
            const arrayBuf = await rawBlob.arrayBuffer();
            const audioBuffer = await ctx.decodeAudioData(arrayBuf);
            const wavBlob = audioBufferToWav(audioBuffer);
            await ctx.close();
            const file = new File([wavBlob], 'recorded_reference.wav', { type: 'audio/wav' });
            onReferenceReady(file);
          } else {
            const ext = rawMime.includes('ogg') ? 'ogg' : rawMime.includes('mp4') ? 'm4a' : 'webm';
            const file = new File([rawBlob], `recorded_reference.${ext}`, { type: rawMime });
            onReferenceReady(file);
          }
        } catch {
          // Fallback: use raw blob with its actual mime type
          const ext = rawMime.includes('ogg') ? 'ogg' : rawMime.includes('mp4') ? 'm4a' : 'webm';
          const file = new File([rawBlob], `recorded_reference.${ext}`, { type: rawMime });
          onReferenceReady(file);
        } finally {
          setIsProcessingAudio(false);
        }
      };

      recorder.start(250); // collect chunks every 250ms
      setIsRecording(true);
      setRecordSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      stopStream();
      setIsRecording(false);
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setRecorderError('Microphone permission was denied. Please allow microphone access in your browser address bar.');
      } else if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
        setRecorderError('No microphone detected. Please check your audio input hardware.');
      } else if (err?.name === 'NotReadableError' || err?.name === 'TrackStartError') {
        setRecorderError('Microphone is in use by another app or hardware error occurred.');
      } else {
        setRecorderError(err?.message || 'Failed to start microphone recording.');
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  const cancelRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.onstop = null; // Detach onstop handler so nothing is emitted
      mediaRecorderRef.current.stop();
    }
    stopStream();
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
    audioChunksRef.current = [];
    setRecordSeconds(0);
    setRecorderError(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setRecorderError(null);
      onReferenceReady(file);
    }
  };

  const formatSecs = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: 'var(--space-2)',
        height: '100%',
      }}
    >
      {/* Column 1: Audio Capture Card */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-2)',
          justifyContent: 'space-between',
          backgroundColor: 'var(--color-surface)',
          padding: 'var(--space-2)',
          borderRadius: 'var(--radius-control)',
          border: '1px solid var(--color-border)',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              Reference Audio Source
            </span>
            {referenceFile && (
              <span
                style={{
                  fontSize: '11px',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  backgroundColor: 'var(--color-neutral-pill-bg)',
                  color: 'var(--color-neutral-pill-text)',
                  fontWeight: 600,
                }}
              >
                {referenceFile.name.endsWith('.wav') ? 'WAV' : referenceFile.type || 'Audio'}
              </span>
            )}
          </div>

          <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
            Record or upload 5–25 seconds of clear, continuous speech with minimal background noise and steady volume.
          </p>

          {/* Error Banner */}
          {recorderError && (
            <div
              style={{
                fontSize: 'var(--font-sm)',
                color: 'var(--color-error)',
                backgroundColor: 'var(--color-error-bg)',
                padding: '8px 10px',
                borderRadius: 'var(--radius-control)',
                border: '1px solid var(--color-error-border)',
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                gap: '8px',
              }}
            >
              <span>{recorderError}</span>
              <button
                type="button"
                onClick={() => setRecorderError(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--color-error)',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '14px',
                  lineHeight: 1,
                }}
              >
                ×
              </button>
            </div>
          )}

          {/* Recording / In-Progress UI */}
          {isRecording ? (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-1)',
                padding: 'var(--space-2)',
                backgroundColor: 'var(--color-surface-muted)',
                borderRadius: 'var(--radius-control)',
                border: '1px solid var(--color-error-border)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    style={{
                      width: '10px',
                      height: '10px',
                      borderRadius: '50%',
                      backgroundColor: 'var(--color-error)',
                      display: 'inline-block',
                      animation: 'pulse 1.2s infinite ease-in-out',
                    }}
                  />
                  <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-error)' }}>
                    Recording... {formatSecs(recordSeconds)}
                  </span>
                </div>
                <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
                  Min 5s • Best 10-20s
                </span>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-1)', marginTop: '4px' }}>
                <Button
                  size="sm"
                  variant="danger"
                  onClick={stopRecording}
                  style={{ flex: 1 }}
                >
                  ⏹ Stop & Use Recording
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={cancelRecording}
                >
                  ✖ Cancel
                </Button>
              </div>
            </div>
          ) : isProcessingAudio ? (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: 'var(--space-2)',
                backgroundColor: 'var(--color-surface-muted)',
                borderRadius: 'var(--radius-control)',
                fontSize: 'var(--font-sm)',
                color: 'var(--color-text-secondary)',
              }}
            >
              <span>Processing audio encoding into standard WAV format...</span>
            </div>
          ) : (
            /* Action Buttons (Record / Upload) */
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-1)', marginTop: '4px' }}>
              <Button
                size="sm"
                variant={referenceFile ? 'secondary' : 'primary'}
                onClick={startRecording}
                style={{ flex: 1, minWidth: '130px' }}
              >
                🎙 {referenceFile ? 'Re-record mic' : 'Record microphone'}
              </Button>

              <Button
                size="sm"
                variant="secondary"
                onClick={() => fileInputRef.current?.click()}
                style={{ flex: 1, minWidth: '130px' }}
              >
                📁 {referenceFile ? 'Replace audio file' : 'Upload audio file'}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </div>
          )}

          {/* Reference File Metadata (if uploaded or recorded) */}
          {referenceFile && !isRecording && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                borderRadius: 'var(--radius-control)',
                backgroundColor: 'var(--color-surface-muted)',
                fontSize: '11px',
                color: 'var(--color-text-secondary)',
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '200px' }}>
                📎 {referenceFile.name} ({(referenceFile.size / 1024).toFixed(0)} KB)
              </span>
              {onClearReference && (
                <button
                  type="button"
                  onClick={onClearReference}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--color-text-muted)',
                    cursor: 'pointer',
                    fontSize: '11px',
                    textDecoration: 'underline',
                  }}
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>

        {/* Legal Consent Confirmation */}
        <label
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 'var(--space-1)',
            cursor: 'pointer',
            fontSize: 'var(--font-sm)',
            color: 'var(--color-text-primary)',
            padding: 'var(--space-1) var(--space-2)',
            backgroundColor: consent ? 'var(--color-success-bg)' : 'var(--color-surface-muted)',
            borderRadius: 'var(--radius-control)',
            border: `1px solid ${consent ? 'var(--color-success-border)' : 'var(--color-border)'}`,
            transition: 'background-color 150ms ease, border-color 150ms ease',
          }}
        >
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => onConsentChange(e.target.checked)}
            style={{
              accentColor: 'var(--color-accent)',
              width: '16px',
              height: '16px',
              marginTop: '2px',
              flexShrink: 0,
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <span style={{ fontWeight: 600 }}>Consent & Authorization</span>
            <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)', lineHeight: 1.3 }}>
              I confirm that I own this voice or possess explicit written authorization from the speaker for voice cloning synthesis.
            </span>
          </div>
        </label>
      </div>

      {/* Column 2: Reference Preview & Acoustic Quality Feedback Card */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-1)',
          justifyContent: 'space-between',
          backgroundColor: 'var(--color-surface)',
          padding: 'var(--space-2)',
          borderRadius: 'var(--radius-control)',
          border: '1px solid var(--color-border)',
        }}
      >
        {referenceAudioUrl ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
            <AudioPlayer
              src={referenceAudioUrl}
              title="Reference Audio Clip Preview"
              compact={false}
            />

            {/* Quality Score & Diagnostic Assessment */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                padding: 'var(--space-1) var(--space-2)',
                borderRadius: 'var(--radius-control)',
                backgroundColor: checkResult?.is_usable
                  ? 'var(--color-success-bg)'
                  : checkError
                  ? 'var(--color-error-bg)'
                  : 'var(--color-surface-muted)',
                border: `1px solid ${
                  checkResult?.is_usable
                    ? 'var(--color-success-border)'
                    : checkError
                    ? 'var(--color-error-border)'
                    : 'var(--color-border)'
                }`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600 }}>
                  Acoustic Quality Score:{' '}
                  {isChecking ? (
                    <span style={{ color: 'var(--color-text-muted)' }}>Evaluating...</span>
                  ) : checkResult ? (
                    <span>{checkResult.score}/100</span>
                  ) : (
                    <span style={{ color: 'var(--color-text-muted)' }}>Unvalidated</span>
                  )}
                </span>

                {isChecking ? (
                  <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>Analyzing waveform</span>
                ) : checkResult ? (
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: '10px',
                      backgroundColor: checkResult.is_usable ? 'var(--color-success)' : 'var(--color-error)',
                      color: '#ffffff',
                    }}
                  >
                    {checkResult.is_usable ? '✓ Usable for Cloning' : '⚠ Attention Required'}
                  </span>
                ) : checkError ? (
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: '10px',
                      backgroundColor: 'var(--color-error)',
                      color: '#ffffff',
                    }}
                  >
                    Check Failed
                  </span>
                ) : null}
              </div>

              {/* Metrics Detail (if available from genuine backend validation) */}
              {checkResult?.metrics && (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
                    gap: '4px',
                    fontSize: '11px',
                    color: 'var(--color-text-secondary)',
                    padding: '4px 0',
                    borderTop: '1px solid rgba(0,0,0,0.06)',
                    borderBottom: '1px solid rgba(0,0,0,0.06)',
                  }}
                >
                  <span>Speech: <strong>{checkResult.metrics.speech_length_s}s</strong></span>
                  <span>SNR: <strong>{checkResult.metrics.snr_db} dB</strong></span>
                  <span>Loudness: <strong>{checkResult.metrics.loudness_dbfs} dBFS</strong></span>
                  <span>Clip: <strong>{(checkResult.metrics.clipping_ratio * 100).toFixed(1)}%</strong></span>
                </div>
              )}

              {/* Diagnostic Advice or Error */}
              {checkError ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                  <span style={{ fontSize: '11px', color: 'var(--color-error)' }}>
                    {checkError}
                  </span>
                  {onRetryCheck && (
                    <button
                      type="button"
                      onClick={onRetryCheck}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--color-error)',
                        fontSize: '11px',
                        cursor: 'pointer',
                        textDecoration: 'underline',
                        fontWeight: 600,
                      }}
                    >
                      Retry check
                    </button>
                  )}
                </div>
              ) : (
                <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', lineHeight: 1.35 }}>
                  {checkResult?.advice?.[0] ||
                    (isChecking
                      ? 'Probing speech duration, noise floor SNR, and acoustic clipping...'
                      : 'Audio ready for acoustic validation.')}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div
            style={{
              height: '100%',
              minHeight: '140px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'var(--color-surface-muted)',
              border: '1px dashed var(--color-border)',
              borderRadius: 'var(--radius-control)',
              padding: 'var(--space-2)',
              textAlign: 'center',
              gap: '6px',
            }}
          >
            <span style={{ fontSize: '24px' }}>🎙</span>
            <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              No reference audio selected
            </span>
            <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', maxWidth: '240px' }}>
              Record through your microphone or upload a clip to evaluate voice similarity and preview speech.
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
