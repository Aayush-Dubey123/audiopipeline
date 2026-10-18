import React, { useState, useRef, useEffect } from 'react';
import { Upload, Mic, RefreshCw, AlertCircle, CheckCircle2, Square, Volume2 } from 'lucide-react';
import { api, ApiError } from '../../lib/api';
import type { Voice, JobStatusResponse } from '../../lib/types';
import { AudioPlayer } from '../common/AudioPlayer';

interface VCStudioProps {
  voices: Voice[];
  onNavigateBack: () => void;
}

export const VCStudio: React.FC<VCStudioProps> = ({
  voices,
  onNavigateBack,
}) => {
  // Step 1: Input Audio State
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceAudioUrl, setSourceAudioUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  // Step 2: Voice Out Choice State
  const rvcVoices = voices.filter((v) => v.model_type === 'rvc_model');
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>(
    rvcVoices.length > 0 ? rvcVoices[0].id : ''
  );
  const [pitchShift, setPitchShift] = useState<number>(0);

  // Step 3: Result & Polling State
  const [currentJob, setCurrentJob] = useState<JobStatusResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      if (sourceAudioUrl) URL.revokeObjectURL(sourceAudioUrl);
    };
  }, []);

  useEffect(() => {
    if (rvcVoices.length > 0 && !rvcVoices.some(v => v.id === selectedVoiceId)) {
      setSelectedVoiceId(rvcVoices[0].id);
    }
  }, [rvcVoices]);

  // Handle File Selection
  const handleFileChange = (file: File) => {
    setSourceFile(file);
    if (sourceAudioUrl) URL.revokeObjectURL(sourceAudioUrl);
    setSourceAudioUrl(URL.createObjectURL(file));
  };

  // Microphone recording
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
        const recordedFile = new File([audioBlob], 'input_speech.wav', { type: 'audio/wav' });
        handleFileChange(recordedFile);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 30) {
            stopRecording();
            return 30;
          }
          return prev + 1;
        });
      }, 1000);
    } catch {
      setErrorMessage('Microphone access denied or unsupported.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  // Submit Voice Conversion Job
  const handleConvert = async () => {
    if (!sourceFile) {
      setErrorMessage('Please upload or record an audio file to convert.');
      return;
    }
    if (!selectedVoiceId) {
      setErrorMessage('No target voice available for voice conversion.');
      return;
    }

    setErrorMessage(null);
    setIsSubmitting(true);

    try {
      const jobCreated = await api.createVCJob({
        file: sourceFile,
        voiceId: selectedVoiceId,
        pitchShift,
      });

      pollJobStatus(jobCreated.job_id);
    } catch (err) {
      setIsSubmitting(false);
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to queue voice conversion job.');
      }
    }
  };

  const pollJobStatus = (jobId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    const checkStatus = async () => {
      try {
        const job = await api.getJobStatus(jobId);
        setCurrentJob(job);

        if (job.status === 'completed') {
          setIsSubmitting(false);
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        } else if (job.status === 'failed') {
          setIsSubmitting(false);
          setErrorMessage(job.error_message || 'Voice conversion failed during processing.');
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        }
      } catch {
        // Polling retry
      }
    };

    checkStatus();
    pollIntervalRef.current = window.setInterval(checkStatus, 1500);
  };

  const isAudioInDone = Boolean(sourceFile);
  const isResultDone = currentJob?.status === 'completed';

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
      {/* Breadcrumb matching Mockup */}
      <div style={{ marginBottom: '12px' }}>
        <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
          <button
            onClick={onNavigateBack}
            style={{
              border: 'none',
              background: 'transparent',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: 0,
              fontFamily: 'inherit',
              fontSize: '0.875rem',
            }}
          >
            Create
          </button>
          {' / Voice to voice'}
        </span>
      </div>

      {/* Main Heading */}
      <h1 style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '20px', color: 'var(--text-primary)' }}>
        Voice to voice
      </h1>

      {/* Stepper Pills matching Mockup */}
      <div className="stepper-container">
        <span className={`step-pill ${isAudioInDone ? 'done' : 'active'}`}>
          {isAudioInDone ? <CheckCircle2 size={15} /> : null}
          <span>{isAudioInDone ? '1 Audio in done' : '1 Audio in'}</span>
        </span>
        <span className={`step-pill ${isResultDone ? 'done' : 'active'}`}>
          <span>2 Voice out</span>
        </span>
        <span className={`step-pill ${isResultDone ? 'done' : 'inactive'}`}>
          <span>3 Result</span>
        </span>
      </div>

      {/* 2-Column Responsive Layout */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 1fr)',
        gap: '32px',
        alignItems: 'start',
      }}>
        {/* Left Column: Step 1 & Step 2 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Card 1: Audio in */}
          <div className="studio-card" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
              1 · Audio in
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Upload or record speech to convert. Best results with clean speech from one person without background music.
            </p>

            <div style={{
              border: '2px dashed var(--border-medium)',
              borderRadius: 'var(--radius-md)',
              padding: '24px',
              textAlign: 'center',
              backgroundColor: 'var(--bg-primary)',
            }}>
              <input
                type="file"
                accept="audio/wav,audio/mp3,audio/mpeg,audio/ogg,audio/flac"
                id="vc-upload-file"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileChange(e.target.files[0]);
                  }
                }}
              />

              {sourceFile ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <CheckCircle2 size={18} color="var(--accent-green-text)" />
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{sourceFile.name}</span>
                    <button
                      onClick={() => {
                        setSourceFile(null);
                        setSourceAudioUrl(null);
                      }}
                      style={{ border: 'none', background: 'transparent', color: '#dc2626', cursor: 'pointer', fontSize: '0.8125rem' }}
                    >
                      Remove
                    </button>
                  </div>

                  {sourceAudioUrl && (
                    <audio controls src={sourceAudioUrl} style={{ width: '100%', maxWidth: '380px', height: '36px' }} />
                  )}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                  <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
                    <label
                      htmlFor="vc-upload-file"
                      className="btn-secondary"
                      style={{ cursor: 'pointer' }}
                    >
                      <Upload size={16} />
                      <span>Choose Audio File</span>
                    </label>

                    <button
                      type="button"
                      onClick={isRecording ? stopRecording : startRecording}
                      className="btn-secondary"
                      style={{
                        backgroundColor: isRecording ? '#fee2e2' : '#ffffff',
                        color: isRecording ? '#dc2626' : 'inherit',
                      }}
                    >
                      {isRecording ? <Square size={16} /> : <Mic size={16} />}
                      <span>{isRecording ? `Recording (${recordingSeconds}s)` : 'Record Mic'}</span>
                    </button>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Supported formats: WAV, MP3, FLAC (under 30 seconds)
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Card 2: Voice out */}
          <div className="studio-card" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
              2 · Voice out
            </h2>

            {/* Clear Message when no RVC target voices exist */}
            {rvcVoices.length === 0 ? (
              <div style={{
                backgroundColor: 'var(--bg-tertiary)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                padding: '20px',
                textAlign: 'left',
              }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                  <span className="status-pill status-needs-voices" style={{ marginTop: '2px' }}>
                    Needs more voices
                  </span>
                  <div>
                    <h4 style={{ fontSize: '0.9375rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                      No trained target voices available
                    </h4>
                    <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      Voice-to-voice conversion requires an RVC target voice model checkpoint.
                      Currently, verified library voices are optimized for text-to-speech. More voice conversion models will appear here as they are added and verified.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {rvcVoices.map((v) => {
                  const isSelected = selectedVoiceId === v.id;
                  return (
                    <div
                      key={v.id}
                      onClick={() => setSelectedVoiceId(v.id)}
                      style={{
                        border: `1.5px solid ${isSelected ? 'var(--accent-green)' : 'var(--border-subtle)'}`,
                        backgroundColor: isSelected ? 'var(--accent-green-light)' : '#ffffff',
                        borderRadius: 'var(--radius-md)',
                        padding: '16px 18px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        cursor: 'pointer',
                      }}
                    >
                      <div>
                        <span style={{ fontWeight: 700, fontSize: '0.9375rem' }}>{v.name}</span>
                        <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          {v.description}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Pitch Shift Slider */}
            <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                <span>Pitch shift:</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>
                  {pitchShift > 0 ? `+${pitchShift}` : pitchShift} semitones
                </span>
              </div>
              <input
                type="range"
                min={-12}
                max={12}
                step={1}
                value={pitchShift}
                disabled={rvcVoices.length === 0}
                onChange={(e) => setPitchShift(parseInt(e.target.value, 10))}
                style={{ width: '100%', accentColor: 'var(--accent-green)', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                <span>-12 (Male to Female deep)</span>
                <span>0 (Default)</span>
                <span>+12 (Female pitch)</span>
              </div>
            </div>

            {/* Error Message */}
            {errorMessage && (
              <div style={{
                marginTop: '16px',
                padding: '12px 14px',
                backgroundColor: '#fef2f2',
                border: '1px solid #fee2e2',
                borderRadius: 'var(--radius-md)',
                color: '#991b1b',
                fontSize: '0.875rem',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                <AlertCircle size={16} />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Action Button */}
            <div style={{ marginTop: '24px' }}>
              <button
                onClick={handleConvert}
                disabled={isSubmitting || !sourceFile || rvcVoices.length === 0}
                className="btn-primary"
                style={{ width: '100%', padding: '12px 20px', fontSize: '1rem' }}
              >
                {isSubmitting ? (
                  <>
                    <RefreshCw size={18} className="pulse-dot" />
                    <span>Converting voice with RVC...</span>
                  </>
                ) : (
                  <span>Convert voice</span>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Step 3 (Result) */}
        <div>
          <div className="studio-card" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
              3 · Result
            </h2>

            {isSubmitting ? (
              <div style={{
                backgroundColor: '#111815',
                borderRadius: 'var(--radius-lg)',
                padding: '48px 24px',
                textAlign: 'center',
                color: '#ffffff',
                border: '1px solid #23312a',
              }}>
                <RefreshCw size={28} className="pulse-dot" style={{ margin: '0 auto 16px auto', display: 'block', color: '#2dd4bf' }} />
                <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '6px' }}>Converting Voice...</h4>
                <p style={{ fontSize: '0.8125rem', color: '#a3b8b0' }}>
                  {currentJob?.progress_message || 'Extracting RMVPE pitch and synthesizing with RVC...'}
                </p>
              </div>
            ) : currentJob?.status === 'completed' && currentJob.output_url ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <AudioPlayer
                  src={currentJob.output_url}
                  comparisonSrc={sourceAudioUrl || undefined}
                  comparisonLabel="Input Audio"
                  title="Converted Voice"
                  subtitle={`Target: ${selectedVoiceId} • Pitch: ${pitchShift > 0 ? `+${pitchShift}` : pitchShift}`}
                  aiAttribution={true}
                />
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  Results are deleted automatically after the retention period. Do not use voices you do not have permission for.
                </p>
              </div>
            ) : (
              <div style={{
                backgroundColor: 'var(--bg-primary)',
                border: '1px dashed var(--border-medium)',
                borderRadius: 'var(--radius-lg)',
                padding: '48px 24px',
                textAlign: 'center',
                color: 'var(--text-muted)',
              }}>
                <Volume2 size={32} style={{ margin: '0 auto 12px auto', opacity: 0.5 }} />
                <h4 style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  No output yet
                </h4>
                <p style={{ fontSize: '0.8125rem' }}>
                  Your converted audio will appear here after clicking <strong>Convert voice</strong>.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
