import React, { useState, useRef } from 'react';
import {
  Mic,
  Square,
  Upload,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Sparkles,
  Loader2,
  Info,
} from 'lucide-react';
import type { Voice, JobStatusResponse } from '../../lib/types';
import { api, ApiError } from '../../lib/api';
import { AudioPlayer } from '../common/AudioPlayer';

interface VCStudioProps {
  voices: Voice[];
  onJobStarted: (jobId: string) => void;
}

export const VCStudio: React.FC<VCStudioProps> = ({ voices, onJobStarted }) => {
  // Target voice selection (only rvc_model or available models)
  const rvcVoices = voices.filter((v) => v.model_type === 'rvc_model');
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>(
    rvcVoices[0]?.id || voices[0]?.id || 'my_voice'
  );

  // Audio input states
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceBlobUrl, setSourceBlobUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  // Pitch shift in semitones (-12 to +12)
  const [pitchShift, setPitchShift] = useState<number>(0);

  // Generation & Status states
  const [isConverting, setIsConverting] = useState(false);
  const [currentJob, setCurrentJob] = useState<JobStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Recorder refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const timerIntervalRef = useRef<number | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Recording controls
  const startRecording = async () => {
    try {
      setErrorMessage(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const file = new File([audioBlob], 'mic_source.wav', { type: 'audio/wav' });
        setSourceFile(file);
        setSourceBlobUrl(URL.createObjectURL(audioBlob));
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(250);
      setIsRecording(true);
      setRecordingSeconds(0);
      timerIntervalRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch {
      setErrorMessage('Microphone access denied. Please upload an audio file instead.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSourceFile(file);
      setSourceBlobUrl(URL.createObjectURL(file));
      setErrorMessage(null);
    }
  };

  const handleConvert = async () => {
    if (!sourceFile) {
      setErrorMessage('Please upload or record source audio to convert.');
      return;
    }

    setIsConverting(true);
    setErrorMessage(null);
    setCurrentJob(null);

    try {
      const created = await api.createVCJob({
        voiceId: selectedVoiceId,
        file: sourceFile,
        pitchShift,
      });

      onJobStarted(created.job_id);

      const finalStatus = await api.pollJob(
        created.job_id,
        (progress) => setCurrentJob(progress),
        1000,
        120000
      );

      setCurrentJob(finalStatus);
      if (finalStatus.status === 'failed') {
        setErrorMessage(finalStatus.error_message || 'Voice conversion failed.');
      }
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : 'Voice conversion job failed.';
      setErrorMessage(msg);
    } finally {
      setIsConverting(false);
    }
  };

  const selectedVoice = voices.find((v) => v.id === selectedVoiceId);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Headline */}
      <div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }} className="gradient-text">
          Voice-to-Voice Conversion Studio
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          Convert existing spoken speech into a target voice character using high-accuracy pitch extraction (RMVPE).
        </p>
      </div>

      <div className="grid-cols-2">
        {/* Left Column: Source Audio & RVC Controls */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Target Voice Selection */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>
              Target Conversion Voice
            </label>

            {voices.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Loading voice models...</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {voices.map((voice) => (
                  <div
                    key={voice.id}
                    onClick={() => setSelectedVoiceId(voice.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '12px 16px',
                      borderRadius: '12px',
                      background: selectedVoiceId === voice.id ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${selectedVoiceId === voice.id ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                      cursor: 'pointer',
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>{voice.name}</span>
                        <span className={`badge ${voice.model_type === 'rvc_model' ? 'badge-cyan' : 'badge-violet'}`} style={{ fontSize: '0.65rem' }}>
                          {voice.model_type === 'rvc_model' ? 'RVC Model' : 'OpenVoice Ref'}
                        </span>
                        {voice.status === 'experimental' && (
                          <span className="badge badge-warning" style={{ fontSize: '0.65rem' }}>Local Only</span>
                        )}
                      </div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                        {voice.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Source Audio Upload / Record Panel */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>
              Source Audio Recording
            </label>

            <div style={{
              border: '2px dashed var(--border-medium)',
              borderRadius: '14px',
              padding: '20px',
              textAlign: 'center',
              backgroundColor: 'rgba(10, 12, 18, 0.5)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', marginBottom: '14px' }}>
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`btn ${isRecording ? 'btn-danger' : 'btn-secondary'}`}
                >
                  {isRecording ? <Square size={16} /> : <Mic size={16} />}
                  <span>{isRecording ? `Stop Recording (${recordingSeconds}s)` : 'Record Mic Audio'}</span>
                </button>

                <label className="btn btn-secondary" style={{ cursor: 'pointer' }}>
                  <Upload size={16} />
                  <span>Upload Audio File</span>
                  <input type="file" accept="audio/*,.wav,.mp3,.m4a,.flac" onChange={handleFileUpload} style={{ display: 'none' }} />
                </label>
              </div>

              {sourceFile ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#22d3ee', fontSize: '0.85rem' }}>
                  <CheckCircle2 size={16} />
                  <span>Source: <strong>{sourceFile.name}</strong> ({Math.round(sourceFile.size / 1024)} KB)</span>
                </div>
              ) : (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Upload or record 2 to 60 seconds of clear speech to convert.
                </p>
              )}

              {sourceBlobUrl && (
                <div style={{ marginTop: '12px' }}>
                  <audio src={sourceBlobUrl} controls style={{ height: '32px', width: '100%', maxWidth: '300px' }} />
                </div>
              )}
            </div>
          </div>

          {/* Pitch Shift Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '0.825rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sliders size={14} /> Pitch Shift (Semitones)
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.825rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan-light)' }}>
                  {pitchShift > 0 ? `+${pitchShift}` : pitchShift} st
                </span>
                {pitchShift !== 0 && (
                  <button onClick={() => setPitchShift(0)} className="btn btn-secondary btn-sm" style={{ padding: '2px 6px', fontSize: '0.7rem' }}>
                    Reset
                  </button>
                )}
              </div>
            </div>
            <input
              type="range"
              min="-12"
              max="12"
              step="1"
              value={pitchShift}
              onChange={(e) => setPitchShift(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              <span>-12 (Deeper / Male)</span>
              <span>0 (Original)</span>
              <span>+12 (Higher / Female)</span>
            </div>
          </div>

          {/* Error Message */}
          {errorMessage && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '12px 16px',
              borderRadius: '10px',
              backgroundColor: 'var(--danger-bg)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#f87171',
              fontSize: '0.875rem',
            }}>
              <AlertCircle size={18} />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            onClick={handleConvert}
            disabled={isConverting || !sourceFile}
            className="btn btn-accent"
            style={{ width: '100%', padding: '14px', fontSize: '1rem' }}
          >
            {isConverting ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                <span>Converting Audio...</span>
              </>
            ) : (
              <>
                <Sparkles size={18} />
                <span>Convert Voice</span>
              </>
            )}
          </button>
        </div>

        {/* Right Column: Output & A/B Comparison */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {isConverting && (
            <div className="glass-panel animate-pulse-glow" style={{ padding: '24px', textAlign: 'center' }}>
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
                <Loader2 size={36} color="var(--accent-cyan)" className="animate-spin" />
              </div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '6px' }}>
                Running Voice Conversion Pipeline
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                {currentJob?.progress_message || 'Extracting RMVPE pitch contours...'}
              </p>
            </div>
          )}

          {currentJob && currentJob.status === 'completed' && currentJob.output_url ? (
            <AudioPlayer
              src={currentJob.output_url}
              comparisonSrc={sourceBlobUrl || undefined}
              title={`Converted: ${selectedVoice?.name || 'Target Voice'}`}
              subtitle="Converted speech using RMVPE neural pitch tracking"
              wavDownloadUrl={currentJob.output_url}
              mp3DownloadUrl={`${currentJob.output_url}?format=mp3`}
              aiAttribution={true}
            />
          ) : !isConverting ? (
            <div className="glass-panel" style={{ padding: '28px', color: 'var(--text-secondary)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
                <Info size={20} color="var(--accent-cyan)" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  How RVC Conversion Works
                </h3>
              </div>
              <ul style={{ paddingLeft: '20px', fontSize: '0.875rem', lineHeight: 1.8, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <li>
                  <strong>Neural Pitch Extraction:</strong> Extracts fundamental frequency ($F_0$) using a pre-trained RMVPE deep learning network.
                </li>
                <li>
                  <strong>A/B Comparison:</strong> Compare the source recording with the synthesized voice track side by side using the player.
                </li>
                <li>
                  <strong>Pitch Adjustment:</strong> If converting male-to-female speech, shift +12 semitones. For female-to-male speech, shift -12 semitones.
                </li>
                <li>
                  <strong>Mutual GPU Unloading:</strong> Heavy RVC and OpenVoice models are automatically unloaded when not in use to preserve VRAM below 2GB.
                </li>
              </ul>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
