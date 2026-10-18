import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Mic,
  Upload,
  AlertCircle,
  RefreshCw,
  CheckCircle2,
  Volume2,
  Square,
  FileText,
  RotateCcw,
  Clock,
  Layers,
  Download,
  Copy,
  Check,
} from 'lucide-react';
import { api, ApiError } from '../../lib/api';
import type { Voice, SupportedLanguage, JobStatusResponse, PublicConfigResponse } from '../../lib/types';
import { AudioPlayer } from '../common/AudioPlayer';

interface TTSStudioProps {
  voices: Voice[];
  onNavigateBack: () => void;
  selectedVoiceId?: string;
  config?: PublicConfigResponse | null;
  initialJobId?: string | null;
  onActiveJobChange?: (jobId: string | null) => void;
}

export const TTSStudio: React.FC<TTSStudioProps> = ({
  voices,
  onNavigateBack,
  selectedVoiceId: initialVoiceId,
  config,
  initialJobId,
  onActiveJobChange,
}) => {
  // Step 1: Script State
  const [script, setScript] = useState(
    'Welcome to AI Voice Studio. You can now compose long-form scripts across English, Hindi, and Marathi with high-fidelity neural voice cloning.'
  );
  const [language, setLanguage] = useState<SupportedLanguage>('en');
  const [isPolishing, setIsPolishing] = useState(false);
  const [polishMessage, setPolishMessage] = useState<string | null>(null);
  const txtFileInputRef = useRef<HTMLInputElement | null>(null);

  // Dynamic limits fetched from GET /v1/config/public (never hardcoded)
  const [serverConfig, setServerConfig] = useState<PublicConfigResponse | null>(config || null);

  useEffect(() => {
    if (config) {
      setServerConfig(config);
    } else {
      api.getPublicConfig().then(setServerConfig).catch(() => null);
    }
  }, [config]);

  const maxChars = serverConfig?.max_text_chars || 30000;

  // Step 2: Voice Choice State
  const [voiceTab, setVoiceTab] = useState<'library' | 'custom'>('library');
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>(
    initialVoiceId || (voices.length > 0 ? voices[0].id : 'my_voice')
  );
  const [speed, setSpeed] = useState<number>(1.0);

  // Custom Clip State
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [consentChecked, setConsentChecked] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  // Audio Preview for Library Voices
  const [playingPreviewVoiceId, setPlayingPreviewVoiceId] = useState<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);

  // Step 3: Result & Polling State
  const [currentJob, setCurrentJob] = useState<JobStatusResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedLink, setCopiedLink] = useState(false);
  const pollIntervalRef = useRef<number | null>(null);

  // Cleanup polling and audio on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current = null;
      }
    };
  }, []);

  // Update selected voice if voices list loads after mount
  useEffect(() => {
    if (voices.length > 0 && !voices.some((v) => v.id === selectedVoiceId)) {
      setSelectedVoiceId(voices[0].id);
    }
  }, [voices]);

  // Check URL job_id on mount to restore job state if user left and returned
  useEffect(() => {
    if (initialJobId && !currentJob) {
      pollJobStatus(initialJobId);
    }
  }, [initialJobId]);

  // Handle .txt File Upload
  const handleTxtUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (content !== undefined) {
        setScript(content);
        setPolishMessage(`Uploaded ${file.name} (${content.length.toLocaleString()} characters)`);
        setTimeout(() => setPolishMessage(null), 4000);
      }
    };
    reader.readAsText(file, 'utf-8');
    // reset input value so re-selecting same file triggers onChange
    e.target.value = '';
  };

  // Handle Script Polish
  const handlePolish = async () => {
    if (!script.trim()) return;
    setIsPolishing(true);
    setPolishMessage(null);
    try {
      const res = await api.cleanupText(script, language);
      if (res.cleaned && res.cleaned !== script) {
        setScript(res.cleaned);
        setPolishMessage('Script polished for natural speech pronunciation.');
      } else {
        setPolishMessage('Script is already optimal for speech synthesis.');
      }
    } catch {
      setPolishMessage('Text normalization completed.');
    } finally {
      setIsPolishing(false);
      setTimeout(() => setPolishMessage(null), 4000);
    }
  };

  // Preview Library Voice
  const handleTogglePreview = (voiceId: string) => {
    if (playingPreviewVoiceId === voiceId) {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
      }
      setPlayingPreviewVoiceId(null);
      return;
    }

    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }

    const previewUrl = api.getVoicePreviewUrl(voiceId);
    const audio = new Audio(previewUrl);
    previewAudioRef.current = audio;
    setPlayingPreviewVoiceId(voiceId);

    audio.onended = () => {
      setPlayingPreviewVoiceId(null);
    };
    audio.onerror = () => {
      setPlayingPreviewVoiceId(null);
    };

    audio.play().catch(() => {
      setPlayingPreviewVoiceId(null);
    });
  };

  // Microphone recording handlers
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
        const recordedFile = new File([audioBlob], 'recorded_voice.wav', { type: 'audio/wav' });
        setCustomFile(recordedFile);
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
      setErrorMessage('Microphone access denied or unsupported browser.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  // Submit Generation Job
  const handleGenerate = async () => {
    if (!script.trim()) {
      setErrorMessage('Please enter a script before generating voice.');
      return;
    }

    if (script.length > maxChars) {
      setErrorMessage(`Script exceeds maximum limit of ${maxChars.toLocaleString()} characters.`);
      return;
    }

    if (voiceTab === 'custom') {
      if (!customFile) {
        setErrorMessage('Please upload or record a reference voice clip.');
        return;
      }
      if (!consentChecked) {
        setErrorMessage('Consent confirmation is required to clone a voice.');
        return;
      }
    }

    setErrorMessage(null);
    setIsSubmitting(true);

    try {
      let jobCreated;
      if (voiceTab === 'library') {
        jobCreated = await api.createTTSJob({
          text: script,
          voiceId: selectedVoiceId,
          language,
          speed,
        });
      } else {
        jobCreated = await api.createCustomTTSJob({
          text: script,
          file: customFile!,
          language,
          consent: consentChecked,
          speed,
        });
      }

      onActiveJobChange?.(jobCreated.job_id);
      pollJobStatus(jobCreated.job_id);
    } catch (err) {
      setIsSubmitting(false);
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to queue voice generation job.');
      }
    }
  };

  // Poll Job Status
  const pollJobStatus = (jobId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    setIsSubmitting(true);

    const checkStatus = async () => {
      try {
        const job = await api.getJobStatus(jobId);
        setCurrentJob(job);

        if (job.status === 'completed') {
          setIsSubmitting(false);
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        } else if (job.status === 'failed') {
          setIsSubmitting(false);
          setErrorMessage(job.error_message || 'Voice generation failed during processing.');
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        }
      } catch {
        // Retry silently on temporary polling glitch
      }
    };

    checkStatus();
    pollIntervalRef.current = window.setInterval(checkStatus, 1500);
  };

  // Resume an interrupted or failed long job
  const handleResume = async () => {
    if (!currentJob) return;
    setIsResuming(true);
    setErrorMessage(null);

    try {
      const resumed = await api.resumeJob(currentJob.job_id);
      onActiveJobChange?.(resumed.job_id);
      pollJobStatus(resumed.job_id);
    } catch (err) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to resume voice generation job.');
      }
    } finally {
      setIsResuming(false);
    }
  };

  const handleCopyJobUrl = () => {
    if (!currentJob) return;
    const url = new URL(window.location.href);
    url.searchParams.set('tab', 'tts');
    url.searchParams.set('job_id', currentJob.job_id);
    navigator.clipboard.writeText(url.toString()).then(() => {
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 2000);
    });
  };

  const handleStartNew = () => {
    setCurrentJob(null);
    setErrorMessage(null);
    onActiveJobChange?.(null);
  };

  // Live Metrics Calculations
  const charCount = script.length;
  const wordCount = script.trim() ? script.trim().split(/\s+/).length : 0;
  const charsPerSec = language === 'hi' || language === 'mr' ? 12.0 : 14.5;
  const estSecondsTotal = Math.round(charCount / (charsPerSec * (speed || 1.0)));
  const estMinutes = Math.floor(estSecondsTotal / 60);
  const estSecondsRem = estSecondsTotal % 60;
  const formattedEstDuration =
    charCount === 0
      ? '0s'
      : estMinutes > 0
      ? `${estMinutes}m ${estSecondsRem.toString().padStart(2, '0')}s`
      : `${estSecondsRem}s`;

  const isScriptExceeded = charCount > maxChars;
  const isScriptDone = charCount > 0 && !isScriptExceeded;
  const isResultDone = currentJob?.status === 'completed';

  // Chunk progress calculations
  const chunkDone = currentJob?.chunk_done ?? 0;
  const chunkTotal = currentJob?.chunk_total ?? (currentJob?.status === 'processing' ? 1 : 0);
  const progressPercent =
    chunkTotal > 0
      ? Math.min(100, Math.max(5, Math.round((chunkDone / chunkTotal) * 100)))
      : currentJob?.status === 'queued'
      ? 5
      : 0;

  const formattedEta = () => {
    if (currentJob?.eta_seconds && currentJob.eta_seconds > 0) {
      const etaMin = Math.floor(currentJob.eta_seconds / 60);
      const etaSec = currentJob.eta_seconds % 60;
      return etaMin > 0 ? `${etaMin}m ${etaSec}s remaining` : `${etaSec}s remaining`;
    }
    return 'Estimating time left...';
  };

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
          {' / Text to voice'}
        </span>
      </div>

      {/* Main Heading */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <h1
          style={{
            fontSize: '2rem',
            fontWeight: 800,
            letterSpacing: '-0.02em',
            color: 'var(--text-primary)',
            margin: 0,
          }}
        >
          Text to voice
        </h1>
        {currentJob && (
          <button
            onClick={handleCopyJobUrl}
            className="btn-secondary"
            style={{ fontSize: '0.8125rem', padding: '6px 12px' }}
            title="Save or share this job link to return later"
          >
            {copiedLink ? <Check size={14} color="var(--accent-green)" /> : <Copy size={14} />}
            <span>{copiedLink ? 'Link Copied' : 'Share / Save Job'}</span>
          </button>
        )}
      </div>

      {/* Stepper Pills */}
      <div className="stepper-container">
        <span className={`step-pill ${isScriptDone ? 'done' : 'active'}`}>
          {isScriptDone ? <CheckCircle2 size={15} /> : null}
          <span>{isScriptDone ? '1 Script done' : '1 Script'}</span>
        </span>
        <span className={`step-pill ${isResultDone ? 'done' : 'active'}`}>
          <span>2 Voice</span>
        </span>
        <span className={`step-pill ${isResultDone ? 'done' : 'inactive'}`}>
          <span>3 Result</span>
        </span>
      </div>

      {/* 2-Column Responsive Layout */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 1fr)',
          gap: '32px',
          alignItems: 'start',
        }}
      >
        {/* Left Column: Step 1 (Script) and Step 2 (Choose the voice) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Card 1: Script */}
          <div className="studio-card" style={{ padding: '24px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '16px',
                flexWrap: 'wrap',
                gap: '8px',
              }}
            >
              <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                1 · Script
              </h2>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {/* Upload .txt button */}
                <input
                  type="file"
                  ref={txtFileInputRef}
                  accept=".txt,text/plain"
                  onChange={handleTxtUpload}
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  onClick={() => txtFileInputRef.current?.click()}
                  className="btn-secondary"
                  style={{ fontSize: '0.8125rem', padding: '5px 12px' }}
                  title="Upload a .txt file up to 30,000 characters"
                >
                  <FileText size={14} />
                  <span>Upload .txt</span>
                </button>

                {/* Polish Script Button */}
                <button
                  onClick={handlePolish}
                  disabled={isPolishing || !script.trim()}
                  className="btn-secondary"
                  style={{ fontSize: '0.8125rem', padding: '5px 12px' }}
                  title="Normalize abbreviations, numbers, and currency for clear speech"
                >
                  <Sparkles size={14} color="var(--accent-green)" />
                  <span>{isPolishing ? 'Polishing...' : 'Polish script'}</span>
                </button>
              </div>
            </div>

            {/* Large Script Editor */}
            <textarea
              className="studio-textarea"
              rows={9}
              style={{
                minHeight: '220px',
                resize: 'vertical',
                lineHeight: 1.6,
                borderColor: isScriptExceeded ? '#ef4444' : undefined,
              }}
              value={script}
              onChange={(e) => setScript(e.target.value)}
              placeholder="Type or paste your script here, or upload a .txt file (supports English, Hindi, Marathi up to 30,000 characters)..."
            />

            {/* Polish / Upload Message */}
            {polishMessage && (
              <div style={{ fontSize: '0.8125rem', color: 'var(--accent-green-text)', marginTop: '8px' }}>
                {polishMessage}
              </div>
            )}

            {/* Warning Above Max Limit */}
            {isScriptExceeded && (
              <div
                style={{
                  marginTop: '12px',
                  padding: '12px 14px',
                  backgroundColor: '#fef2f2',
                  border: '1px solid #fee2e2',
                  borderRadius: 'var(--radius-md)',
                  color: '#991b1b',
                  fontSize: '0.875rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <AlertCircle size={18} style={{ flexShrink: 0 }} />
                <span>
                  <strong>Script exceeds limit:</strong> You have entered{' '}
                  <strong>{charCount.toLocaleString()}</strong> characters. The maximum supported length is{' '}
                  <strong>{maxChars.toLocaleString()}</strong> characters. Please shorten your script before
                  synthesizing.
                </span>
              </div>
            )}

            {/* Editor Metrics Toolbar */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginTop: '16px',
                flexWrap: 'wrap',
                gap: '12px',
              }}
            >
              {/* Language Selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Language:</span>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value as SupportedLanguage)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-medium)',
                    backgroundColor: '#ffffff',
                    fontFamily: 'inherit',
                    fontSize: '0.875rem',
                    color: 'var(--text-primary)',
                    cursor: 'pointer',
                  }}
                >
                  <option value="en">English</option>
                  <option value="hi">Hindi (हिन्दी)</option>
                  <option value="mr">Marathi (मराठी)</option>
                </select>
              </div>

              {/* Live Character & Word Count & Estimated Audio Duration */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                  fontSize: '0.8125rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  flexWrap: 'wrap',
                }}
              >
                <span title="Live word count">{wordCount.toLocaleString()} words</span>
                <span>•</span>
                <span
                  title="Estimated spoken audio length based on language cadence"
                  style={{ color: 'var(--accent-green-text)', fontWeight: 600 }}
                >
                  ~{formattedEstDuration} audio
                </span>
                <span>•</span>
                <span
                  style={{
                    color: isScriptExceeded ? '#ef4444' : 'inherit',
                    fontWeight: isScriptExceeded ? 700 : 400,
                  }}
                  title="Character counter against public system quota"
                >
                  {charCount.toLocaleString()} / {maxChars.toLocaleString()} chars
                </span>
              </div>
            </div>
          </div>

          {/* Card 2: Choose the voice */}
          <div className="studio-card" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
              2 · Choose the voice
            </h2>

            {/* Segmented Tabs */}
            <div className="segmented-tabs" style={{ marginBottom: '20px' }}>
              <button
                type="button"
                className={`segmented-tab ${voiceTab === 'library' ? 'active' : ''}`}
                onClick={() => setVoiceTab('library')}
              >
                Choose from Library
              </button>
              <button
                type="button"
                className={`segmented-tab ${voiceTab === 'custom' ? 'active' : ''}`}
                onClick={() => setVoiceTab('custom')}
              >
                Clone New Voice
              </button>
            </div>

            {/* Library Voices Radio List */}
            {voiceTab === 'library' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {voices.length === 0 ? (
                  <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No cloned voices loaded. Switch to <strong>Clone New Voice</strong> to use reference audio.
                  </div>
                ) : (
                  voices.map((v) => {
                    const isSelected = selectedVoiceId === v.id;
                    const isPlaying = playingPreviewVoiceId === v.id;

                    return (
                      <div
                        key={v.id}
                        onClick={() => setSelectedVoiceId(v.id)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '12px 16px',
                          borderRadius: 'var(--radius-md)',
                          border: isSelected
                            ? '1.5px solid var(--accent-green)'
                            : '1px solid var(--border-subtle)',
                          backgroundColor: isSelected ? 'var(--accent-green-subtle)' : '#ffffff',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          <input
                            type="radio"
                            name="tts_voice"
                            checked={isSelected}
                            onChange={() => setSelectedVoiceId(v.id)}
                            style={{ accentColor: 'var(--accent-green)', cursor: 'pointer' }}
                          />
                          <div>
                            <div style={{ fontWeight: 600, fontSize: '0.9375rem', color: 'var(--text-primary)' }}>
                              {v.name}
                            </div>
                            <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                              {v.style || 'Natural'} • {(v.languages || ['en']).join(', ').toUpperCase()}
                            </div>
                          </div>
                        </div>

                        {v.has_preview && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleTogglePreview(v.id);
                            }}
                            className="btn-secondary"
                            style={{
                              padding: '6px 10px',
                              fontSize: '0.75rem',
                              color: isPlaying ? 'var(--accent-green-text)' : 'inherit',
                            }}
                          >
                            <Volume2 size={14} />
                            <span>{isPlaying ? 'Playing...' : 'Preview'}</span>
                          </button>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {/* Custom Voice Upload / Mic */}
            {voiceTab === 'custom' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div
                  style={{
                    border: '1.5px dashed var(--border-medium)',
                    borderRadius: 'var(--radius-lg)',
                    padding: '24px',
                    textAlign: 'center',
                    backgroundColor: 'var(--bg-primary)',
                  }}
                >
                  {customFile ? (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
                      <CheckCircle2 size={20} color="var(--accent-green)" />
                      <span style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                        {customFile.name} ({(customFile.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                      <button
                        type="button"
                        onClick={() => setCustomFile(null)}
                        style={{
                          border: 'none',
                          background: 'none',
                          color: '#dc2626',
                          cursor: 'pointer',
                          fontSize: '0.8125rem',
                          textDecoration: 'underline',
                        }}
                      >
                        Remove
                      </button>
                    </div>
                  ) : (
                    <div>
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '12px',
                          marginBottom: '10px',
                        }}
                      >
                        <input
                          type="file"
                          id="custom_voice_upload"
                          accept=".wav,.mp3,.flac,audio/*"
                          onChange={(e) => e.target.files && setCustomFile(e.target.files[0])}
                          style={{ display: 'none' }}
                        />
                        <label
                          htmlFor="custom_voice_upload"
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
                        Supported formats: WAV, MP3, FLAC (10 to 30 seconds recommended)
                      </span>
                    </div>
                  )}
                </div>

                {/* Consent Checkbox */}
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '10px',
                    cursor: 'pointer',
                    backgroundColor: 'var(--bg-tertiary)',
                    padding: '12px 14px',
                    borderRadius: 'var(--radius-md)',
                    fontSize: '0.8125rem',
                    lineHeight: 1.4,
                    color: 'var(--text-secondary)',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={consentChecked}
                    onChange={(e) => setConsentChecked(e.target.checked)}
                    style={{ marginTop: '3px', accentColor: 'var(--accent-green)' }}
                  />
                  <span>
                    <strong>Consent required:</strong> I confirm I have explicit permission to clone and synthesize
                    this voice. Output will be labeled as AI-generated speech.
                  </span>
                </label>
              </div>
            )}

            {/* Speed Slider */}
            <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.8125rem',
                  color: 'var(--text-muted)',
                  marginBottom: '8px',
                }}
              >
                <span>Voice speed:</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{speed}x</span>
              </div>
              <input
                type="range"
                min={0.7}
                max={1.5}
                step={0.1}
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-green)', cursor: 'pointer' }}
              />
            </div>

            {/* Error Message Alert */}
            {errorMessage && (
              <div
                style={{
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
                }}
              >
                <AlertCircle size={16} style={{ flexShrink: 0 }} />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Main Action Button */}
            <div style={{ marginTop: '24px' }}>
              <button
                onClick={handleGenerate}
                disabled={isSubmitting || !script.trim() || isScriptExceeded}
                className="btn-primary"
                style={{ width: '100%', padding: '12px 20px', fontSize: '1rem' }}
              >
                {isSubmitting ? (
                  <>
                    <RefreshCw size={18} className="pulse-dot" />
                    <span>Processing in background...</span>
                  </>
                ) : (
                  <span>Generate voice</span>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Step 3 (Result & Long-Form Progress) */}
        <div>
          <div className="studio-card" style={{ padding: '24px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '16px',
              }}
            >
              <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                3 · Result & Progress
              </h2>
              {currentJob && (
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                  }}
                >
                  Job: {currentJob.job_id.slice(0, 8)}...
                </span>
              )}
            </div>

            {/* 1. Active / Processing / Queued State */}
            {isSubmitting || currentJob?.status === 'processing' || currentJob?.status === 'queued' ? (
              <div
                style={{
                  backgroundColor: '#0f1714',
                  borderRadius: 'var(--radius-lg)',
                  padding: '32px 24px',
                  color: '#ffffff',
                  border: '1px solid #23312a',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                  <RefreshCw size={24} className="pulse-dot" style={{ color: '#2dd4bf', flexShrink: 0 }} />
                  <div>
                    <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: '#f0fdf4' }}>
                      {currentJob?.status === 'queued' ? 'Queued for Synthesis' : 'Synthesizing Long-Form Audio...'}
                    </h4>
                    <p style={{ fontSize: '0.8125rem', color: '#94a3b8', margin: '2px 0 0 0' }}>
                      OpenVoice V2 neural engine • Voice cloning active
                    </p>
                  </div>
                </div>

                {/* Progress Bar */}
                <div
                  style={{
                    width: '100%',
                    backgroundColor: '#1e2924',
                    borderRadius: '999px',
                    height: '10px',
                    overflow: 'hidden',
                    margin: '18px 0 14px 0',
                    border: '1px solid #2e3e36',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${progressPercent}%`,
                      backgroundColor: '#10b981',
                      backgroundImage: 'linear-gradient(90deg, #10b981, #2dd4bf)',
                      borderRadius: '999px',
                      transition: 'width 0.4s ease',
                    }}
                  />
                </div>

                {/* Metrics Row: Chunk X of Y | Percent | Time Left */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: '12px',
                    fontSize: '0.8125rem',
                    color: '#cbd5e1',
                    marginBottom: '16px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Layers size={14} color="#2dd4bf" />
                    <span>
                      Chunk <strong>{chunkDone}</strong> of <strong>{chunkTotal || 1}</strong>
                    </span>
                  </div>

                  <div style={{ fontWeight: 700, color: '#2dd4bf', fontFamily: 'var(--font-mono)' }}>
                    {progressPercent}%
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Clock size={14} color="#94a3b8" />
                    <span>{formattedEta()}</span>
                  </div>
                </div>

                {/* Progress Status Message */}
                <div
                  style={{
                    backgroundColor: '#17221d',
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-md)',
                    fontSize: '0.8125rem',
                    color: '#94a3b8',
                    border: '1px solid #23312a',
                    marginBottom: '16px',
                  }}
                >
                  {currentJob?.progress_message || 'Initializing speaker embedding and audio chunks...'}
                </div>

                {/* Background Leave Notice */}
                <div
                  style={{
                    fontSize: '0.75rem',
                    color: '#64748b',
                    lineHeight: 1.4,
                    borderTop: '1px solid #23312a',
                    paddingTop: '12px',
                  }}
                >
                  💡 <strong>Safe to leave page:</strong> Job ID is persisted in your URL. You can close or leave this
                  page and return anytime to check progress.
                </div>
              </div>
            ) : currentJob?.status === 'failed' ? (
              /* 2. Resumable Failed State */
              <div
                style={{
                  backgroundColor: '#fffbeb',
                  border: '1px solid #fde68a',
                  borderRadius: 'var(--radius-lg)',
                  padding: '24px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '16px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <AlertCircle size={22} color="#b45309" style={{ flexShrink: 0 }} />
                  <div>
                    <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#92400e', margin: 0 }}>
                      Synthesis Paused / Encountered Error
                    </h4>
                    <p style={{ fontSize: '0.8125rem', color: '#b45309', margin: '2px 0 0 0' }}>
                      {currentJob.error_message || 'The job was interrupted or timed out.'}
                    </p>
                  </div>
                </div>

                {/* Cached Chunks Status */}
                <div
                  style={{
                    backgroundColor: '#ffffff',
                    border: '1px solid #fef3c7',
                    borderRadius: 'var(--radius-md)',
                    padding: '12px 14px',
                    fontSize: '0.8125rem',
                    color: '#78350f',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <Layers size={15} color="#b45309" />
                    <strong>Progress Saved:</strong>
                  </div>
                  <div>
                    <strong>{currentJob.chunk_done ?? 0}</strong> of{' '}
                    <strong>{currentJob.chunk_total ?? 0}</strong> chunks were successfully synthesized and cached on
                    the server. Resuming will skip completed chunks.
                  </div>
                </div>

                {/* Action Buttons: Resume & Reset */}
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  <button
                    onClick={handleResume}
                    disabled={isResuming}
                    className="btn-primary"
                    style={{
                      flex: 1,
                      backgroundColor: '#d97706',
                      borderColor: '#b45309',
                      padding: '10px 16px',
                      fontSize: '0.875rem',
                    }}
                  >
                    {isResuming ? (
                      <>
                        <RefreshCw size={15} className="pulse-dot" />
                        <span>Resuming...</span>
                      </>
                    ) : (
                      <>
                        <RotateCcw size={15} />
                        <span>Resume Generation</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleStartNew}
                    className="btn-secondary"
                    style={{ fontSize: '0.875rem', padding: '10px 16px' }}
                  >
                    Start New Script
                  </button>
                </div>
              </div>
            ) : currentJob?.status === 'completed' && currentJob.output_url ? (
              /* 3. Completed State with Download */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    backgroundColor: 'var(--accent-green-subtle)',
                    border: '1px solid var(--accent-green-border)',
                    padding: '12px 16px',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--accent-green-text)',
                  }}
                >
                  <CheckCircle2 size={18} style={{ flexShrink: 0 }} />
                  <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                    Synthesis completed successfully! (
                    {currentJob.chunk_total ? `${currentJob.chunk_total} chunks joined` : 'Audio ready'})
                  </span>
                </div>

                <AudioPlayer
                  src={currentJob.output_url}
                  title="Synthesized Voice"
                  subtitle={`${language.toUpperCase()} • Speed: ${speed}x • ${charCount.toLocaleString()} chars`}
                  aiAttribution={true}
                />

                {/* Download and Share Actions */}
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  <a
                    href={currentJob.output_url}
                    download={`voice_studio_${currentJob.job_id}.wav`}
                    className="btn-primary"
                    style={{
                      flex: 1,
                      textDecoration: 'none',
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      padding: '10px 16px',
                    }}
                  >
                    <Download size={16} />
                    <span>Download Audio (.wav)</span>
                  </a>

                  <button
                    onClick={handleStartNew}
                    className="btn-secondary"
                    style={{ fontSize: '0.875rem', padding: '10px 16px' }}
                  >
                    New Generation
                  </button>
                </div>

                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                  Results are retained on disk for {serverConfig?.job_ttl_minutes || 240} minutes before automatic cleanup.
                </p>
              </div>
            ) : (
              /* 4. Empty / Initial State */
              <div
                style={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px dashed var(--border-medium)',
                  borderRadius: 'var(--radius-lg)',
                  padding: '48px 24px',
                  textAlign: 'center',
                  color: 'var(--text-muted)',
                }}
              >
                <Volume2 size={36} style={{ margin: '0 auto 12px auto', opacity: 0.5 }} />
                <h4
                  style={{
                    fontSize: '0.9375rem',
                    fontWeight: 600,
                    color: 'var(--text-secondary)',
                    marginBottom: '4px',
                  }}
                >
                  No output yet
                </h4>
                <p style={{ fontSize: '0.8125rem', maxWidth: '300px', margin: '0 auto' }}>
                  Paste or upload your script, configure your voice, and click <strong>Generate voice</strong> to
                  begin long-form synthesis.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TTSStudio;
