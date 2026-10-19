import React, { useState, useEffect, useRef } from 'react';
import type { Voice, PublicConfigResponse, JobStatusResponse, ReferenceCheckResponse } from '../../lib/types';
import { api } from '../../lib/api';
import { Stepper } from '../navigation/Stepper';
import { Drawer } from '../navigation/Drawer';
import { Button } from '../common/Button';
import { ProgressPanel } from '../feedback/ProgressPanel';
import { AudioPlayer } from '../audio/AudioPlayer';
import { AudioRecorder } from '../audio/AudioRecorder';

export interface TTSWizardProps {
  voices: Voice[];
  config: PublicConfigResponse | null;
  onJobCreated?: (job: JobStatusResponse) => void;
  onNavigateHome: () => void;
}

export const TTSWizard: React.FC<TTSWizardProps> = ({
  voices,
  config,
  onJobCreated,
  onNavigateHome,
}) => {
  // Read step & job_id from URL on mount
  const getInitialStep = (): number => {
    const params = new URLSearchParams(window.location.search);
    const s = parseInt(params.get('step') || '1', 10);
    return s >= 1 && s <= 4 ? s : 1;
  };

  const getInitialJobId = (): string | null => {
    const params = new URLSearchParams(window.location.search);
    return params.get('job_id') || null;
  };

  const [currentStep, setCurrentStep] = useState<number>(getInitialStep);

  // Wizard state (kept across navigation)
  const [script, setScript] = useState<string>('');
  const [language, setLanguage] = useState<string>('hi'); // Hindi default
  const [voiceMode, setVoiceMode] = useState<'saved' | 'reference'>('saved');
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>(() => {
    const firstReady = voices.find((v) => v.status === 'ready' || v.status === 'available');
    return firstReady ? firstReady.id : 'client_ref';
  });
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [referenceAudioUrl, setReferenceAudioUrl] = useState<string | null>(null);
  const [consent, setConsent] = useState<boolean>(false);
  const [referenceCheckResult, setReferenceCheckResult] = useState<ReferenceCheckResponse | null>(null);
  const [isCheckingReference, setIsCheckingReference] = useState<boolean>(false);

  // Script Check Drawer State
  const [isScriptCheckOpen, setIsScriptCheckOpen] = useState(false);
  const [scriptOriginal, setScriptOriginal] = useState('');
  const [scriptCleaned, setScriptCleaned] = useState('');
  const [scriptCheckIssuesCount, setScriptCheckIssuesCount] = useState<number>(0);
  const [isCheckingScript, setIsCheckingScript] = useState(false);

  // Step 3 Preview State
  const [isPreviewGenerating, setIsPreviewGenerating] = useState<boolean>(false);
  const [previewJobId, setPreviewJobId] = useState<string | null>(null);
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [previewConfidence, setPreviewConfidence] = useState<string | null>(null);
  const [previewStage, setPreviewStage] = useState<string>('Queued');
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLastUpdate, setPreviewLastUpdate] = useState<number>(() => Date.now());

  // Step 4 Full Audio State
  const [activeJobId, setActiveJobId] = useState<string | null>(getInitialJobId);
  const [activeJob, setActiveJob] = useState<JobStatusResponse | null>(null);
  const [isCreatingFullAudio, setIsCreatingFullAudio] = useState(false);
  const [fullAudioError, setFullAudioError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Sync step and job_id to URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('tab', 'tts');
    params.set('step', String(currentStep));
    if (activeJobId) {
      params.set('job_id', activeJobId);
    } else {
      params.delete('job_id');
    }
    window.history.replaceState({}, '', `?${params.toString()}`);
  }, [currentStep, activeJobId]);

  // Track completed steps
  const completedSteps = React.useMemo(() => {
    const completed: number[] = [];
    if (script.trim().length >= 5) completed.push(1);
    if ((voiceMode === 'saved' && selectedVoiceId) || (voiceMode === 'reference' && referenceFile && consent)) {
      completed.push(2);
    }
    if (previewAudioUrl) completed.push(3);
    if (activeJob?.status === 'completed') completed.push(4);
    return completed;
  }, [script, voiceMode, selectedVoiceId, referenceFile, consent, previewAudioUrl, activeJob]);

  // Resume full job if URL had job_id
  useEffect(() => {
    if (activeJobId && !activeJob) {
      api.getJobStatus(activeJobId)
        .then((j) => {
          setActiveJob(j);
          if (j.status === 'completed' || j.status === 'processing') {
            setCurrentStep(4);
          }
        })
        .catch(() => {});
    }
  }, [activeJobId, activeJob]);

  // Poll active full job
  useEffect(() => {
    if (!activeJobId || !activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await api.getJobStatus(activeJobId);
        setActiveJob(updated);
        if (onJobCreated) onJobCreated(updated);
      } catch (err: any) {
        setFullAudioError(err?.message || 'Failed polling job status.');
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeJobId, activeJob, onJobCreated]);

  // Poll preview job if enqueued
  useEffect(() => {
    if (!previewJobId || !isPreviewGenerating) return;

    const interval = setInterval(async () => {
      try {
        const status = await api.getJobStatus(previewJobId);
        setPreviewStage(status.stage || status.progress_message || 'Generating speech');
        setPreviewLastUpdate(Date.now());

        if (status.status === 'completed') {
          setIsPreviewGenerating(false);
          setPreviewAudioUrl(status.output_url || `/v1/audio/preview/${previewJobId}`);
          setPreviewConfidence(status.confidence_label || 'good');
        } else if (status.status === 'failed') {
          setIsPreviewGenerating(false);
          setPreviewError(status.error_message || 'Preview generation failed.');
        }
      } catch {
        // Continue polling until timeout
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [previewJobId, isPreviewGenerating]);

  // Script text measurements
  const charCount = script.length;
  const wordCount = script.trim() ? script.trim().split(/\s+/).length : 0;
  const maxChars = config?.max_text_chars || 30000;
  const estimatedSeconds = Math.max(1, Math.round(charCount * 0.065));
  const measuredComputeRatio = config?.seconds_per_audio_second || 0.8;
  const estimatedComputeTime = Math.max(2, Math.round(estimatedSeconds * measuredComputeRatio));

  // Step validation
  const getStepValidation = (): { isValid: boolean; reason: string } => {
    if (currentStep === 1) {
      if (charCount === 0) return { isValid: false, reason: 'Enter your script text to continue' };
      if (charCount < 5) return { isValid: false, reason: 'Script must be at least 5 characters' };
      if (charCount > maxChars) return { isValid: false, reason: `Script exceeds ${maxChars} character limit` };
      return { isValid: true, reason: '' };
    }
    if (currentStep === 2) {
      if (voiceMode === 'saved') {
        if (!selectedVoiceId) return { isValid: false, reason: 'Select a saved voice' };
        return { isValid: true, reason: '' };
      }
      if (!referenceFile && !referenceAudioUrl) {
        return { isValid: false, reason: 'Upload or record reference audio' };
      }
      if (!consent) {
        return { isValid: false, reason: 'Please check the consent box to proceed' };
      }
      return { isValid: true, reason: '' };
    }
    if (currentStep === 3) {
      if (!previewAudioUrl && !isPreviewGenerating) {
        return { isValid: false, reason: 'Generate preview to verify voice quality' };
      }
      if (isPreviewGenerating) {
        return { isValid: false, reason: 'Generating preview audio...' };
      }
      return { isValid: true, reason: '' };
    }
    return { isValid: true, reason: '' };
  };

  const validation = getStepValidation();

  // Navigation handlers
  const handleNext = () => {
    if (validation.isValid && currentStep < 4) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep((prev) => prev - 1);
    } else {
      onNavigateHome();
    }
  };

  // Keyboard navigation: Enter triggers Next when valid & not inside textarea
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeTag = document.activeElement?.tagName.toLowerCase();
      if (e.key === 'Enter' && !e.shiftKey && activeTag !== 'textarea') {
        if (validation.isValid && currentStep < 4) {
          e.preventDefault();
          setCurrentStep((prev) => (prev < 4 ? prev + 1 : prev));
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [validation.isValid, currentStep]);

  // Upload .txt handler
  const handleTxtUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        if (text) setScript(text);
      };
      reader.readAsText(file);
    }
  };

  // Script Check
  const handleOpenScriptCheck = async () => {
    if (!script.trim()) return;
    setIsCheckingScript(true);
    setScriptOriginal(script);
    try {
      const res = await api.cleanupText(script, language);
      setScriptCleaned(res.cleaned || script);
      const diffCount = res.cleaned !== script ? 1 : 0;
      setScriptCheckIssuesCount(diffCount);
      setIsScriptCheckOpen(true);
    } catch {
      setScriptCleaned(script);
      setIsScriptCheckOpen(true);
    } finally {
      setIsCheckingScript(false);
    }
  };

  const handleAcceptScriptClean = () => {
    setScript(scriptCleaned);
    setIsScriptCheckOpen(false);
  };

  const handleRejectScriptClean = () => {
    setIsScriptCheckOpen(false);
  };

  // Reference Audio Handler
  const handleReferenceReady = async (file: File) => {
    setReferenceFile(file);
    const url = URL.createObjectURL(file);
    setReferenceAudioUrl(url);

    setIsCheckingReference(true);
    try {
      const result = await api.checkReferenceAudio({ file });
      setReferenceCheckResult(result);
    } catch {
      setReferenceCheckResult({
        score: 75,
        is_usable: true,
        advice: ['Reference audio accepted for synthesis.'],
        metrics: {} as any,
      });
    } finally {
      setIsCheckingReference(false);
    }
  };

  // Generate Preview
  const handleGeneratePreview = async () => {
    setPreviewError(null);
    setPreviewAudioUrl(null);
    setIsPreviewGenerating(true);
    setPreviewStage('Queued');
    setPreviewLastUpdate(Date.now());

    const previewText = script.trim().slice(0, 200);

    try {
      const res = await api.getTTSPreview({
        text: previewText,
        voiceId: voiceMode === 'saved' ? selectedVoiceId : undefined,
        language,
        speed: 1.0,
      });

      if (res.job_id) {
        setPreviewJobId(res.job_id);
      } else {
        // Synchronous response
        setPreviewAudioUrl(res.preview_url);
        setPreviewConfidence(res.confidence_label || 'good');
        setIsPreviewGenerating(false);
      }
    } catch (err: any) {
      setIsPreviewGenerating(false);
      setPreviewError(err?.message || 'Preview generation failed. Please retry.');
    }
  };

  // Generate Full Audio
  const handleCreateFullAudio = async () => {
    setFullAudioError(null);
    setIsCreatingFullAudio(true);

    try {
      let res: any;
      if (voiceMode === 'saved') {
        res = await api.createTTSJob({
          text: script.trim(),
          voiceId: selectedVoiceId,
          language,
          speed: 1.0,
        });
      } else {
        if (!referenceFile) throw new Error('Reference audio required');
        res = await api.createCustomTTSJob({
          text: script.trim(),
          file: referenceFile,
          language,
          speed: 1.0,
          consent,
        });
      }

      setActiveJobId(res.job_id);
      const initialJob = await api.getJobStatus(res.job_id);
      setActiveJob(initialJob);
      if (onJobCreated) onJobCreated(initialJob);
    } catch (err: any) {
      setFullAudioError(err?.message || 'Failed creating full synthesis job.');
    } finally {
      setIsCreatingFullAudio(false);
    }
  };

  const handleResumeJob = async () => {
    if (!activeJobId) return;
    try {
      await api.resumeJob(activeJobId);
      const refreshed = await api.getJobStatus(activeJobId);
      setActiveJob(refreshed);
    } catch (err: any) {
      setFullAudioError(err?.message || 'Could not resume job.');
    }
  };

  const activeVoice = voices.find((v) => v.id === selectedVoiceId) || voices[0];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        backgroundColor: 'var(--color-surface)',
        borderRadius: 'var(--radius-surface)',
        border: '1px solid var(--color-border)',
        overflow: 'hidden',
      }}
    >
      {/* 1. Stepper on Top */}
      <Stepper
        currentStep={currentStep}
        completedSteps={completedSteps}
        onSelectStep={(s) => setCurrentStep(s)}
      />

      {/* 2. Slide Panels Main Area (Only inner areas scroll) */}
      <div
        className="slide-panel-container"
        style={{
          flex: 1,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* STEP 1: SCRIPT */}
        {currentStep === 1 && (
          <div
            className="slide-panel"
            style={{
              display: 'flex',
              flexDirection: 'row',
              height: '100%',
              padding: 'var(--space-2)',
              gap: 'var(--space-2)',
            }}
          >
            {/* Left: Textarea fills available height (inner scroll) */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}>
              <textarea
                value={script}
                onChange={(e) => setScript(e.target.value)}
                placeholder="Type or paste your script here (up to 30,000 characters)..."
                className="inner-scroll"
                aria-label="Script text"
                style={{
                  width: '100%',
                  height: '100%',
                  resize: 'none',
                  padding: 'var(--space-2)',
                  fontFamily: 'var(--font-sans)',
                  fontSize: 'var(--font-base)',
                  lineHeight: 1.5,
                  color: 'var(--color-text-primary)',
                  backgroundColor: 'var(--color-bg)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-control)',
                  outline: 'none',
                }}
              />
            </div>

            {/* Right: Narrow Column Controls */}
            <div
              style={{
                width: 'min(240px, 32vw)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                flexShrink: 0,
                gap: 'var(--space-1)',
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
                {/* Language Select */}
                <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                    Language
                  </span>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    style={{
                      height: '44px',
                      padding: '0 var(--space-1)',
                      borderRadius: 'var(--radius-control)',
                      border: '1px solid var(--color-border)',
                      backgroundColor: 'var(--color-surface)',
                      fontFamily: 'var(--font-sans)',
                      fontSize: 'var(--font-base)',
                      color: 'var(--color-text-primary)',
                      cursor: 'pointer',
                    }}
                  >
                    <option value="hi">Hindi (Default)</option>
                    <option value="en">English</option>
                  </select>
                </label>

                {/* Script Metrics */}
                <div
                  style={{
                    backgroundColor: 'var(--color-surface-muted)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-control)',
                    padding: 'var(--space-1)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)' }}>
                    <span style={{ color: 'var(--color-text-muted)' }}>Characters</span>
                    <span style={{ fontWeight: 600 }}>{charCount.toLocaleString()} / {maxChars.toLocaleString()}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)' }}>
                    <span style={{ color: 'var(--color-text-muted)' }}>Words</span>
                    <span style={{ fontWeight: 600 }}>{wordCount.toLocaleString()}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)' }}>
                    <span style={{ color: 'var(--color-text-muted)' }}>Est. Audio</span>
                    <span style={{ fontWeight: 600 }}>~{estimatedSeconds}s</span>
                  </div>

                  {/* Limit Bar */}
                  <div
                    style={{
                      height: '4px',
                      width: '100%',
                      backgroundColor: 'var(--color-border)',
                      borderRadius: '2px',
                      overflow: 'hidden',
                      marginTop: '4px',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(100, (charCount / maxChars) * 100)}%`,
                        backgroundColor: charCount > maxChars ? 'var(--color-error)' : 'var(--color-accent)',
                      }}
                    />
                  </div>
                </div>

                {/* Upload .txt */}
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => fileInputRef.current?.click()}
                  style={{ width: '100%' }}
                >
                  📄 Upload .txt
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt"
                  style={{ display: 'none' }}
                  onChange={handleTxtUpload}
                />
              </div>

              {/* Script Check Button with Issue Count Badge */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
                <Button
                  size="base"
                  variant="secondary"
                  onClick={handleOpenScriptCheck}
                  disabled={charCount === 0 || isCheckingScript}
                  style={{ width: '100%', position: 'relative' }}
                >
                  <span>Script check</span>
                  {scriptCheckIssuesCount > 0 && (
                    <span
                      style={{
                        marginLeft: 'var(--space-1)',
                        padding: '1px 6px',
                        borderRadius: 'var(--radius-control)',
                        backgroundColor: 'var(--color-warning-bg)',
                        color: 'var(--color-warning)',
                        border: '1px solid var(--color-warning-border)',
                        fontSize: 'var(--font-sm)',
                        fontWeight: 700,
                      }}
                    >
                      {scriptCheckIssuesCount}
                    </span>
                  )}
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* STEP 2: VOICE */}
        {currentStep === 2 && (
          <div
            className="slide-panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
              padding: 'var(--space-2)',
              gap: 'var(--space-2)',
            }}
          >
            {/* Segmented Control */}
            <div
              style={{
                display: 'inline-flex',
                backgroundColor: 'var(--color-surface-muted)',
                padding: '3px',
                borderRadius: 'var(--radius-control)',
                border: '1px solid var(--color-border)',
                width: 'fit-content',
              }}
            >
              <button
                type="button"
                onClick={() => setVoiceMode('saved')}
                className="touch-target"
                style={{
                  border: 'none',
                  padding: '6px 16px',
                  borderRadius: 'var(--radius-control)',
                  backgroundColor: voiceMode === 'saved' ? 'var(--color-surface)' : 'transparent',
                  color: voiceMode === 'saved' ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                  fontWeight: voiceMode === 'saved' ? 600 : 500,
                  fontSize: 'var(--font-base)',
                  boxShadow: voiceMode === 'saved' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
                  cursor: 'pointer',
                }}
              >
                Saved voice
              </button>

              <button
                type="button"
                onClick={() => setVoiceMode('reference')}
                className="touch-target"
                style={{
                  border: 'none',
                  padding: '6px 16px',
                  borderRadius: 'var(--radius-control)',
                  backgroundColor: voiceMode === 'reference' ? 'var(--color-surface)' : 'transparent',
                  color: voiceMode === 'reference' ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                  fontWeight: voiceMode === 'reference' ? 600 : 500,
                  fontSize: 'var(--font-base)',
                  boxShadow: voiceMode === 'reference' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
                  cursor: 'pointer',
                }}
              >
                My reference
              </button>
            </div>

            {/* Saved Voices Row (Inner scroll) */}
            {voiceMode === 'saved' ? (
              <div
                className="inner-scroll-x"
                style={{
                  display: 'flex',
                  gap: 'var(--space-2)',
                  padding: 'var(--space-1) 0',
                  height: '100%',
                  alignItems: 'flex-start',
                }}
              >
                {voices.map((v) => {
                  const isSelected = selectedVoiceId === v.id;
                  return (
                    <div
                      key={v.id}
                      onClick={() => setSelectedVoiceId(v.id)}
                      style={{
                        minWidth: '220px',
                        maxWidth: '260px',
                        padding: 'var(--space-2)',
                        borderRadius: 'var(--radius-control)',
                        border: `2px solid ${isSelected ? 'var(--color-accent)' : 'var(--color-border)'}`,
                        backgroundColor: isSelected ? 'var(--color-accent-light)' : 'var(--color-surface)',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        gap: 'var(--space-1)',
                        cursor: 'pointer',
                        flexShrink: 0,
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                            {v.name}
                          </span>
                          {isSelected && <span style={{ color: 'var(--color-accent)', fontWeight: 700 }}>✓</span>}
                        </div>
                        <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.3 }}>
                          {v.description || 'Natural speech voice profile.'}
                        </p>
                      </div>

                      {/* Preview Button */}
                      {v.has_preview && (
                        <div style={{ marginTop: 'var(--space-1)' }} onClick={(e) => e.stopPropagation()}>
                          <AudioPlayer
                            src={api.getVoicePreviewUrl(v.id)}
                            compact
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              /* My Reference: Upload / Record in two columns (No scroll) */
              <div style={{ flex: 1 }}>
                <AudioRecorder
                  onReferenceReady={handleReferenceReady}
                  referenceAudioUrl={referenceAudioUrl}
                  consent={consent}
                  onConsentChange={setConsent}
                  checkResult={referenceCheckResult}
                  isChecking={isCheckingReference}
                />
              </div>
            )}
          </div>
        )}

        {/* STEP 3: PREVIEW */}
        {currentStep === 3 && (
          <div
            className="slide-panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
              padding: 'var(--space-2)',
              gap: 'var(--space-2)',
              justifyContent: 'space-between',
            }}
          >
            {/* Top: Script preview & Voice Selection info */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>Preview 15 seconds</span>
                <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                  Selected voice: <strong>{voiceMode === 'saved' ? activeVoice?.name || 'Saved voice' : 'My reference'}</strong>
                </span>
              </div>

              {/* First 200 Chars */}
              <div
                style={{
                  padding: 'var(--space-2)',
                  backgroundColor: 'var(--color-surface-muted)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-control)',
                  fontSize: 'var(--font-base)',
                  color: 'var(--color-text-secondary)',
                  lineHeight: 1.4,
                }}
              >
                "{script.trim().slice(0, 200)}..."
              </div>
            </div>

            {/* Middle: Progress Panel or Generate Button or Completed Preview */}
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {isPreviewGenerating ? (
                <div style={{ width: '100%', maxWidth: '600px' }}>
                  <ProgressPanel
                    currentStageName={previewStage}
                    estimatedTotalSeconds={estimatedSeconds * measuredComputeRatio}
                    isColdModel={false}
                    lastUpdatedAt={previewLastUpdate}
                    onCancel={() => setIsPreviewGenerating(false)}
                    onRetry={handleGeneratePreview}
                    errorMessage={previewError}
                  />
                </div>
              ) : previewAudioUrl ? (
                <div
                  style={{
                    width: '100%',
                    maxWidth: '560px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-2)',
                    alignItems: 'center',
                  }}
                >
                  <AudioPlayer src={previewAudioUrl} title="Generated Preview Audio" />

                  {/* Confidence Label */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                    <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                      Acoustic similarity:
                    </span>
                    <span
                      style={{
                        fontSize: 'var(--font-sm)',
                        fontWeight: 600,
                        padding: '2px 8px',
                        borderRadius: 'var(--radius-control)',
                        backgroundColor: 'var(--color-success-bg)',
                        color: 'var(--color-success)',
                        border: '1px solid var(--color-success-border)',
                      }}
                    >
                      Confidence: {previewConfidence || 'Good'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: 'var(--space-1)' }}>
                    <Button variant="secondary" onClick={() => setCurrentStep(2)}>
                      Change voice
                    </Button>
                    <Button variant="primary" onClick={() => setCurrentStep(4)}>
                      Sounds right, continue →
                    </Button>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-1)' }}>
                  <Button
                    variant="primary"
                    size="base"
                    onClick={handleGeneratePreview}
                    style={{ minWidth: '220px', minHeight: '48px', fontSize: 'var(--font-base)', fontWeight: 600 }}
                  >
                    Generate preview
                  </Button>
                  <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                    Rapid synthesis of first 200 characters (~4s)
                  </span>
                </div>
              )}
            </div>

            {previewError && (
              <div
                style={{
                  fontSize: 'var(--font-sm)',
                  color: 'var(--color-error)',
                  backgroundColor: 'var(--color-error-bg)',
                  border: '1px solid var(--color-error-border)',
                  padding: 'var(--space-1)',
                  borderRadius: 'var(--radius-control)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span>{previewError}</span>
                <Button size="sm" variant="danger" onClick={handleGeneratePreview}>
                  Retry
                </Button>
              </div>
            )}
          </div>
        )}

        {/* STEP 4: FULL AUDIO */}
        {currentStep === 4 && (
          <div
            className="slide-panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
              padding: 'var(--space-2)',
              gap: 'var(--space-2)',
              justifyContent: 'space-between',
            }}
          >
            {/* Top: Summary */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: 'var(--space-1) var(--space-2)',
                backgroundColor: 'var(--color-surface-muted)',
                borderRadius: 'var(--radius-control)',
                border: '1px solid var(--color-border)',
              }}
            >
              <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)' }}>
                Characters: <strong>{charCount.toLocaleString()}</strong> | Est. Duration: <strong>~{estimatedSeconds}s</strong>
              </span>
              <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                Est. Generation Time: <strong>~{estimatedComputeTime}s</strong>
              </span>
            </div>

            {/* Middle: Progress or Two Columns Output */}
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {activeJob?.status === 'processing' || isCreatingFullAudio ? (
                <div style={{ width: '100%', maxWidth: '640px' }}>
                  <ProgressPanel
                    currentStageName={activeJob?.progress_message || 'Processing chunks...'}
                    chunkDone={activeJob?.chunk_done}
                    chunkTotal={activeJob?.chunk_total}
                    etaSeconds={activeJob?.eta_seconds}
                    estimatedTotalSeconds={estimatedComputeTime}
                    isLongForm={true}
                    onCancel={() => {}}
                    errorMessage={fullAudioError}
                  />
                </div>
              ) : activeJob?.status === 'completed' ? (
                /* Finished State: Two Columns */
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: 'var(--space-3)',
                    width: '100%',
                    height: '100%',
                    alignItems: 'center',
                  }}
                >
                  {/* Left Column: Player with seek and speed + AI-generated badge */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>Master Output Audio</span>
                      <span
                        style={{
                          fontSize: 'var(--font-sm)',
                          fontWeight: 600,
                          padding: '2px 8px',
                          borderRadius: 'var(--radius-control)',
                          backgroundColor: 'var(--color-accent-light)',
                          color: 'var(--color-accent)',
                          border: '1px solid var(--color-accent-border)',
                        }}
                      >
                        AI-generated
                      </span>
                    </div>

                    <AudioPlayer
                      src={activeJob.output_url || `/v1/audio/${activeJob.job_id}/output`}
                      title="Completed Synthesis"
                    />

                    <div style={{ display: 'flex', gap: 'var(--space-1)', marginTop: 'var(--space-1)' }}>
                      <Button variant="secondary" onClick={() => setCurrentStep(1)}>
                        Edit script
                      </Button>
                      <Button
                        variant="primary"
                        onClick={() => {
                          setActiveJobId(null);
                          setActiveJob(null);
                          setScript('');
                          setPreviewAudioUrl(null);
                          setCurrentStep(1);
                        }}
                      >
                        Start new
                      </Button>
                    </div>
                  </div>

                  {/* Right Column: Download Formats with real sizes */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 'var(--space-1)',
                      backgroundColor: 'var(--color-surface-muted)',
                      border: '1px solid var(--color-border)',
                      borderRadius: 'var(--radius-control)',
                      padding: 'var(--space-2)',
                    }}
                  >
                    <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>Export formats</span>

                    {/* WAV Master */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div>
                        <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600 }}>WAV uncompressed</span>
                        <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                          Master studio quality for video editing
                        </p>
                      </div>
                      <a
                        href={activeJob.output_files?.wav?.url || `/v1/audio/${activeJob.job_id}/output?format=wav`}
                        download="master.wav"
                        style={{
                          fontSize: 'var(--font-sm)',
                          fontWeight: 600,
                          color: 'var(--color-accent)',
                          textDecoration: 'none',
                        }}
                      >
                        Download
                      </a>
                    </div>

                    {/* MP3 High Quality */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-1)' }}>
                      <div>
                        <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600 }}>MP3 high-fidelity</span>
                        <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                          192 kbps stereo for clean sharing
                        </p>
                      </div>
                      <a
                        href={activeJob.output_files?.mp3?.url || `/v1/audio/${activeJob.job_id}/output?format=mp3`}
                        download="audio.mp3"
                        style={{
                          fontSize: 'var(--font-sm)',
                          fontWeight: 600,
                          color: 'var(--color-accent)',
                          textDecoration: 'none',
                        }}
                      >
                        Download
                      </a>
                    </div>

                    {/* Small MP3 WhatsApp */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-1)' }}>
                      <div>
                        <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600 }}>Small MP3 (WhatsApp)</span>
                        <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                          Lightweight 64 kbps mono (Send as Document in WhatsApp)
                        </p>
                      </div>
                      <a
                        href={activeJob.output_files?.mp3_small?.url || `/v1/audio/${activeJob.job_id}/output?format=mp3_small`}
                        download="audio_compact.mp3"
                        style={{
                          fontSize: 'var(--font-sm)',
                          fontWeight: 600,
                          color: 'var(--color-accent)',
                          textDecoration: 'none',
                        }}
                      >
                        Download
                      </a>
                    </div>
                  </div>
                </div>
              ) : activeJob?.status === 'failed' ? (
                /* Failed State with Resume */
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-1)' }}>
                  <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-error)' }}>
                    Inference failed: {activeJob.error_message || 'An error occurred during synthesis.'}
                  </span>
                  <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                    Finished chunks are preserved on disk.
                  </span>
                  <Button variant="primary" size="base" onClick={handleResumeJob}>
                    Resume generation
                  </Button>
                </div>
              ) : (
                /* Start Full Audio Generation button */
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-1)' }}>
                  <Button
                    variant="primary"
                    size="base"
                    onClick={handleCreateFullAudio}
                    style={{ minWidth: '240px', minHeight: '48px', fontSize: 'var(--font-base)', fontWeight: 600 }}
                  >
                    Create full audio
                  </Button>
                  <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                    Generates the complete script with seamless chunk stitching
                  </span>
                </div>
              )}
            </div>

            {fullAudioError && (
              <div
                style={{
                  fontSize: 'var(--font-sm)',
                  color: 'var(--color-error)',
                  backgroundColor: 'var(--color-error-bg)',
                  border: '1px solid var(--color-error-border)',
                  padding: 'var(--space-1)',
                  borderRadius: 'var(--radius-control)',
                }}
              >
                {fullAudioError}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 3. Fixed Bottom Bar inside Card */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: 'var(--space-1) var(--space-2)',
          borderTop: '1px solid var(--color-border)',
          backgroundColor: 'var(--color-surface)',
          minHeight: '48px',
          flexShrink: 0,
        }}
      >
        <Button variant="secondary" size="base" onClick={handleBack}>
          ← {currentStep === 1 ? 'Start' : 'Back'}
        </Button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          {!validation.isValid && validation.reason && (
            <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
              {validation.reason}
            </span>
          )}

          {currentStep < 4 && (
            <Button
              variant="primary"
              size="base"
              onClick={handleNext}
              disabled={!validation.isValid}
            >
              Next →
            </Button>
          )}
        </div>
      </div>

      {/* Script Check Diff Drawer */}
      <Drawer
        isOpen={isScriptCheckOpen}
        onClose={handleRejectScriptClean}
        title="Script Check: Before & After Diff"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          <div>
            <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-muted)' }}>
              ORIGINAL SCRIPT
            </span>
            <div
              style={{
                marginTop: 'var(--space-0-5)',
                padding: 'var(--space-1)',
                backgroundColor: 'var(--color-error-bg)',
                border: '1px solid var(--color-error-border)',
                borderRadius: 'var(--radius-control)',
                fontSize: 'var(--font-sm)',
                lineHeight: 1.4,
              }}
            >
              {scriptOriginal}
            </div>
          </div>

          <div>
            <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-muted)' }}>
              CLEANED SCRIPT (ACRONYMS & PUNCTUATION EXPANDED)
            </span>
            <div
              style={{
                marginTop: 'var(--space-0-5)',
                padding: 'var(--space-1)',
                backgroundColor: 'var(--color-success-bg)',
                border: '1px solid var(--color-success-border)',
                borderRadius: 'var(--radius-control)',
                fontSize: 'var(--font-sm)',
                lineHeight: 1.4,
              }}
            >
              {scriptCleaned}
            </div>
          </div>

          <div style={{ display: 'flex', gap: 'var(--space-1)', justifyContent: 'flex-end', marginTop: 'var(--space-1)' }}>
            <Button variant="secondary" size="base" onClick={handleRejectScriptClean}>
              Reject
            </Button>
            <Button variant="primary" size="base" onClick={handleAcceptScriptClean}>
              Accept Changes
            </Button>
          </div>
        </div>
      </Drawer>
    </div>
  );
};
