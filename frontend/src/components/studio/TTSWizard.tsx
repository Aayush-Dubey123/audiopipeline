import React, { useState, useEffect, useRef } from 'react';
import {
  countWordsAccurately,
  estimateSpeechDuration,
  formatDuration,
} from '../../lib/types';
import type {
  Voice,
  PublicConfigResponse,
  JobStatusResponse,
  ReferenceCheckResponse,
  TTSPreviewResponse,
} from '../../lib/types';
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

// Ready-to-use sample scripts for rapid testing
const SAMPLE_SCRIPTS: Record<string, string> = {
  hi: 'नमस्ते! AI Voice Studio में आपका स्वागत है। यह एक उच्च गुणवत्ता वाली बहुभाषी स्पीच सिंथेसिस प्रणाली है, जो आपकी आवाज़ को हूबहू प्रस्तुत करती है।',
  mr: 'नमस्कार! AI Voice Studio मध्ये आपले स्वागत आहे. ही प्रणाली मराठी, हिंदी आणि इंग्रजी भाषेत अचूक आणि नैसर्गिक आवाजात संवाद साधण्यासाठी सक्षम आहे.',
  en: 'Welcome to AI Voice Studio. This advanced neural voice cloning studio generates natural, expressive speech with studio-grade fidelity across multiple languages.',
};

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
  const [referenceCheckError, setReferenceCheckError] = useState<string | null>(null);
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
  const referenceUrlRef = useRef<string | null>(null);

  // Revoke object URLs on unmount
  useEffect(() => {
    return () => {
      if (referenceUrlRef.current) {
        URL.revokeObjectURL(referenceUrlRef.current);
      }
    };
  }, []);

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

  // Accurate text measurements
  const charCount = Array.from(script).length;
  const wordCount = countWordsAccurately(script);
  const maxChars = config?.max_text_chars || 30000;
  const estimatedSeconds = estimateSpeechDuration(wordCount, charCount, language);
  const measuredComputeRatio = config?.seconds_per_audio_second || 0.8;
  const estimatedComputeTime = Math.max(2, Math.round(estimatedSeconds * measuredComputeRatio));

  // Step validation
  const getStepValidation = (): { isValid: boolean; reason: string } => {
    if (currentStep === 1) {
      if (charCount === 0) return { isValid: false, reason: 'Enter or paste script text to continue' };
      if (charCount < 5) return { isValid: false, reason: 'Script must be at least 5 characters' };
      if (charCount > maxChars) return { isValid: false, reason: `Script exceeds ${maxChars.toLocaleString()} character limit` };
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
        return { isValid: false, reason: 'Please confirm voice authorization consent to proceed' };
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

  // Invalidate downstream preview & full audio when voice or reference changes
  const invalidateDownstreamOutputs = () => {
    setPreviewAudioUrl(null);
    setPreviewJobId(null);
    setPreviewConfidence(null);
    setPreviewError(null);
    setActiveJobId(null);
    setActiveJob(null);
    setFullAudioError(null);
  };

  // Reference Audio Handler
  const handleReferenceReady = async (file: File) => {
    invalidateDownstreamOutputs();
    setReferenceFile(file);
    if (referenceUrlRef.current) {
      URL.revokeObjectURL(referenceUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    referenceUrlRef.current = url;
    setReferenceAudioUrl(url);
    setReferenceCheckError(null);
    setReferenceCheckResult(null);

    setIsCheckingReference(true);
    try {
      const result = await api.checkReferenceAudio({ file });
      setReferenceCheckResult(result);
    } catch (err: any) {
      setReferenceCheckResult(null);
      setReferenceCheckError(err?.message || 'Acoustic validation failed. Audio may be too short or unreadable.');
    } finally {
      setIsCheckingReference(false);
    }
  };

  const handleClearReference = () => {
    invalidateDownstreamOutputs();
    if (referenceUrlRef.current) {
      URL.revokeObjectURL(referenceUrlRef.current);
      referenceUrlRef.current = null;
    }
    setReferenceFile(null);
    setReferenceAudioUrl(null);
    setReferenceCheckResult(null);
    setReferenceCheckError(null);
  };

  // Generate Preview
  const handleGeneratePreview = async () => {
    setPreviewError(null);
    setPreviewAudioUrl(null);
    setIsPreviewGenerating(true);
    setPreviewStage('Queued');
    setPreviewLastUpdate(Date.now());

    const previewText = script.trim().slice(0, 200);
    if (!previewText) {
      setIsPreviewGenerating(false);
      setPreviewError('Script text is empty. Enter some text in Step 1 to generate a preview.');
      return;
    }

    try {
      let res: TTSPreviewResponse;
      if (voiceMode === 'saved') {
        if (!selectedVoiceId) {
          throw new Error('Please select a saved voice from the catalog.');
        }
        res = await api.getTTSPreview({
          text: previewText,
          voiceId: selectedVoiceId,
          language,
          speed: 1.0,
        });
      } else {
        if (!referenceFile) {
          throw new Error('Please record or upload a reference voice in Step 2 before generating a preview.');
        }
        if (!consent) {
          throw new Error('Explicit consent confirmation is required before generating custom reference voice speech.');
        }
        res = await api.getCustomTTSPreview({
          text: previewText,
          file: referenceFile,
          language,
          speed: 1.0,
          consent,
        });
      }

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
        {/* STEP 1: SCRIPT EDITOR */}
        {currentStep === 1 && (
          <div
            className="slide-panel"
            style={{
              display: 'flex',
              flexDirection: 'row',
              height: '100%',
              padding: 'var(--space-2)',
              gap: 'var(--space-2)',
              boxSizing: 'border-box',
            }}
          >
            {/* Left: Script Editor Card (Primary) */}
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                backgroundColor: 'var(--color-surface)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-control)',
                overflow: 'hidden',
              }}
            >
              {/* Card Header Toolbar */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  backgroundColor: 'var(--color-surface-muted)',
                  borderBottom: '1px solid var(--color-border)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                    Script Editor
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
                    (Hindi • Marathi • English)
                  </span>
                </div>

                {/* Actions & Samples */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {/* Sample Insert Buttons */}
                  <div style={{ display: 'flex', gap: '3px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginRight: '2px' }}>
                      Sample:
                    </span>
                    {(['hi', 'mr', 'en'] as const).map((langKey) => (
                      <button
                        key={langKey}
                        type="button"
                        onClick={() => {
                          setScript(SAMPLE_SCRIPTS[langKey]);
                          setLanguage(langKey);
                        }}
                        style={{
                          fontSize: '11px',
                          padding: '2px 6px',
                          borderRadius: 'var(--radius-control)',
                          border: '1px solid var(--color-border)',
                          backgroundColor: language === langKey ? 'var(--color-accent-light)' : 'var(--color-surface)',
                          color: language === langKey ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                          cursor: 'pointer',
                          fontWeight: language === langKey ? 600 : 400,
                        }}
                      >
                        {langKey.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  {script && (
                    <button
                      type="button"
                      onClick={() => setScript('')}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--color-text-muted)',
                        cursor: 'pointer',
                        fontSize: '11px',
                        padding: '2px 6px',
                        textDecoration: 'underline',
                      }}
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Textarea Area */}
              <div style={{ flex: 1, position: 'relative' }}>
                <textarea
                  value={script}
                  onChange={(e) => setScript(e.target.value)}
                  placeholder="Enter or paste your script in Hindi, Marathi, or English (e.g., नमस्ते! AI Voice Studio में आपका स्वागत है)..."
                  className="inner-scroll"
                  aria-label="Script text content"
                  style={{
                    width: '100%',
                    height: '100%',
                    resize: 'none',
                    padding: '14px 16px',
                    fontFamily: "'Plus Jakarta Sans', 'Noto Sans Devanagari', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                    fontSize: '15px',
                    lineHeight: 1.65,
                    color: 'var(--color-text-primary)',
                    backgroundColor: 'transparent',
                    border: 'none',
                    outline: 'none',
                    boxSizing: 'border-box',
                  }}
                />
              </div>
            </div>

            {/* Right: Configuration & Metrics Panel */}
            <div
              style={{
                width: 'min(280px, 35vw)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                flexShrink: 0,
                gap: 'var(--space-2)',
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                {/* Language Card */}
                <div
                  style={{
                    backgroundColor: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-control)',
                    padding: 'var(--space-1) var(--space-2)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                  }}
                >
                  <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                        Target Language
                      </span>
                      <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
                        {language === 'en' ? '~150 WPM' : language === 'mr' ? '~125 WPM' : '~130 WPM'}
                      </span>
                    </div>

                    <select
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                      style={{
                        height: '38px',
                        padding: '0 8px',
                        borderRadius: 'var(--radius-control)',
                        border: '1px solid var(--color-border)',
                        backgroundColor: 'var(--color-surface)',
                        fontFamily: 'var(--font-sans)',
                        fontSize: 'var(--font-sm)',
                        color: 'var(--color-text-primary)',
                        cursor: 'pointer',
                      }}
                    >
                      <option value="hi">Hindi (हिन्दी)</option>
                      <option value="mr">Marathi (मराठी)</option>
                      <option value="en">English (English)</option>
                    </select>
                  </label>
                  <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', lineHeight: 1.3 }}>
                    Speech cadence calibrated for conversational natural pacing.
                  </span>
                </div>

                {/* Metrics Card */}
                <div
                  style={{
                    backgroundColor: 'var(--color-surface-muted)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-control)',
                    padding: 'var(--space-1) var(--space-2)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                  }}
                >
                  <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Live Text Measurements
                  </span>

                  {/* Characters Stat */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)' }}>
                      <span style={{ color: 'var(--color-text-secondary)' }}>Characters</span>
                      <span
                        style={{
                          fontWeight: 600,
                          color: charCount > maxChars ? 'var(--color-error)' : 'var(--color-text-primary)',
                        }}
                      >
                        {charCount.toLocaleString()} / {maxChars.toLocaleString()}
                      </span>
                    </div>

                    {/* Character Capacity Bar */}
                    <div
                      style={{
                        height: '4px',
                        width: '100%',
                        backgroundColor: 'var(--color-border)',
                        borderRadius: '2px',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min(100, (charCount / maxChars) * 100)}%`,
                          backgroundColor:
                            charCount > maxChars
                              ? 'var(--color-error)'
                              : charCount > maxChars * 0.9
                              ? 'var(--color-warning)'
                              : 'var(--color-accent)',
                          transition: 'width 150ms ease',
                        }}
                      />
                    </div>
                  </div>

                  {/* Words Stat */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)' }}>
                    <span style={{ color: 'var(--color-text-secondary)' }}>Word Count</span>
                    <span style={{ fontWeight: 600 }}>{wordCount.toLocaleString()}</span>
                  </div>

                  {/* Duration Stat */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)', borderTop: '1px solid var(--color-border)', paddingTop: '6px' }}>
                    <span style={{ color: 'var(--color-text-secondary)' }}>Est. Duration</span>
                    <span style={{ fontWeight: 600, color: 'var(--color-accent)' }}>
                      {formatDuration(estimatedSeconds)} <span style={{ fontSize: '10px', color: 'var(--color-text-muted)', fontWeight: 400 }}>(est.)</span>
                    </span>
                  </div>
                </div>

                {/* File Upload Button */}
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => fileInputRef.current?.click()}
                  style={{ width: '100%' }}
                >
                  📄 Upload text file (.txt)
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt"
                  style={{ display: 'none' }}
                  onChange={handleTxtUpload}
                />
              </div>

              {/* Script Check Button */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
                <Button
                  size="base"
                  variant="secondary"
                  onClick={handleOpenScriptCheck}
                  disabled={charCount === 0 || isCheckingScript}
                  style={{ width: '100%', position: 'relative' }}
                >
                  <span>Script check & normalize</span>
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
                <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', textAlign: 'center', lineHeight: 1.25 }}>
                  Expands digits, acronyms, and punctuation for cleaner synthesis.
                </span>
              </div>
            </div>
          </div>
        )}

        {/* STEP 2: VOICE SELECTION & REFERENCE AUDIO */}
        {currentStep === 2 && (
          <div
            className="slide-panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
              padding: 'var(--space-2)',
              gap: 'var(--space-2)',
              boxSizing: 'border-box',
            }}
          >
            {/* Header: Segmented Toggle Control */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                paddingBottom: '4px',
              }}
            >
              <div
                style={{
                  display: 'inline-flex',
                  backgroundColor: 'var(--color-surface-muted)',
                  padding: '3px',
                  borderRadius: 'var(--radius-control)',
                  border: '1px solid var(--color-border)',
                }}
              >
                <button
                  type="button"
                  onClick={() => {
                    if (voiceMode !== 'saved') {
                      setVoiceMode('saved');
                      invalidateDownstreamOutputs();
                    }
                  }}
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
                  Saved Studio Voice ({voices.length})
                </button>

                <button
                  type="button"
                  onClick={() => {
                    if (voiceMode !== 'reference') {
                      setVoiceMode('reference');
                      invalidateDownstreamOutputs();
                    }
                  }}
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
                  Custom Reference Audio
                </button>
              </div>

              <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
                {voiceMode === 'saved'
                  ? 'Pre-verified studio voice profile'
                  : 'Zero-shot voice cloning from microphone or uploaded file'}
              </span>
            </div>

            {/* Saved Voices View */}
            {voiceMode === 'saved' ? (
              voices.length === 0 ? (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flex: 1,
                    gap: 'var(--space-2)',
                    textAlign: 'center',
                    padding: 'var(--space-3)',
                    backgroundColor: 'var(--color-surface-muted)',
                    borderRadius: 'var(--radius-control)',
                    border: '1px dashed var(--color-border)',
                  }}
                >
                  <span style={{ fontSize: '32px' }}>🎙️</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxWidth: '420px' }}>
                    <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                      No Saved Studio Voices Available
                    </span>
                    <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4, margin: 0 }}>
                      No catalog studio voices are registered in this environment. You can easily generate high-fidelity speech using your own voice by switching to custom reference audio.
                    </p>
                  </div>
                  <Button
                    variant="primary"
                    size="base"
                    onClick={() => {
                      setVoiceMode('reference');
                      invalidateDownstreamOutputs();
                    }}
                  >
                    Switch to Custom Reference Audio →
                  </Button>
                </div>
              ) : (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                    gap: 'var(--space-3)',
                    flex: 1,
                    minHeight: 0,
                    alignItems: 'start',
                    overflowY: 'auto',
                    padding: 'var(--space-0-5) 0',
                  }}
                >
                  {/* Left Column: Compact Selected Voice Card */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 'var(--space-2)',
                      padding: 'var(--space-2)',
                      borderRadius: 'var(--radius-control)',
                      backgroundColor: 'var(--color-surface)',
                      border: '1px solid var(--color-border)',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                    }}
                  >
                    {/* Header: Name, Style, and Selector (if multiple) */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                        <span style={{ fontSize: 'var(--font-base)', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                          {activeVoice?.name || 'Selected Voice'}
                        </span>
                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: 600,
                            padding: '2px 8px',
                            borderRadius: '12px',
                            backgroundColor: 'var(--color-accent-light)',
                            color: 'var(--color-accent)',
                          }}
                        >
                          {activeVoice?.style || 'Conversational'}
                        </span>
                      </div>

                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: '12px',
                          backgroundColor: 'var(--color-success-bg)',
                          color: 'var(--color-success)',
                          border: '1px solid var(--color-success-border)',
                        }}
                      >
                        Active Voice ✓
                      </span>
                    </div>

                    {/* Selector if multiple voices exist */}
                    {voices.length > 1 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                        <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
                          Select Voice:
                        </span>
                        <select
                          value={selectedVoiceId}
                          onChange={(e) => {
                            if (selectedVoiceId !== e.target.value) {
                              setSelectedVoiceId(e.target.value);
                              invalidateDownstreamOutputs();
                            }
                          }}
                          style={{
                            flex: 1,
                            height: '34px',
                            padding: '0 8px',
                            borderRadius: 'var(--radius-control)',
                            border: '1px solid var(--color-border)',
                            backgroundColor: 'var(--color-surface-muted)',
                            fontSize: 'var(--font-sm)',
                            color: 'var(--color-text-primary)',
                            cursor: 'pointer',
                          }}
                        >
                          {voices.map((v) => (
                            <option key={v.id} value={v.id}>
                              {v.name} ({v.languages.join(', ').toUpperCase()})
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.45, margin: 0 }}>
                      {activeVoice?.description || 'Natural speech voice profile calibrated for high-fidelity zero-distortion synthesis.'}
                    </p>

                    {/* Specifications Grid */}
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                        gap: 'var(--space-1)',
                        padding: 'var(--space-1) var(--space-2)',
                        backgroundColor: 'var(--color-surface-muted)',
                        borderRadius: 'var(--radius-control)',
                        border: '1px solid var(--color-border)',
                      }}
                    >
                      <div>
                        <span style={{ fontSize: '10px', color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: 600 }}>
                          Supported Languages
                        </span>
                        <div style={{ display: 'flex', gap: '4px', marginTop: '2px', flexWrap: 'wrap' }}>
                          {(activeVoice?.languages || ['hi', 'en', 'mr']).map((lang) => (
                            <span
                              key={lang}
                              style={{
                                fontSize: '10px',
                                fontWeight: 700,
                                padding: '1px 6px',
                                borderRadius: '4px',
                                backgroundColor: 'var(--color-surface)',
                                color: 'var(--color-text-primary)',
                                border: '1px solid var(--color-border)',
                                textTransform: 'uppercase',
                              }}
                            >
                              {lang === 'hi' ? 'HI (हिन्दी)' : lang === 'mr' ? 'MR (मराठी)' : 'EN (English)'}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div>
                        <span style={{ fontSize: '10px', color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: 600 }}>
                          Synthesis Engine
                        </span>
                        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-primary)', marginTop: '2px' }}>
                          Chatterbox Multilingual
                        </div>
                      </div>
                    </div>

                    {/* Voice Sample Audio Player */}
                    {activeVoice && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: 'var(--space-0-5)' }}>
                        <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                          Sample Voice Preview
                        </span>
                        <AudioPlayer
                          src={api.getVoicePreviewUrl(activeVoice.id)}
                          title={`${activeVoice.name} Sample Preview`}
                          compact
                        />
                      </div>
                    )}
                  </div>

                  {/* Right Column: Studio Guidelines & Pacing Advice */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 'var(--space-2)',
                      padding: 'var(--space-2)',
                      borderRadius: 'var(--radius-control)',
                      backgroundColor: 'var(--color-surface-muted)',
                      border: '1px solid var(--color-border)',
                    }}
                  >
                    <span style={{ fontSize: 'var(--font-sm)', fontWeight: 700, color: 'var(--color-text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Studio Voice Overview
                    </span>

                    <ul style={{ margin: 0, paddingLeft: '18px', fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.5, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <li>
                        <strong>Pre-indexed Conditioning:</strong> Uses local reference recording certified for zero acoustic noise floor and natural dynamics.
                      </li>
                      <li>
                        <strong>Zero Upload Wait:</strong> Synthesis begins immediately without uploading microphone or local audio files.
                      </li>
                      <li>
                        <strong>Calibrated Cadence:</strong> Optimized for conversational pacing across English, Hindi, and Marathi scripts.
                      </li>
                    </ul>

                    <div
                      style={{
                        padding: 'var(--space-1) var(--space-2)',
                        backgroundColor: 'var(--color-accent-light)',
                        borderRadius: 'var(--radius-control)',
                        border: '1px solid var(--color-accent-border)',
                        fontSize: '11px',
                        color: 'var(--color-accent)',
                        lineHeight: 1.4,
                      }}
                    >
                      💡 Want to clone your own voice instead? Click <strong>Custom Reference Audio</strong> above to record or upload a sample.
                    </div>
                  </div>
                </div>
              )
            ) : (
              /* Custom Reference: Microphone Recording & File Upload */
              <div style={{ flex: 1, minHeight: 0 }}>
                <AudioRecorder
                  onReferenceReady={handleReferenceReady}
                  referenceAudioUrl={referenceAudioUrl}
                  referenceFile={referenceFile}
                  consent={consent}
                  onConsentChange={setConsent}
                  checkResult={referenceCheckResult}
                  isChecking={isCheckingReference}
                  checkError={referenceCheckError}
                  onRetryCheck={() => referenceFile && handleReferenceReady(referenceFile)}
                  onClearReference={handleClearReference}
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
              boxSizing: 'border-box',
            }}
          >
            {/* Top: Script preview & Voice Selection info */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>Preview Audio (First ~200 Characters)</span>
                <span style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-muted)' }}>
                  Selected voice:{' '}
                  <strong>
                    {voiceMode === 'saved'
                      ? activeVoice?.name || 'Saved voice'
                      : referenceFile
                      ? `Custom reference (${referenceFile.name})`
                      : 'Custom reference (No audio yet)'}
                  </strong>
                </span>
              </div>
              <div
                style={{
                  padding: 'var(--space-1) var(--space-2)',
                  backgroundColor: 'var(--color-surface-muted)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-control)',
                  fontSize: 'var(--font-sm)',
                  color: 'var(--color-text-secondary)',
                  fontStyle: 'italic',
                }}
              >
                &ldquo;{script.trim().slice(0, 200)}...&rdquo;
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
              ) : voiceMode === 'reference' && !referenceFile ? (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    padding: 'var(--space-3)',
                    backgroundColor: 'var(--color-surface-muted)',
                    borderRadius: 'var(--radius-control)',
                    border: '1px dashed var(--color-border)',
                    maxWidth: '460px',
                    textAlign: 'center',
                  }}
                >
                  <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>
                    Reference audio clip required
                  </span>
                  <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4, margin: 0 }}>
                    You have selected Custom Reference Audio mode. Record or upload your reference speech in Step 2 to generate a preview.
                  </p>
                  <Button variant="primary" size="base" onClick={() => setCurrentStep(2)}>
                    ← Go to Step 2: Voice
                  </Button>
                </div>
              ) : voiceMode === 'reference' && !consent ? (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    padding: 'var(--space-3)',
                    backgroundColor: 'var(--color-surface-muted)',
                    borderRadius: 'var(--radius-control)',
                    border: '1px dashed var(--color-warning-border, #f59e0b)',
                    maxWidth: '460px',
                    textAlign: 'center',
                  }}
                >
                  <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-warning, #d97706)' }}>
                    Consent confirmation required
                  </span>
                  <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4, margin: 0 }}>
                    You must confirm ownership or authorization for voice cloning in Step 2 before synthesizing speech.
                  </p>
                  <Button variant="primary" size="base" onClick={() => setCurrentStep(2)}>
                    Review Consent in Step 2 →
                  </Button>
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
              boxSizing: 'border-box',
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
                Characters: <strong>{charCount.toLocaleString()}</strong> | Words: <strong>{wordCount.toLocaleString()}</strong> | Est. Duration: <strong>{formatDuration(estimatedSeconds)}</strong>
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
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>Master Output Audio</span>
                        <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>
                          Target voice:{' '}
                          <strong>
                            {voiceMode === 'saved'
                              ? activeVoice?.name || selectedVoiceId
                              : referenceFile
                              ? `Custom Reference (${referenceFile.name})`
                              : 'Custom Reference'}
                          </strong>
                        </span>
                      </div>
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
              ) : voiceMode === 'reference' && !referenceFile ? (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    padding: 'var(--space-3)',
                    backgroundColor: 'var(--color-surface-muted)',
                    borderRadius: 'var(--radius-control)',
                    border: '1px dashed var(--color-border)',
                    maxWidth: '460px',
                    textAlign: 'center',
                  }}
                >
                  <span style={{ fontSize: 'var(--font-base)', fontWeight: 600 }}>
                    Reference audio clip required
                  </span>
                  <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4, margin: 0 }}>
                    Custom Reference Audio mode is selected. Please record or upload your reference speech in Step 2 before generating full speech.
                  </p>
                  <Button variant="primary" size="base" onClick={() => setCurrentStep(2)}>
                    ← Go to Step 2: Voice
                  </Button>
                </div>
              ) : voiceMode === 'reference' && !consent ? (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    padding: 'var(--space-3)',
                    backgroundColor: 'var(--color-surface-muted)',
                    borderRadius: 'var(--radius-control)',
                    border: '1px dashed var(--color-warning-border, #f59e0b)',
                    maxWidth: '460px',
                    textAlign: 'center',
                  }}
                >
                  <span style={{ fontSize: 'var(--font-base)', fontWeight: 600, color: 'var(--color-warning, #d97706)' }}>
                    Consent confirmation required
                  </span>
                  <p style={{ fontSize: 'var(--font-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.4, margin: 0 }}>
                    You must confirm ownership or authorization for voice cloning in Step 2 before synthesizing speech.
                  </p>
                  <Button variant="primary" size="base" onClick={() => setCurrentStep(2)}>
                    Review Consent in Step 2 →
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
