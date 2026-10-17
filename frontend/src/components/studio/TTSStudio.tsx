import React, { useState, useRef } from 'react';
import {
  Mic,
  Square,
  Upload,
  Sparkles,
  Play,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Shield,
  Loader2,
  Wand2,
} from 'lucide-react';
import type { Voice, SupportedLanguage, JobStatusResponse } from '../../lib/types';
import { api, ApiError } from '../../lib/api';
import { AudioPlayer } from '../common/AudioPlayer';

interface TTSStudioProps {
  voices: Voice[];
  onJobStarted: (jobId: string) => void;
}

export const TTSStudio: React.FC<TTSStudioProps> = ({ voices, onJobStarted }) => {
  // Mode: catalog voice vs custom reference audio
  const [mode, setMode] = useState<'preset' | 'custom'>('preset');

  // Input states
  const [text, setText] = useState('Welcome to AI Voice Studio. This speech was synthesized using neural tone color transfer.');
  const [selectedVoiceId, setSelectedVoiceId] = useState(voices[0]?.id || 'my_voice');
  const [language, setLanguage] = useState<SupportedLanguage>('en');
  const [speed, setSpeed] = useState<number>(1.0);

  // Custom reference audio states
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [consentGiven, setConsentGiven] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordedBlobUrl, setRecordedBlobUrl] = useState<string | null>(null);

  // Generation & Status states
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCleaning, setIsCleaning] = useState(false);
  const [currentJob, setCurrentJob] = useState<JobStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cleanFeedback, setCleanFeedback] = useState<string | null>(null);

  // Audio Recorder refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const timerIntervalRef = useRef<number | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Filter voices that support the selected language
  const compatibleVoices = voices.filter(
    (v) => v.status === 'available' && v.languages.includes(language)
  );

  // Quick preset samples
  const loadPresetSample = (lang: SupportedLanguage) => {
    setLanguage(lang);
    if (lang === 'en') {
      setText('Welcome to AI Voice Studio. Our system enables crystal-clear text-to-speech with natural intonation.');
    } else if (lang === 'hi') {
      setText('एआई वॉयस स्टूडियो में आपका स्वागत है। यह आवाज़ आधुनिक न्यूरल तकनीक से बनाई गई है।');
    } else if (lang === 'mr') {
      setText('एआय व्हॉइस स्टुडिओमध्ये आपले स्वागत आहे. येथे मराठी भाषेत अतिशय स्पष्ट आवाज तयार होतो.');
    }
  };

  // Text cleanup handler
  const handleCleanText = async () => {
    if (!text.trim()) return;
    setIsCleaning(true);
    setCleanFeedback(null);
    try {
      const res = await api.cleanupText(text, language);
      setText(res.cleaned);
      setCleanFeedback(`Cleaned for TTS (${res.provider_used})`);
      setTimeout(() => setCleanFeedback(null), 4000);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : 'Text cleanup failed.';
      setErrorMessage(msg);
    } finally {
      setIsCleaning(false);
    }
  };

  // Start microphone recording
  const startRecording = async () => {
    try {
      setErrorMessage(null);
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
        setReferenceFile(file);
        setRecordedBlobUrl(URL.createObjectURL(audioBlob));
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(250);
      setIsRecording(true);
      setRecordingSeconds(0);
      timerIntervalRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setErrorMessage('Microphone access denied. Please allow microphone permissions or upload an audio file.');
    }
  };

  // Stop microphone recording
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

  // File upload change
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setReferenceFile(file);
      setRecordedBlobUrl(URL.createObjectURL(file));
      setErrorMessage(null);
    }
  };

  // Submit Generation Job
  const handleGenerate = async () => {
    if (!text.trim()) {
      setErrorMessage('Please enter script text before generating speech.');
      return;
    }

    if (mode === 'custom' && !referenceFile) {
      setErrorMessage('Please provide a reference voice recording (upload or record).');
      return;
    }

    if (mode === 'custom' && !consentGiven) {
      setErrorMessage('You must confirm consent that you have permission to clone this voice.');
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);
    setCurrentJob(null);

    try {
      let created;
      if (mode === 'preset') {
        created = await api.createTTSJob({
          text,
          voiceId: selectedVoiceId,
          language,
          speed,
        });
      } else {
        created = await api.createCustomTTSJob({
          text,
          file: referenceFile!,
          language,
          speed,
          consent: consentGiven,
        });
      }

      onJobStarted(created.job_id);

      // Poll until finished
      const finalStatus = await api.pollJob(
        created.job_id,
        (progress) => {
          setCurrentJob(progress);
        },
        1000,
        120000
      );

      setCurrentJob(finalStatus);
      if (finalStatus.status === 'failed') {
        setErrorMessage(finalStatus.error_message || 'Speech generation failed.');
      }
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : 'Failed to queue speech synthesis job.';
      setErrorMessage(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Banner / Headline */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }} className="gradient-text">
            Text-to-Speech Studio
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            Transform text into natural voices in English, Hindi, and Marathi with zero licensing fees.
          </p>
        </div>

        {/* Mode Selector Tabs */}
        <div style={{
          display: 'flex',
          background: 'rgba(15, 17, 26, 0.8)',
          padding: '4px',
          borderRadius: '12px',
          border: '1px solid var(--border-subtle)',
        }}>
          <button
            onClick={() => setMode('preset')}
            className="btn btn-sm"
            style={{
              background: mode === 'preset' ? 'var(--accent-violet)' : 'transparent',
              color: mode === 'preset' ? '#fff' : 'var(--text-muted)',
              borderRadius: '8px',
            }}
          >
            Catalog Voice
          </button>
          <button
            onClick={() => setMode('custom')}
            className="btn btn-sm"
            style={{
              background: mode === 'custom' ? 'var(--accent-cyan)' : 'transparent',
              color: mode === 'custom' ? '#fff' : 'var(--text-muted)',
              borderRadius: '8px',
            }}
          >
            <Sparkles size={14} />
            <span>Custom Voice Clone</span>
          </button>
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="grid-cols-2">
        {/* Left Column: Script & Synthesis Controls */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Language Selector & Sample Presets */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                Language
              </label>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button onClick={() => loadPresetSample('en')} className="btn btn-secondary btn-sm" style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
                  EN Demo
                </button>
                <button onClick={() => loadPresetSample('hi')} className="btn btn-secondary btn-sm" style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
                  हिन्दी Demo
                </button>
                <button onClick={() => loadPresetSample('mr')} className="btn btn-secondary btn-sm" style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
                  मराठी Demo
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
              {[
                { id: 'en', label: 'English', sub: 'MeloTTS + OpenVoice' },
                { id: 'hi', label: 'Hindi (हिन्दी)', sub: 'Meta MMS-TTS' },
                { id: 'mr', label: 'Marathi (मराठी)', sub: 'Meta MMS-TTS' },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => setLanguage(item.id as SupportedLanguage)}
                  style={{
                    padding: '10px',
                    borderRadius: '10px',
                    textAlign: 'left',
                    background: language === item.id ? 'rgba(139, 92, 246, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                    border: `1px solid ${language === item.id ? 'var(--accent-violet)' : 'var(--border-subtle)'}`,
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{item.label}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{item.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Script Editor */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                Script Input
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: text.length > 500 ? 'var(--danger)' : 'var(--text-muted)' }}>
                  {text.length} / 500 chars
                </span>
                <button
                  onClick={handleCleanText}
                  disabled={isCleaning || !text.trim()}
                  className="btn btn-secondary btn-sm"
                  title="Normalize abbreviations, numbers, and clean punctuation"
                  style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                >
                  {isCleaning ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
                  <span>{cleanFeedback || 'Polish Script'}</span>
                </button>
              </div>
            </div>

            <textarea
              className="input-field textarea-field"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter your script here. Supports English, Hindi (हिन्दी / romanized Hinglish), and Marathi (मराठी)..."
              maxLength={500}
            />
          </div>

          {/* Voice Selection (Preset Mode) */}
          {mode === 'preset' ? (
            <div>
              <label style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>
                Select Voice Model
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {compatibleVoices.map((voice) => (
                  <div
                    key={voice.id}
                    onClick={() => setSelectedVoiceId(voice.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '12px 16px',
                      borderRadius: '12px',
                      background: selectedVoiceId === voice.id ? 'rgba(139, 92, 246, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${selectedVoiceId === voice.id ? 'var(--accent-violet)' : 'var(--border-subtle)'}`,
                      cursor: 'pointer',
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>{voice.name}</span>
                        <span className="badge badge-violet" style={{ fontSize: '0.65rem' }}>Operator Reference</span>
                      </div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                        {voice.description}
                      </p>
                    </div>

                    {voice.has_preview && (
                      <audio controls src={api.getVoicePreviewUrl(voice.id)} style={{ height: '32px', maxWidth: '140px' }} />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            /* Custom Clone Reference Audio Panel */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-cyan-light)', textTransform: 'uppercase' }}>
                Reference Voice Recording (3–30 Seconds)
              </label>

              {/* Live Record or Upload Switcher */}
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
                    <input type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.webm" onChange={handleFileUpload} style={{ display: 'none' }} />
                  </label>
                </div>

                {referenceFile ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#34d399', fontSize: '0.85rem' }}>
                    <CheckCircle2 size={16} />
                    <span>Selected: <strong>{referenceFile.name}</strong> ({Math.round(referenceFile.size / 1024)} KB)</span>
                  </div>
                ) : (
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Provide 10–30 seconds of clean, isolated speech without background noise.
                  </p>
                )}

                {recordedBlobUrl && (
                  <div style={{ marginTop: '12px' }}>
                    <audio src={recordedBlobUrl} controls style={{ height: '32px', width: '100%', maxWidth: '300px' }} />
                  </div>
                )}
              </div>

              {/* Consent Checkbox */}
              <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', fontSize: '0.825rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={consentGiven}
                  onChange={(e) => setConsentGiven(e.target.checked)}
                  style={{ marginTop: '3px', accentColor: 'var(--accent-cyan)' }}
                />
                <span>
                  <Shield size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom' }} />
                  <strong>Voice Authorization Consent:</strong> I confirm that I own or hold explicit authorization to clone and synthesize this speaker's voice.
                </span>
              </label>
            </div>
          )}

          {/* Speed Control Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '0.825rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sliders size={14} /> Speech Speed
              </span>
              <span style={{ fontSize: '0.825rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-violet-light)' }}>
                {speed.toFixed(2)}x
              </span>
            </div>
            <input
              type="range"
              min="0.75"
              max="1.5"
              step="0.05"
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-violet)' }}
            />
          </div>

          {/* Error Banner */}
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

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
            className="btn btn-primary"
            style={{ width: '100%', padding: '14px', fontSize: '1rem' }}
          >
            {isGenerating ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                <span>Synthesizing Voice...</span>
              </>
            ) : (
              <>
                <Play size={18} fill="#fff" />
                <span>Generate Speech ({language.toUpperCase()})</span>
              </>
            )}
          </button>
        </div>

        {/* Right Column: Generation Status & Audio Output Player */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Active Generation Progress Card */}
          {isGenerating && (
            <div className="glass-panel animate-pulse-glow" style={{ padding: '24px', textAlign: 'center' }}>
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
                <Loader2 size={36} color="var(--accent-violet)" className="animate-spin" />
              </div>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '6px' }}>
                Processing Inference Pipeline
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                {currentJob?.progress_message || 'Inference queued in worker daemon...'}
              </p>
            </div>
          )}

          {/* Generated Result Audio Player */}
          {currentJob && currentJob.status === 'completed' && currentJob.output_url ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <AudioPlayer
                src={currentJob.output_url}
                title={`${mode === 'preset' ? 'My Voice' : 'Custom Cloned Voice'} (${language.toUpperCase()})`}
                subtitle={`Generated in ${language.toUpperCase()} with 24kHz studio quality`}
                wavDownloadUrl={currentJob.output_url}
                mp3DownloadUrl={`${currentJob.output_url}?format=mp3`}
                aiAttribution={true}
              />
            </div>
          ) : !isGenerating ? (
            /* Empty State / How it Works Card */
            <div className="glass-panel" style={{ padding: '28px', color: 'var(--text-secondary)' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
                Multilingual Speech Workflow
              </h3>
              <ol style={{ paddingLeft: '20px', fontSize: '0.875rem', lineHeight: 1.8, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <li>
                  <strong>English:</strong> Uses MeloTTS high-fidelity English base acoustic model combined with OpenVoice V2 ToneColorConverter.
                </li>
                <li>
                  <strong>Hindi & Marathi:</strong> Uses Meta MMS-TTS (Massively Multilingual Speech) models with phonetic transliteration and tone conversion.
                </li>
                <li>
                  <strong>Voice Attribution:</strong> Output audio files are automatically injected with C2PA-compliant synthetic speech metadata.
                </li>
                <li>
                  <strong>Privacy & Ephemeral Storage:</strong> Generated files expire after 60 minutes and are permanently purged from server memory.
                </li>
              </ol>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
