/**
 * TypeScript data contracts and schema definitions for AI Voice Studio frontend.
 */

export type ModelType = 'chatterbox_reference';
export type VoiceStatus = 'available' | 'ready' | 'disabled' | 'experimental';
export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'expired';
export type SupportedLanguage = 'en' | 'hi' | 'mr';

export interface Voice {
  id: string;
  name: string;
  description: string;
  style?: string;
  languages: string[];
  model_type: ModelType;
  has_preview: boolean;
  preview_url?: string;
  license: string;
  usage_permission: string;
  status: VoiceStatus;
  limitations?: string;
}

export interface VoiceListResponse {
  voices: Voice[];
  total: number;
}

export interface JobCreatedResponse {
  job_id: string;
  kind: 'tts' | 'tts_custom' | 'vc';
  status: 'queued';
  message: string;
  ai_generated: boolean;
}

export interface AudioFileInfo {
  url: string;
  size_bytes?: number;
  hint?: string;
}

export interface OutputFilesMap {
  wav?: AudioFileInfo;
  mp3?: AudioFileInfo;
  mp3_small?: AudioFileInfo;
}

export interface JobStatusResponse {
  job_id: string;
  kind: 'tts' | 'tts_custom' | 'vc' | 'preview';
  status: JobStatus;
  voice_id?: string | null;
  progress_message?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  expires_at: string;
  stage?: string | null;
  stage_index?: number | null;
  stage_total?: number | null;
  output_url?: string | null;
  source_url?: string | null;
  chunk_done?: number | null;
  chunk_total?: number | null;
  eta_seconds?: number | null;
  output_files?: OutputFilesMap | null;
  confidence_label?: string | null;
}

export interface ReferenceCheckMetrics {
  speech_length_s: number;
  total_duration_s: number;
  loudness_dbfs: number;
  snr_db: number;
  clipping_ratio: number;
  echo_reverb: string;
  speaker_consistency: number;
  multi_speaker_detected: boolean;
  music_detected: boolean;
  duration_seconds?: number;
  speech_duration_seconds?: number;
  speech_ratio?: number;
  rms_db?: number;
}

export interface ReferenceCheckResponse {
  score: number;
  overall_score?: number;
  is_usable: boolean;
  advice: string[];
  warnings?: string[];
  hard_block_reasons?: string[];
  metrics: ReferenceCheckMetrics;
}

export interface TTSPreviewResponse {
  job_id?: string;
  preview_url: string;
  audio_url?: string;
  duration_seconds: number;
  text_preview: string;
  confidence_label?: 'good' | 'fair' | 'low';
}

export interface TextCleanupResponse {
  original: string;
  cleaned: string;
  provider_used: string;
  service_available: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  time: string;
}

export interface PublicConfigResponse {
  max_text_chars: number;
  job_ttl_minutes: number;
  queue_max_size: number;
  job_timeout_seconds: number;
  max_audio_seconds: number;
  max_upload_mb: number;
  expose_experimental_voices: boolean;
  seconds_per_audio_second?: number;
  compute_seconds_per_audio_second?: number;
  features: Record<string, boolean>;
}

export type ActiveTab = 'start' | 'tts' | 'vc';

/**
 * Accurately count words across Unicode, Devanagari (Hindi, Marathi), English, numbers, and mixed text.
 * Punctuation marks (.,!?-।॥) are not counted as words.
 */
export const countWordsAccurately = (text: string): number => {
  if (!text || !text.trim()) return 0;
  const tokens = text.match(/[\p{L}\p{M}\p{N}]+(?:['’\-–—][\p{L}\p{M}\p{N}]+)*/gu);
  return tokens ? tokens.length : 0;
};

/**
 * Calculate estimated speech duration using selected language and documented speaking-rate assumptions:
 * - English ('en'): ~150 WPM (~2.5 words/sec), ~15 chars/sec
 * - Hindi ('hi'): ~130 WPM (~2.17 words/sec), ~12.5 chars/sec
 * - Marathi ('mr'): ~125 WPM (~2.08 words/sec), ~12 chars/sec
 */
export const estimateSpeechDuration = (
  wordCount: number,
  charCount: number,
  language: string
): number => {
  if (wordCount === 0 || charCount === 0) return 0;

  let wordsPerSec = 2.17; // Hindi default (~130 WPM)
  let charsPerSec = 12.5;

  if (language === 'en') {
    wordsPerSec = 2.5; // ~150 WPM
    charsPerSec = 15.0;
  } else if (language === 'mr') {
    wordsPerSec = 2.08; // ~125 WPM
    charsPerSec = 12.0;
  }

  const byWords = wordCount / wordsPerSec;
  const byChars = charCount / charsPerSec;

  const estSeconds = Math.round(byWords * 0.75 + byChars * 0.25);
  return Math.max(1, estSeconds);
};

export const formatDuration = (secs: number): string => {
  if (secs <= 0) return '0s';
  if (secs < 60) return `~${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return s > 0 ? `~${m}m ${s}s` : `~${m}m`;
};

