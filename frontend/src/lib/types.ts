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
  duration_seconds: number;
  speech_duration_seconds: number;
  speech_ratio: number;
  rms_db: number;
  peak_amplitude: number;
  snr_db: number;
  clipping_ratio: number;
  reverb_score: number;
  speaker_consistency_score: number;
  music_detected: boolean;
  music_score: number;
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
