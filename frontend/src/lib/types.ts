/**
 * TypeScript data contracts and schema definitions for AI Voice Studio frontend.
 */

export type ModelType = 'openvoice_reference' | 'rvc_model';
export type VoiceStatus = 'available' | 'disabled' | 'experimental';
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

export interface JobStatusResponse {
  job_id: string;
  kind: 'tts' | 'tts_custom' | 'vc';
  status: JobStatus;
  voice_id?: string | null;
  progress_message?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  expires_at: string;
  output_url?: string | null;
  source_url?: string | null;
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
  version: string;
  allowed_languages: string[];
  max_upload_mb: number;
  max_audio_seconds: number;
  max_text_chars: number;
  job_ttl_minutes: number;
  openvoice_watermark_enabled: boolean;
}

export type ActiveTab = 'tts' | 'vc' | 'catalog' | 'history';
