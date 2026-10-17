/**
 * Typed API client for AI Voice Studio backend.
 * Provides complete bindings for catalog, generation, polling, and text enhancement endpoints.
 */

import type {
  HealthResponse,
  PublicConfigResponse,
  VoiceListResponse,
  Voice,
  JobCreatedResponse,
  JobStatusResponse,
  TextCleanupResponse,
} from './types';

const API_BASE = '/v1';

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(message: string, code: string = 'API_ERROR', status: number = 500) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMsg = `Server request failed with status ${res.status}`;
    let errorCode = String(res.status);

    try {
      const errorJson = await res.json();
      if (errorJson.error) {
        errorMsg = errorJson.error.message || errorMsg;
        errorCode = errorJson.error.code || errorCode;
      } else if (errorJson.detail) {
        errorMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Body not JSON
    }

    throw new ApiError(errorMsg, errorCode, res.status);
  }

  return res.json() as Promise<T>;
}

export const api = {
  /**
   * Check backend server health and uptime.
   */
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<HealthResponse>(res);
  },

  /**
   * Retrieve server configuration limits and public parameters.
   */
  async getPublicConfig(): Promise<PublicConfigResponse> {
    const res = await fetch(`${API_BASE}/config/public`);
    return handleResponse<PublicConfigResponse>(res);
  },

  /**
   * Fetch all registered voices from the voice catalog.
   */
  async getVoices(): Promise<VoiceListResponse> {
    const res = await fetch(`${API_BASE}/voices`);
    return handleResponse<VoiceListResponse>(res);
  },

  /**
   * Fetch profile details for a specific voice.
   */
  async getVoice(voiceId: string): Promise<Voice> {
    const res = await fetch(`${API_BASE}/voices/${encodeURIComponent(voiceId)}`);
    return handleResponse<Voice>(res);
  },

  /**
   * Get direct preview streaming URL for a voice.
   */
  getVoicePreviewUrl(voiceId: string): string {
    return `${API_BASE}/voices/${encodeURIComponent(voiceId)}/preview`;
  },

  /**
   * Queue a standard Text-to-Speech synthesis job.
   */
  async createTTSJob(params: {
    text: string;
    voiceId: string;
    language?: string;
    speed?: number;
  }): Promise<JobCreatedResponse> {
    const res = await fetch(`${API_BASE}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: params.text,
        voice_id: params.voiceId,
        language: params.language || 'en',
        speed: params.speed ?? 1.0,
      }),
    });
    return handleResponse<JobCreatedResponse>(res);
  },

  /**
   * Queue a custom voice cloning Text-to-Speech job with user reference audio.
   */
  async createCustomTTSJob(params: {
    text: string;
    file: File | Blob;
    language?: string;
    speed?: number;
    consent: boolean;
  }): Promise<JobCreatedResponse> {
    const formData = new FormData();
    formData.append('text', params.text);
    formData.append('language', params.language || 'en');
    formData.append('speed', String(params.speed ?? 1.0));
    formData.append('consent', String(params.consent));
    formData.append('file', params.file, 'reference.wav');

    const res = await fetch(`${API_BASE}/tts/custom`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<JobCreatedResponse>(res);
  },

  /**
   * Queue an audio-to-audio Voice Conversion job.
   */
  async createVCJob(params: {
    voiceId: string;
    file: File | Blob;
    pitchShift?: number;
  }): Promise<JobCreatedResponse> {
    const formData = new FormData();
    formData.append('voice_id', params.voiceId);
    formData.append('pitch_shift', String(params.pitchShift ?? 0));
    formData.append('file', params.file, 'source.wav');

    const res = await fetch(`${API_BASE}/vc`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<JobCreatedResponse>(res);
  },

  /**
   * Clean and normalize script text for TTS synthesis.
   */
  async cleanupText(text: string, language: string = 'en'): Promise<TextCleanupResponse> {
    const res = await fetch(`${API_BASE}/text/cleanup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
    });
    return handleResponse<TextCleanupResponse>(res);
  },

  /**
   * Fetch current status and progress of an asynchronous job.
   */
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
    return handleResponse<JobStatusResponse>(res);
  },

  /**
   * Delete job artifacts and database record.
   */
  async deleteJob(jobId: string): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
    });
    return handleResponse<{ status: string }>(res);
  },

  /**
   * Helper to poll a job until completion or failure with backoff.
   */
  async pollJob(
    jobId: string,
    onProgress?: (job: JobStatusResponse) => void,
    intervalMs: number = 1000,
    timeoutMs: number = 120000
  ): Promise<JobStatusResponse> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const status = await this.getJobStatus(jobId);
      if (onProgress) {
        onProgress(status);
      }

      if (status.status === 'completed' || status.status === 'failed' || status.status === 'expired') {
        return status;
      }

      await new Promise((r) => setTimeout(r, intervalMs));
    }

    throw new ApiError('Job polling timed out before generation completed.', 'TIMEOUT', 408);
  },
};
