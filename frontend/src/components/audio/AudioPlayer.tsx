import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '../common/Button';

export interface AudioPlayerProps {
  src: string;
  title?: string;
  autoPlay?: boolean;
  onEnded?: () => void;
  compact?: boolean;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  src,
  title,
  autoPlay = false,
  onEnded,
  compact = false,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Keep playback rate synced
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  // Audio event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadStart = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      setHasError(false);
      setErrorMessage(null);
    };
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => {
      if (isFinite(audio.duration) && audio.duration > 0) {
        setDuration(audio.duration);
      }
      setHasError(false);
    };
    const handleDurationChange = () => {
      if (isFinite(audio.duration) && audio.duration > 0) {
        setDuration(audio.duration);
      }
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      if (onEnded) onEnded();
    };
    const handleError = () => {
      setIsPlaying(false);
      setHasError(true);
      setErrorMessage('Audio format unsupported or unavailable.');
    };

    audio.addEventListener('loadstart', handleLoadStart);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('durationchange', handleDurationChange);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);

    return () => {
      audio.removeEventListener('loadstart', handleLoadStart);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('durationchange', handleDurationChange);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
    };
  }, [src, onEnded]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      if (duration > 0 && currentTime >= duration) {
        audio.currentTime = 0;
      }
      audio.play().catch((err) => {
        setHasError(true);
        setErrorMessage(err?.message || 'Playback blocked or failed.');
      });
    }
  }, [isPlaying, currentTime, duration]);

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackRate(speed);
    if (audioRef.current) {
      audioRef.current.playbackRate = speed;
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs) || secs < 0 || !isFinite(secs)) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: compact ? '4px' : 'var(--space-1)',
        backgroundColor: 'var(--color-surface-muted)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-control)',
        padding: compact ? '6px 10px' : 'var(--space-1) var(--space-2)',
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <audio
        ref={audioRef}
        src={src}
        autoPlay={autoPlay}
        preload="metadata"
      />

      {title && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 'var(--font-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            {title}
          </span>
          {duration > 0 && (
            <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
              {formatTime(duration)}
            </span>
          )}
        </div>
      )}

      {hasError ? (
        <div
          style={{
            fontSize: 'var(--font-sm)',
            color: 'var(--color-error)',
            backgroundColor: 'var(--color-error-bg)',
            padding: '4px 8px',
            borderRadius: 'var(--radius-control)',
            border: '1px solid var(--color-error-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span>{errorMessage || 'Audio could not be played.'}</span>
          <button
            type="button"
            onClick={() => {
              setHasError(false);
              audioRef.current?.load();
            }}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-error)',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '11px',
              textDecoration: 'underline',
            }}
          >
            Retry
          </button>
        </div>
      ) : (
        /* Controls Bar */
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
          <Button
            size="sm"
            variant="primary"
            onClick={togglePlay}
            aria-label={isPlaying ? 'Pause' : 'Play'}
            style={{
              width: compact ? '32px' : '36px',
              minWidth: compact ? '32px' : '36px',
              height: compact ? '30px' : '34px',
              minHeight: compact ? '30px' : '34px',
              padding: 0,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {isPlaying ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </Button>

          {/* Scrubber & Time */}
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                fontSize: '11px',
                fontVariantNumeric: 'tabular-nums',
                color: 'var(--color-text-muted)',
                minWidth: '28px',
              }}
            >
              {formatTime(currentTime)}
            </span>
            <input
              type="range"
              min={0}
              max={duration > 0 ? duration : 1}
              step={0.05}
              value={currentTime}
              onChange={handleSeek}
              aria-label="Seek audio position"
              style={{
                flex: 1,
                accentColor: 'var(--color-accent)',
                cursor: 'pointer',
                height: '4px',
              }}
            />
            <span
              style={{
                fontSize: '11px',
                fontVariantNumeric: 'tabular-nums',
                color: 'var(--color-text-muted)',
                minWidth: '28px',
                textAlign: 'right',
              }}
            >
              {formatTime(duration)}
            </span>
          </div>

          {/* Speed Selector (non-compact) */}
          {!compact && (
            <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
              {[0.75, 1, 1.25, 1.5].map((speed) => (
                <button
                  key={speed}
                  type="button"
                  onClick={() => handleSpeedChange(speed)}
                  style={{
                    fontSize: '11px',
                    padding: '2px 5px',
                    borderRadius: 'var(--radius-control)',
                    border: '1px solid var(--color-border)',
                    backgroundColor: playbackRate === speed ? 'var(--color-accent)' : 'var(--color-surface)',
                    color: playbackRate === speed ? 'var(--color-text-on-accent)' : 'var(--color-text-secondary)',
                    cursor: 'pointer',
                    fontWeight: playbackRate === speed ? 600 : 400,
                  }}
                >
                  {speed}x
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
