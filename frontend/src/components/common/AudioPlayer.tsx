import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Download, Volume2, VolumeX, Sparkles } from 'lucide-react';

interface AudioPlayerProps {
  src: string;
  title?: string;
  subtitle?: string;
  comparisonSrc?: string;
  comparisonLabel?: string;
  aiAttribution?: boolean;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  src,
  comparisonSrc,
  comparisonLabel = 'Source Audio',
  aiAttribution = true,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [activeTrack, setActiveTrack] = useState<'output' | 'source'>('output');

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const currentSrc = activeTrack === 'output' ? src : comparisonSrc || src;

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
    }
  }, [src, comparisonSrc, activeTrack]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false));
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      if (audioRef.current.duration && !isNaN(audioRef.current.duration)) {
        setDuration(audioRef.current.duration);
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs) || secs < 0) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  // Download URLs
  const wavUrl = activeTrack === 'output' ? src : (comparisonSrc || src);
  const mp3Url = activeTrack === 'output' ? `${src}${src.includes('?') ? '&' : '?'}format=mp3` : (comparisonSrc || src);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
    }}>
      <audio
        ref={audioRef}
        src={currentSrc}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleTimeUpdate}
        onEnded={() => setIsPlaying(false)}
      />

      {/* A/B comparison toggle if source audio exists */}
      {comparisonSrc && (
        <div style={{ display: 'flex', gap: '6px', marginBottom: '2px' }}>
          <button
            onClick={() => setActiveTrack('output')}
            className={`btn-secondary ${activeTrack === 'output' ? 'active' : ''}`}
            style={{
              padding: '5px 12px',
              fontSize: '0.8125rem',
              backgroundColor: activeTrack === 'output' ? 'var(--accent-green)' : '#ffffff',
              color: activeTrack === 'output' ? '#ffffff' : 'var(--text-secondary)',
              borderColor: activeTrack === 'output' ? 'var(--accent-green)' : 'var(--border-medium)',
            }}
          >
            Converted Voice
          </button>
          <button
            onClick={() => setActiveTrack('source')}
            className={`btn-secondary ${activeTrack === 'source' ? 'active' : ''}`}
            style={{
              padding: '5px 12px',
              fontSize: '0.8125rem',
              backgroundColor: activeTrack === 'source' ? 'var(--accent-green)' : '#ffffff',
              color: activeTrack === 'source' ? '#ffffff' : 'var(--text-secondary)',
              borderColor: activeTrack === 'source' ? 'var(--accent-green)' : 'var(--border-medium)',
            }}
          >
            {comparisonLabel}
          </button>
        </div>
      )}

      {/* Dark sleek waveform box matching Page 2 mockup */}
      <div style={{
        backgroundColor: '#111815',
        borderRadius: 'var(--radius-lg)',
        padding: '24px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        border: '1px solid #23312a',
      }}>
        {/* Animated/Static Waveform visualization bars */}
        <div style={{
          height: '52px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '3px',
          padding: '0 4px',
          overflow: 'hidden',
        }}>
          {Array.from({ length: 48 }).map((_, i) => {
            const progress = duration > 0 ? currentTime / duration : 0;
            const barProgress = i / 48;
            const isPlayed = barProgress <= progress;
            // Simulated waveform height variation
            const baseHeights = [
              20, 35, 60, 45, 75, 90, 65, 80, 50, 40, 70, 85,
              95, 80, 60, 40, 55, 75, 90, 85, 70, 50, 65, 80,
              60, 40, 50, 75, 85, 70, 60, 45, 65, 85, 90, 75,
              55, 40, 60, 70, 80, 65, 50, 40, 55, 70, 45, 30
            ];
            const heightPct = baseHeights[i % baseHeights.length];

            return (
              <div
                key={i}
                style={{
                  flex: 1,
                  height: `${heightPct}%`,
                  borderRadius: '2px',
                  backgroundColor: isPlayed ? '#2dd4bf' : '#223830',
                  transition: 'background-color 0.15s ease',
                }}
              />
            );
          })}
        </div>

        {/* Playback Controls & Time */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          color: '#ffffff',
          fontSize: '0.875rem',
        }}>
          <button
            onClick={togglePlay}
            style={{
              border: 'none',
              background: 'transparent',
              color: '#ffffff',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontFamily: 'inherit',
              fontWeight: 600,
              fontSize: '0.9375rem',
            }}
          >
            {isPlaying ? <Pause size={18} fill="#ffffff" /> : <Play size={18} fill="#ffffff" />}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </button>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.8125rem',
            color: '#a3b8b0',
          }}>
            <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
            <button
              onClick={toggleMute}
              style={{
                border: 'none',
                background: 'transparent',
                color: '#a3b8b0',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
              }}
              title={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            </button>
          </div>
        </div>

        {/* Hidden/Subtle seek track */}
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.05}
          value={currentTime}
          onChange={handleSeek}
          style={{
            width: '100%',
            height: '4px',
            accentColor: '#2dd4bf',
            cursor: 'pointer',
            opacity: 0.8,
          }}
        />
      </div>

      {/* AI Attribution Badge */}
      {aiAttribution && (
        <div>
          <span className="badge-ai-generated">
            <Sparkles size={13} />
            <span>AI-generated voice</span>
          </span>
        </div>
      )}

      {/* Download Buttons */}
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <a
          href={wavUrl}
          download="synthesized_speech.wav"
          className="btn-secondary"
          style={{ textDecoration: 'none', flex: 1 }}
        >
          <Download size={15} />
          <span>Download WAV</span>
        </a>
        <a
          href={mp3Url}
          download="synthesized_speech.mp3"
          className="btn-secondary"
          style={{ textDecoration: 'none', flex: 1 }}
        >
          <Download size={15} />
          <span>Download MP3</span>
        </a>
      </div>
    </div>
  );
};
