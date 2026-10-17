import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Download, Volume2, VolumeX, ShieldCheck } from 'lucide-react';

interface AudioPlayerProps {
  src: string;
  title?: string;
  subtitle?: string;
  aiAttribution?: boolean;
  wavDownloadUrl?: string;
  mp3DownloadUrl?: string;
  comparisonSrc?: string;
  comparisonLabel?: string;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  src,
  title = 'Synthesized Audio',
  subtitle = 'Neural generation output',
  aiAttribution = true,
  wavDownloadUrl,
  mp3DownloadUrl,
  comparisonSrc,
  comparisonLabel = 'Original Audio',
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeTrack, setActiveTrack] = useState<'output' | 'source'>('output');
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const currentSrc = activeTrack === 'output' ? src : comparisonSrc || src;

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.load();
    }
  }, [currentSrc]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration || 0);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    setIsMuted(val === 0);
    if (audioRef.current) {
      audioRef.current.volume = val;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    const newMuted = !isMuted;
    setIsMuted(newMuted);
    audioRef.current.muted = newMuted;
  };

  const cycleSpeed = () => {
    const speeds = [0.8, 1.0, 1.25, 1.5];
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length;
    const nextRate = speeds[nextIdx];
    setPlaybackRate(nextRate);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextRate;
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs)) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="glass-panel" style={{ padding: '20px', borderRadius: '16px' }}>
      <audio
        ref={audioRef}
        src={currentSrc}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleTimeUpdate}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Header with Title and AI Attribution Badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            {activeTrack === 'output' ? title : `${comparisonLabel}`}
          </h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{subtitle}</p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {comparisonSrc && (
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.4)', borderRadius: '8px', padding: '2px', border: '1px solid var(--border-subtle)' }}>
              <button
                onClick={() => setActiveTrack('output')}
                className="btn btn-sm"
                style={{
                  background: activeTrack === 'output' ? 'var(--accent-violet)' : 'transparent',
                  color: activeTrack === 'output' ? '#fff' : 'var(--text-muted)',
                  fontSize: '0.75rem',
                  padding: '4px 8px',
                }}
              >
                Output Voice
              </button>
              <button
                onClick={() => setActiveTrack('source')}
                className="btn btn-sm"
                style={{
                  background: activeTrack === 'source' ? 'var(--accent-cyan)' : 'transparent',
                  color: activeTrack === 'source' ? '#fff' : 'var(--text-muted)',
                  fontSize: '0.75rem',
                  padding: '4px 8px',
                }}
              >
                Source
              </button>
            </div>
          )}

          {aiAttribution && (
            <span className="badge badge-ai" title="Watermarked synthetic speech">
              <ShieldCheck size={13} color="#a78bfa" />
              <span>AI-Generated</span>
            </span>
          )}
        </div>
      </div>

      {/* Dynamic Simulated Waveform Bars */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '3px',
        height: '42px',
        padding: '6px 12px',
        background: 'rgba(10, 12, 18, 0.7)',
        borderRadius: '10px',
        marginBottom: '14px',
        border: '1px solid rgba(255, 255, 255, 0.05)',
      }}>
        {Array.from({ length: 36 }).map((_, idx) => {
          const progress = duration > 0 ? currentTime / duration : 0;
          const barProgress = idx / 36;
          const isPassed = barProgress <= progress;
          // Deterministic synthetic waveform heights
          const baseHeight = Math.sin((idx / 36) * Math.PI) * 26 + 6;
          const height = Math.max(4, Math.round(baseHeight * (isPlaying && isPassed ? 1.15 : 0.85)));

          return (
            <div
              key={idx}
              style={{
                flex: 1,
                height: `${height}px`,
                backgroundColor: isPassed
                  ? activeTrack === 'output'
                    ? 'var(--accent-violet)'
                    : 'var(--accent-cyan)'
                  : 'rgba(255, 255, 255, 0.15)',
                borderRadius: '3px',
                transition: 'height 0.15s ease, background-color 0.2s ease',
              }}
            />
          );
        })}
      </div>

      {/* Scrubber Progress Slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px' }}>
        <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', minWidth: '38px' }}>
          {formatTime(currentTime)}
        </span>
        <input
          type="range"
          min="0"
          max={duration || 100}
          step="0.05"
          value={currentTime}
          onChange={handleSeek}
          style={{
            flex: 1,
            accentColor: activeTrack === 'output' ? 'var(--accent-violet)' : 'var(--accent-cyan)',
            cursor: 'pointer',
          }}
        />
        <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', minWidth: '38px', textAlign: 'right' }}>
          {formatTime(duration)}
        </span>
      </div>

      {/* Bottom Controls Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        {/* Playback Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={togglePlay}
            className="btn btn-primary"
            style={{
              width: '44px',
              height: '44px',
              padding: 0,
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {isPlaying ? <Pause size={20} fill="#fff" /> : <Play size={20} fill="#fff" style={{ marginLeft: '2px' }} />}
          </button>

          {/* Speed Toggle */}
          <button
            onClick={cycleSpeed}
            className="btn btn-secondary btn-sm"
            title="Playback speed"
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', padding: '6px 10px' }}
          >
            {playbackRate}x
          </button>

          {/* Volume Control */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button onClick={toggleMute} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex' }}>
              {isMuted || volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              style={{ width: '60px', accentColor: 'var(--accent-violet)', cursor: 'pointer' }}
            />
          </div>
        </div>

        {/* Download Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {wavDownloadUrl && (
            <a
              href={wavDownloadUrl}
              download="synthesized_speech.wav"
              className="btn btn-secondary btn-sm"
              title="Download lossless 24kHz PCM WAV"
            >
              <Download size={14} />
              <span>WAV</span>
            </a>
          )}
          {mp3DownloadUrl && (
            <a
              href={mp3DownloadUrl}
              download="synthesized_speech.mp3"
              className="btn btn-secondary btn-sm"
              title="Download 192kbps MP3"
            >
              <Download size={14} />
              <span>MP3</span>
            </a>
          )}
        </div>
      </div>
    </div>
  );
};
