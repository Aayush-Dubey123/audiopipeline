import React, { useEffect } from 'react';

export interface ToastProps {
  message: string | null;
  type?: 'info' | 'success' | 'error';
  onClose: () => void;
  durationMs?: number;
}

export const Toast: React.FC<ToastProps> = ({
  message,
  type = 'info',
  onClose,
  durationMs = 4000,
}) => {
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => {
      onClose();
    }, durationMs);
    return () => clearTimeout(timer);
  }, [message, durationMs, onClose]);

  if (!message) return null;

  const bgMap = {
    info: 'var(--color-surface)',
    success: 'var(--color-success-bg)',
    error: 'var(--color-error-bg)',
  };

  const borderMap = {
    info: 'var(--color-border)',
    success: 'var(--color-success-border)',
    error: 'var(--color-error-border)',
  };

  const textMap = {
    info: 'var(--color-text-primary)',
    success: 'var(--color-success)',
    error: 'var(--color-error)',
  };

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: 'fixed',
        bottom: 'var(--space-2)',
        right: 'var(--space-2)',
        zIndex: 9999,
        backgroundColor: bgMap[type],
        border: `1px solid ${borderMap[type]}`,
        borderRadius: 'var(--radius-control)',
        padding: 'var(--space-1) var(--space-2)',
        boxShadow: '0 4px 12px rgba(18, 30, 23, 0.15)',
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-1)',
        maxWidth: '360px',
      }}
    >
      <span style={{ fontSize: 'var(--font-sm)', color: textMap[type], fontWeight: 500 }}>
        {message}
      </span>
      <button
        onClick={onClose}
        style={{
          background: 'none',
          border: 'none',
          color: textMap[type],
          cursor: 'pointer',
          padding: '2px',
          fontSize: '16px',
          lineHeight: 1,
        }}
        aria-label="Close notification"
      >
        ×
      </button>
    </div>
  );
};
