import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'base';
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'secondary',
  size = 'base',
  loading = false,
  disabled = false,
  icon,
  className = '',
  style,
  ...props
}) => {
  const isDisabled = disabled || loading;

  const baseStyles: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 'var(--space-1)',
    fontFamily: 'var(--font-sans)',
    fontSize: size === 'sm' ? 'var(--font-sm)' : 'var(--font-base)',
    fontWeight: 500,
    borderRadius: 'var(--radius-control)',
    border: '1px solid transparent',
    cursor: isDisabled ? 'not-allowed' : 'pointer',
    opacity: isDisabled ? 0.6 : 1,
    minHeight: '44px',
    padding: size === 'sm' ? '0 var(--space-1)' : '0 var(--space-2)',
    transition: 'background-color 150ms ease, border-color 150ms ease',
    textDecoration: 'none',
    whiteSpace: 'nowrap',
    userSelect: 'none',
    flexShrink: 0,
    ...style,
  };

  const variantStyles: Record<string, React.CSSProperties> = {
    primary: {
      backgroundColor: 'var(--color-accent)',
      color: 'var(--color-text-on-accent)',
      borderColor: 'var(--color-accent)',
    },
    secondary: {
      backgroundColor: 'var(--color-surface)',
      color: 'var(--color-text-primary)',
      borderColor: 'var(--color-border)',
    },
    ghost: {
      backgroundColor: 'transparent',
      color: 'var(--color-text-secondary)',
      borderColor: 'transparent',
    },
    danger: {
      backgroundColor: 'var(--color-error-bg)',
      color: 'var(--color-error)',
      borderColor: 'var(--color-error-border)',
    },
  };

  const currentVariantStyle = variantStyles[variant] || variantStyles.secondary;

  return (
    <button
      {...props}
      disabled={isDisabled}
      className={`touch-target ${className}`}
      style={{
        ...baseStyles,
        ...currentVariantStyle,
      }}
    >
      {loading ? (
        <span
          style={{
            display: 'inline-block',
            width: '14px',
            height: '14px',
            border: '2px solid currentColor',
            borderTopColor: 'transparent',
            borderRadius: '50%',
            animation: 'spin 0.6s linear infinite',
          }}
          aria-hidden="true"
        />
      ) : (
        icon && <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>
      )}
      <span>{children}</span>
    </button>
  );
};
