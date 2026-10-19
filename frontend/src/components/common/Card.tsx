import React from 'react';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  interactive?: boolean;
  selected?: boolean;
  padding?: 'none' | 'sm' | 'base' | 'lg';
}

export const Card: React.FC<CardProps> = ({
  children,
  interactive = false,
  selected = false,
  padding = 'base',
  className = '',
  style,
  onClick,
  ...props
}) => {
  const paddingMap = {
    none: '0',
    sm: 'var(--space-1)',
    base: 'var(--space-2)',
    lg: 'var(--space-3)',
  };

  const cardStyle: React.CSSProperties = {
    backgroundColor: 'var(--color-surface)',
    border: `1px solid ${selected ? 'var(--color-accent)' : 'var(--color-border)'}`,
    borderRadius: 'var(--radius-surface)',
    padding: paddingMap[padding],
    boxShadow: 'var(--shadow-card)',
    cursor: interactive ? 'pointer' : 'default',
    transition: 'border-color 150ms ease, background-color 150ms ease',
    outline: selected ? '1px solid var(--color-accent)' : 'none',
    position: 'relative',
    ...style,
  };

  return (
    <div
      {...props}
      onClick={onClick}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onKeyDown={
        interactive && onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick(e as unknown as React.MouseEvent<HTMLDivElement>);
              }
            }
          : undefined
      }
      className={`card ${interactive ? 'interactive' : ''} ${className}`}
      style={cardStyle}
    >
      {children}
    </div>
  );
};
