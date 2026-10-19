import React from 'react';

export interface StepItem {
  number: number;
  label: string;
}

const STEPS: StepItem[] = [
  { number: 1, label: 'Script' },
  { number: 2, label: 'Voice' },
  { number: 3, label: 'Preview' },
  { number: 4, label: 'Full audio' },
];

export interface StepperProps {
  currentStep: number;
  completedSteps: number[];
  onSelectStep: (stepNumber: number) => void;
}

export const Stepper: React.FC<StepperProps> = ({
  currentStep,
  completedSteps,
  onSelectStep,
}) => {
  return (
    <nav
      aria-label="Wizard Steps"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
        padding: '0 var(--space-2)',
        backgroundColor: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        minHeight: '44px',
        flexShrink: 0,
      }}
    >
      <ol
        style={{
          display: 'flex',
          alignItems: 'center',
          listStyle: 'none',
          width: '100%',
          justifyContent: 'space-between',
          margin: 0,
          padding: 0,
        }}
      >
        {STEPS.map((s, idx) => {
          const isCurrent = s.number === currentStep;
          const isCompleted = completedSteps.includes(s.number);
          const isClickable = isCompleted || s.number <= currentStep;

          return (
            <React.Fragment key={s.number}>
              <li style={{ display: 'flex', alignItems: 'center' }}>
                <button
                  type="button"
                  disabled={!isClickable}
                  onClick={() => isClickable && onSelectStep(s.number)}
                  aria-current={isCurrent ? 'step' : undefined}
                  className="touch-target"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 'var(--space-1)',
                    background: 'none',
                    border: 'none',
                    cursor: isClickable ? 'pointer' : 'default',
                    opacity: isClickable ? 1 : 0.45,
                    padding: 'var(--space-0-5) var(--space-1)',
                    borderRadius: 'var(--radius-control)',
                    color: isCurrent
                      ? 'var(--color-accent)'
                      : isCompleted
                      ? 'var(--color-text-primary)'
                      : 'var(--color-text-muted)',
                    fontFamily: 'var(--font-sans)',
                    fontSize: 'var(--font-base)',
                    fontWeight: isCurrent ? 700 : isCompleted ? 600 : 400,
                  }}
                >
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '24px',
                      height: '24px',
                      borderRadius: '50%',
                      backgroundColor: isCurrent
                        ? 'var(--color-accent)'
                        : isCompleted
                        ? 'var(--color-accent-light)'
                        : 'var(--color-surface-muted)',
                      color: isCurrent
                        ? 'var(--color-text-on-accent)'
                        : isCompleted
                        ? 'var(--color-accent)'
                        : 'var(--color-text-muted)',
                      border: `1px solid ${
                        isCurrent
                          ? 'var(--color-accent)'
                          : isCompleted
                          ? 'var(--color-accent-border)'
                          : 'var(--color-border)'
                      }`,
                      fontSize: 'var(--font-sm)',
                      fontWeight: 700,
                    }}
                  >
                    {isCompleted && !isCurrent ? '✓' : s.number}
                  </span>
                  <span style={{ whiteSpace: 'nowrap' }}>{s.label}</span>
                </button>
              </li>

              {idx < STEPS.length - 1 && (
                <li
                  aria-hidden="true"
                  style={{
                    flex: 1,
                    height: '1px',
                    backgroundColor: isCompleted ? 'var(--color-accent-border)' : 'var(--color-border)',
                    margin: '0 var(--space-1)',
                    minWidth: '8px',
                  }}
                />
              )}
            </React.Fragment>
          );
        })}
      </ol>
    </nav>
  );
};
