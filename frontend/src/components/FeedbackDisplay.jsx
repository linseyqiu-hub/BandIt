export default function FeedbackDisplay({ feedback, tone }) {
  return (
    <div style={{ marginTop: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 14, fontWeight: 500 }}>Examiner feedback</span>
        <span style={{
          fontSize:     11,
          padding:      '3px 8px',
          borderRadius: 6,
          background:   tone === 'strict' ? '#FAECE7' : '#E6F1FB',
          color:        tone === 'strict' ? '#993C1D' : '#185FA5',
        }}>
          {tone === 'strict' ? 'Strict' : 'Coaching'}
        </span>
      </div>
      <p style={{
        fontSize:   14,
        lineHeight: 1.75,
        color:      'var(--text-primary)',
        margin:     0,
      }}>
        {feedback}
      </p>
    </div>
  )
}