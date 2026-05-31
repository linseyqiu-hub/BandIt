export default function ErrorBanner({ message }) {
  if (!message) return null
  return (
    <div style={{
      display:     'flex',
      alignItems:  'flex-start',
      gap:         10,
      background:  '#FCEBEB',
      border:      '0.5px solid #F09595',
      borderRadius: 8,
      padding:     '12px 14px',
      marginTop:   '1rem',
      animation:   'fadeIn 0.2s ease',
    }}>
      <span style={{ fontSize: 16, color: '#A32D2D', marginTop: 1, flexShrink: 0 }}>⚠</span>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: '#A32D2D', marginBottom: 2 }}>
          Something went wrong
        </div>
        <div style={{ fontSize: 12, color: '#791F1F' }}>{message}</div>
      </div>
    </div>
  )
}