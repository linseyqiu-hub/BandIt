export default function EssayForm({
  question, setQuestion,
  essay,    setEssay,
  onSubmit,
  loading,
  fieldError,
}) {
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

        {/* Question */}
        <div>
          <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>Question</div>
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Paste the IELTS question here..."
            style={{
              width:        '100%',
              height:       160,
              background:   '#fff',
              border:       '0.5px solid #e0e0e0',
              borderRadius: 8,
              padding:      '10px 12px',
              fontSize:     14,
              fontFamily:   'inherit',
              resize:       'none',
              outline:      'none',
            }}
          />
        </div>

        {/* Essay */}
        <div>
          <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>Essay</div>
          <textarea
            value={essay}
            onChange={e => setEssay(e.target.value)}
            placeholder="Paste your essay here..."
            style={{
              width:        '100%',
              height:       160,
              background:   '#fff',
              border:       `0.5px solid ${fieldError ? '#E24B4A' : '#e0e0e0'}`,
              borderRadius: 8,
              padding:      '10px 12px',
              fontSize:     14,
              fontFamily:   'inherit',
              resize:       'none',
              outline:      'none',
              transition:   'border-color 0.2s',
            }}
          />
          {/* field-level error */}
          {fieldError && (
            <div style={{
              fontSize:   12,
              color:      '#A32D2D',
              marginTop:  4,
              display:    'flex',
              alignItems: 'center',
              gap:        4,
              animation:  'fadeIn 0.2s ease',
            }}>
              ⚠ {fieldError}
            </div>
          )}
        </div>

      </div>

      <button
        onClick={onSubmit}
        disabled={loading}
        style={{
          width:        '100%',
          marginTop:    '1.25rem',
          padding:      11,
          background:   loading ? '#9991d4' : '#534AB7',
          border:       'none',
          borderRadius: 8,
          color:        '#fff',
          fontSize:     14,
          fontWeight:   500,
          cursor:       loading ? 'not-allowed' : 'pointer',
          transition:   'background 0.2s',
        }}
      >
        {loading ? 'Scoring...' : 'Score my essay'}
      </button>
    </div>
  )
}