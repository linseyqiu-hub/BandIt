const autoResize = (e) => {
  e.target.style.height = 'auto'
  e.target.style.height = e.target.scrollHeight + 'px'
}

const textareaStyle = (hasError) => ({
  width:        '100%',
  height:       'auto',
  minHeight:    80,
  background:   '#fff',
  border:       `0.5px solid ${hasError ? '#E24B4A' : '#e0e0e0'}`,
  borderRadius: 8,
  padding:      '10px 12px',
  fontSize:     14,
  fontFamily:   'inherit',
  lineHeight:   1.6,
  resize:       'none',
  overflow:     'hidden',
  outline:      'none',
  transition:   'border-color 0.2s',
  boxSizing:    'border-box',
})

export default function EssayForm({
  question,   setQuestion,
  essay,      setEssay,
  tone,       setTone,
  onSubmit,
  loading,
  fieldError,
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

      <div>
        <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>Question</div>
        <textarea
          value={question}
          placeholder="Paste the IELTS question here..."
          style={textareaStyle(false)}
          onChange={e => { setQuestion(e.target.value); autoResize(e) }}
          onInput={autoResize}
        />
      </div>

      <div>
        <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>Essay</div>
        <textarea
          value={essay}
          placeholder="Paste your essay here..."
          style={textareaStyle(!!fieldError)}
          onChange={e => { setEssay(e.target.value); autoResize(e) }}
          onInput={autoResize}
        />
        {fieldError && (
          <div style={{
            fontSize:   12,
            color:      '#A32D2D',
            marginTop:  4,
            display:    'flex',
            alignItems: 'center',
            gap:        4,
          }}>
            ⚠ {fieldError}
          </div>
        )}
      </div>

      <div style={{
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 13, color: '#888' }}>Feedback tone</span>
        <div style={{
          display:      'flex',
          border:       '0.5px solid #e0e0e0',
          borderRadius: 8,
          overflow:     'hidden',
        }}>
          {['coaching', 'strict'].map(t => (
            <button
              key={t}
              onClick={() => setTone(t)}
              style={{
                padding:    '6px 16px',
                fontSize:   13,
                cursor:     'pointer',
                border:     'none',
                background: tone === t ? '#f0f0f0' : 'transparent',
                color:      tone === t ? '#222' : '#888',
                fontWeight: tone === t ? 500 : 400,
                transition: 'background 0.15s',
              }}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={onSubmit}
        disabled={loading}
        style={{
          width:        '100%',
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