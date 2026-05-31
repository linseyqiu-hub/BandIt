import { useEffect, useRef } from 'react'

const CRITERIA = [
  { key: 'task_response',              label: 'Task response' },
  { key: 'coherence_cohesion',         label: 'Coherence & cohesion' },
  { key: 'lexical_resource',           label: 'Lexical resource' },
  { key: 'grammatical_range_accuracy', label: 'Grammatical range' },
]

function SubCard({ label, score, animate }) {
  const barRef = useRef(null)

  useEffect(() => {
    if (!animate) return
    const pct = (score / 9) * 100
    const timer = setTimeout(() => {
      if (barRef.current) barRef.current.style.width = `${pct}%`
    }, 100)
    return () => clearTimeout(timer)
  }, [animate, score])

  return (
    <div style={{
      background:   '#f2f2f2',
      borderRadius: 8,
      padding:      '14px 12px',
      textAlign:    'center',
      animation:    'fadeUp 0.4s ease both',
    }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 8, lineHeight: 1.3 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 500, color: '#111' }}>{score.toFixed(1)}</div>
      <div style={{ height: 3, background: '#e0e0e0', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
        <div
          ref={barRef}
          style={{
            height:     3,
            width:      '0%',
            background: '#534AB7',
            borderRadius: 2,
            transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />
      </div>
    </div>
  )
}

export default function ScoreResults({ scores, meta }) {
  return (
    <div style={{ animation: 'fadeUp 0.4s ease' }}>
      <hr style={{ border: 'none', borderTop: '0.5px solid #e0e0e0', margin: '2rem 0' }} />

      {/* overall */}
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>Overall band score</div>
        <div style={{
          fontSize:      64,
          fontWeight:    500,
          color:         '#534AB7',
          lineHeight:    1,
          letterSpacing: '-2px',
          animation:     'scaleIn 0.6s ease',
        }}>
          {scores.overall.toFixed(1)}
        </div>
        <div style={{ fontSize: 13, color: '#888', marginTop: 4 }}>
          {meta.word_count} words · IELTS Task 2
        </div>
      </div>

      {/* sub-score cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
        {CRITERIA.map(({ key, label }) => (
          <SubCard key={key} label={label} score={scores[key]} animate={true} />
        ))}
      </div>

      {/* meta row */}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: '1rem', flexWrap: 'wrap' }}>
        <div style={{ fontSize: 12, color: '#888', background: '#f2f2f2', borderRadius: 8, padding: '3px 10px' }}>
          {meta.word_count} words
        </div>
        {meta.truncated && (
          <div style={{ fontSize: 12, color: '#533400', background: '#FAEEDA', borderRadius: 8, padding: '3px 10px' }}>
            ⚠ essay truncated — scores may be less accurate
          </div>
        )}
      </div>
    </div>
  )
}