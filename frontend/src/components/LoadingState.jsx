import { useEffect, useState } from 'react'

const MESSAGES = [
  'Analysing your essay...',
  'Evaluating task response...',
  'Checking coherence and cohesion...',
  'Assessing lexical resource...',
  'Reviewing grammatical range...',
  'Almost there...',
]

function SkeletonCard() {
  const shimmer = {
    position:   'relative',
    overflow:   'hidden',
    background: '#e8e8e8',
    borderRadius: 4,
  }
  const after = {
    content:    '""',
    position:   'absolute',
    inset:      0,
    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent)',
    animation:  'shimmer 1.4s infinite',
  }
  return (
    <div style={{ background: '#f2f2f2', borderRadius: 8, padding: '14px 12px', textAlign: 'center' }}>
      <div style={{ ...shimmer, height: 10, width: '70%', margin: '0 auto 10px' }}>
        <div style={after} />
      </div>
      <div style={{ ...shimmer, height: 22, width: '40%', margin: '0 auto 10px' }}>
        <div style={after} />
      </div>
      <div style={{ ...shimmer, height: 3, width: '100%' }}>
        <div style={after} />
      </div>
    </div>
  )
}

export default function LoadingState() {
  const [msgIdx, setMsgIdx] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setMsgIdx(i => (i + 1) % MESSAGES.length)
    }, 1400)
    return () => clearInterval(interval)
  }, [])

  return (
    <div>
      <hr style={{ border: 'none', borderTop: '0.5px solid #e0e0e0', margin: '2rem 0' }} />

      {/* dots + message */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: '1rem 0' }}>
        <div style={{ display: 'flex', gap: 6 }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width:           7,
              height:          7,
              borderRadius:    '50%',
              background:      '#534AB7',
              animation:       `bounce 1.2s infinite ease-in-out`,
              animationDelay:  `${i * 0.2}s`,
            }} />
          ))}
        </div>
        <div style={{ fontSize: 13, color: '#888', animation: 'fadeIn 0.3s ease' }} key={msgIdx}>
          {MESSAGES[msgIdx]}
        </div>
      </div>

      {/* skeleton cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginTop: '1rem' }}>
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    </div>
  )
}