export default function Nav() {
  return (
    <nav style={{
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'space-between',
      padding:        '14px 0',
      borderBottom:   '0.5px solid #e0e0e0',
      marginBottom:   '2rem',
    }}>
      <div style={{ fontSize: 20, fontWeight: 500, letterSpacing: '-0.3px' }}>
        Band<span style={{ color: '#534AB7' }}>It</span>
      </div>
      <div style={{
        fontSize:     12,
        color:        '#888',
        background:   '#f2f2f2',
        border:       '0.5px solid #e0e0e0',
        borderRadius: 8,
        padding:      '3px 10px',
      }}>
        IELTS Task 2 Scorer
      </div>
    </nav>
  )
}