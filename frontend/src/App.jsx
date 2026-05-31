import { useState } from 'react'
import Nav           from './components/Nav'
import EssayForm     from './components/EssayForm'
import LoadingState  from './components/LoadingState'
import ScoreResults  from './components/ScoreResults'
import ErrorBanner   from './components/ErrorBanner'

const API_BASE = '/api'

export default function App() {
  const [question,     setQuestion]     = useState('')
  const [essay,        setEssay]        = useState('')
  const [scores,       setScores]       = useState(null)
  const [meta,         setMeta]         = useState(null)
  const [loading,      setLoading]      = useState(false)
  const [fieldError,   setFieldError]   = useState(null)   // essay field error
  const [serverError,  setServerError]  = useState(null)   // banner error

  const handleSubmit = async () => {
    // reset all error + result state
    setFieldError(null)
    setServerError(null)
    setScores(null)
    setMeta(null)

    // basic client-side guard before hitting the API
    if (!essay.trim()) {
      setFieldError('Essay cannot be empty.')
      return
    }

    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/score`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question, essay }),
      })

      const data = await res.json()

      if (!res.ok) {
        // 400 / 422 / 500 from FastAPI
        const code    = data.detail?.code    || 'ERROR'
        const message = data.detail?.message || 'Something went wrong.'

        // field-level errors go under the essay box
        const fieldCodes = ['ESSAY_TOO_SHORT', 'ESSAY_TOO_LONG', 'EMPTY_INPUT']
        if (fieldCodes.includes(code)) {
          setFieldError(message)
        } else {
          setServerError(message)
        }
        return
      }

      setScores(data.scores)
      setMeta(data.meta)

    } catch {
      // network failure — FastAPI not running
      setServerError('Could not reach the scoring server. Is it running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Nav />

      <EssayForm
        question={question}   setQuestion={setQuestion}
        essay={essay}         setEssay={setEssay}
        onSubmit={handleSubmit}
        loading={loading}
        fieldError={fieldError}
      />

      <ErrorBanner message={serverError} />

      {loading && <LoadingState />}

      {scores && meta && !loading && (
        <ScoreResults scores={scores} meta={meta} />
      )}
    </div>
  )
}