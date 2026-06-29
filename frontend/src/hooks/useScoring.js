import { useState } from 'react'

const API_BASE = '/api'
const FIELD_CODES = ['ESSAY_TOO_SHORT', 'ESSAY_TOO_LONG', 'EMPTY_INPUT']

export function useScoring() {
  const [question,        setQuestion]        = useState('')
  const [essay,           setEssay]           = useState('')
  const [tone,            setTone]            = useState('coaching')
  const [scores,          setScores]          = useState(null)
  const [meta,            setMeta]            = useState(null)
  const [feedback,        setFeedback]        = useState(null)
  const [loading,         setLoading]         = useState(false)
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [fieldError,      setFieldError]      = useState(null)
  const [serverError,     setServerError]     = useState(null)

  const handleSubmit = async () => {
    setFieldError(null)
    setServerError(null)
    setScores(null)
    setMeta(null)
    setFeedback(null)

    if (!essay.trim()) {
      setFieldError('Essay cannot be empty.')
      return
    }

    setLoading(true)
    try {
      const res  = await fetch(`${API_BASE}/score`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question, essay }),
      })
      const data = await res.json()

      if (!res.ok) {
        const code    = data.detail?.code    || 'ERROR'
        const message = data.detail?.message || 'Something went wrong.'
        if (FIELD_CODES.includes(code)) setFieldError(message)
        else setServerError(message)
        return
      }

      setScores(data.scores)
      setMeta(data.meta)

      setFeedbackLoading(true)
      try {
        const fbRes  = await fetch(`${API_BASE}/feedback`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ question, essay, scores: data.scores, tone }),
        })
        const fbData = await fbRes.json()
        if (fbRes.ok) setFeedback(fbData.feedback)
        else setServerError('Feedback unavailable.')
      } catch {
        setServerError('Could not reach the feedback server.')
      } finally {
        setFeedbackLoading(false)
      }

    } catch {
      setServerError('Could not reach the scoring server. Is it running?')
    } finally {
      setLoading(false)
    }
  }

  return {
    question,   setQuestion,
    essay,      setEssay,
    tone,       setTone,
    scores,     meta,
    feedback,   feedbackLoading,
    loading,    fieldError,
    serverError,
    handleSubmit,
  }
}