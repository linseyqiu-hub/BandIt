import Nav             from './components/Nav'
import EssayForm       from './components/EssayForm'
import LoadingState    from './components/LoadingState'
import ScoreResults    from './components/ScoreResults'
import ErrorBanner     from './components/ErrorBanner'
import FeedbackDisplay from './components/FeedbackDisplay'
import { useScoring }  from './hooks/useScoring'

export default function App() {
  const {
    question,   setQuestion,
    essay,      setEssay,
    tone,       setTone,
    scores,     meta,
    feedback,   feedbackLoading,
    loading,    fieldError,
    serverError,
    handleSubmit,
  } = useScoring()

  return (
    <div>
      <Nav />

      <EssayForm
        question={question}   setQuestion={setQuestion}
        essay={essay}         setEssay={setEssay}
        tone={tone}           setTone={setTone}
        onSubmit={handleSubmit}
        loading={loading}
        fieldError={fieldError}
      />

      <ErrorBanner message={serverError} />

      {loading && <LoadingState />}

      {scores && meta && !loading && (
        <ScoreResults scores={scores} meta={meta} />
      )}

      {feedbackLoading && <LoadingState message="BandIt engine is thinking..." />}

      {feedback && !feedbackLoading && (
        <FeedbackDisplay feedback={feedback} tone={tone} />
      )}
    </div>
  )
}