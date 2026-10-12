import { useState } from 'react'
import { askAgent } from './api/client'
import './App.css'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  question: string
  answer: string
  toolsCalled: string[]
}

// ── Collapsible reasoning trace ───────────────────────────────────────────────

function ReasoningTrace({ toolsCalled }: { toolsCalled: string[] }) {
  const [open, setOpen] = useState<boolean>(false)

  if (toolsCalled.length === 0) return null

  return (
    <div className="reasoning-trace">
      <button
        className="reasoning-toggle"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        🔍 Reasoning trace {open ? '▲' : '▼'}
      </button>
      {open && (
        <ul className="reasoning-list">
          {toolsCalled.map((tool, i) => (
            <li key={i}>{tool}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ── Main app ──────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages]               = useState<Message[]>([])
  const [currentQuestion, setCurrentQuestion] = useState<string>('')
  const [isLoading, setIsLoading]             = useState<boolean>(false)
  const [error, setError]                     = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const question = currentQuestion.trim()
    if (!question) return

    setIsLoading(true)
    setError(null)

    try {
      const result = await askAgent(question)
      setMessages((prev) => [
        ...prev,
        {
          question,
          answer: result.answer,
          toolsCalled: result.tools_called,
        },
      ])
      setCurrentQuestion('')
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'An unexpected error occurred.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <h1>Vehicle Complaint Analyst</h1>
        <p className="app-subtitle">
          Ask natural-language questions about NHTSA vehicle safety complaints
        </p>
      </header>

      {/* ── Message list ── */}
      <main className="message-list">
        {messages.length === 0 && !isLoading && (
          <p className="empty-state">Ask a question below to get started.</p>
        )}

        {messages.map((msg, i) => (
          <div key={i} className="message-pair">
            <div className="message-question">
              <span className="message-label">You</span>
              <p>{msg.question}</p>
            </div>
            <div className="message-answer">
              <span className="message-label">Agent</span>
              <p>{msg.answer}</p>
              <ReasoningTrace toolsCalled={msg.toolsCalled} />
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="thinking-indicator" aria-live="polite">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-text">Thinking…</span>
          </div>
        )}
      </main>

      {/* ── Error banner ── */}
      {error !== null && (
        <div className="error-banner" role="alert">
          <strong>Error:</strong> {error}
          <button
            className="error-dismiss"
            type="button"
            aria-label="Dismiss error"
            onClick={() => setError(null)}
          >
            ✕
          </button>
        </div>
      )}

      {/* ── Input form ── */}
      <footer className="input-area">
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            id="question-input"
            className="question-input"
            type="text"
            placeholder="Ask about vehicle safety complaints…"
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            disabled={isLoading}
            aria-label="Your question"
          />
          <button
            id="send-button"
            className="send-button"
            type="submit"
            disabled={isLoading || currentQuestion.trim() === ''}
          >
            {isLoading ? 'Sending…' : 'Send'}
          </button>
        </form>
      </footer>
    </div>
  )
}
