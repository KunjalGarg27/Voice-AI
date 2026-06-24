import { useRef, useState } from 'react'

const VOICE_QUERY_URL = 'http://localhost:8000/voice-query'

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [response, setResponse] = useState('')
  const [error, setError] = useState('')

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const streamRef = useRef(null)

  /**
   * Request microphone access and begin capturing audio with MediaRecorder.
   */
  const startRecording = async () => {
    setError('')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        streamRef.current?.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        sendAudioToBackend()
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch {
      setError('Could not access the microphone. Check permissions and try again.')
    }
  }

  /**
   * Stop the active MediaRecorder session. Audio is sent after onstop fires.
   */
  const stopRecording = () => {
    const mediaRecorder = mediaRecorderRef.current
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      return
    }

    mediaRecorder.stop()
    setIsRecording(false)
  }

  /**
   * Upload the recorded audio blob to the FastAPI /voice-query endpoint.
   */
  const sendAudioToBackend = async () => {
    const mimeType = mediaRecorderRef.current?.mimeType || 'audio/webm'
    const blob = new Blob(audioChunksRef.current, { type: mimeType })

    if (blob.size === 0) {
      setError('Recording was empty. Please try again.')
      return
    }

    const extension = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('wav') ? 'wav' : 'webm'
    const formData = new FormData()
    formData.append('audio', blob, `recording.${extension}`)

    setIsLoading(true)
    setTranscript('')
    setResponse('')
    setError('')

    try {
      const result = await fetch(VOICE_QUERY_URL, {
        method: 'POST',
        body: formData,
      })

      const data = await result.json().catch(() => null)

      if (!result.ok) {
        const detail = data?.detail
        const message =
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((item) => item.msg).join(', ')
              : `Request failed with status ${result.status}`
        throw new Error(message)
      }

      setTranscript(data.transcript ?? '')
      setResponse(data.response ?? '')
      if (data.audio_url) {
          const audio = new Audio(data.audio_url)

          audio.play().catch((err) => {
              console.error("Audio playback failed:", err)
          })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process audio.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main style={{ maxWidth: '640px', margin: '40px auto', padding: '0 20px', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ marginBottom: '8px' }}>Voice AI Assistant</h1>
      <p style={{ marginBottom: '24px', color: '#555' }}>
        Record a question, then hear the transcript and AI response.
      </p>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <button
          type="button"
          onClick={startRecording}
          disabled={isRecording || isLoading}
        >
          Start Recording
        </button>
        <button
          type="button"
          onClick={stopRecording}
          disabled={!isRecording || isLoading}
        >
          Stop Recording
        </button>
      </div>

      {isLoading && (
        <p style={{ marginBottom: '16px', color: '#2563eb' }}>
          Processing audio… transcribing and generating a response.
        </p>
      )}

      {error && (
        <p style={{ marginBottom: '16px', color: '#b91c1c' }}>
          {error}
        </p>
      )}

      <section style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '1rem', marginBottom: '8px' }}>Transcript</h2>
        <p style={{ background: '#f3f4f6', padding: '12px', borderRadius: '6px', minHeight: '48px' }}>
          {transcript || (isLoading ? '…' : 'No transcript yet.')}
        </p>
      </section>

      <section>
        <h2 style={{ fontSize: '1rem', marginBottom: '8px' }}>AI Response</h2>
        <p style={{ background: '#f3f4f6', padding: '12px', borderRadius: '6px', minHeight: '48px' }}>
          {response || (isLoading ? '…' : 'No response yet.')}
        </p>
      </section>
    </main>
  )
}

export default App
