import { useEffect, useId, useRef, useState } from 'react'

import { bytes } from '../lib/format'
import { Num } from './Num'

const ACCEPT = 'image/png,image/jpeg,image/webp'
const MAX_BYTES = 15 * 1024 * 1024

interface SlotProps {
  label: string
  hint: string
  file: Blob | null
  onFile: (file: Blob | null) => void
  onProblem: (message: string) => void
  disabled: boolean
}

function Slot({ label, hint, file, onFile, onProblem, disabled }: SlotProps) {
  const id = useId()
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const accept = (candidate: File | undefined) => {
    if (!candidate) return
    if (!ACCEPT.split(',').includes(candidate.type)) {
      onProblem(`${label}: that is a ${candidate.type || 'file of unknown type'}. Use a PNG, JPEG or WebP image.`)
      return
    }
    if (candidate.size > MAX_BYTES) {
      onProblem(`${label}: that file is ${bytes(candidate.size)}. The limit is 15 MB.`)
      return
    }
    onFile(candidate)
  }

  return (
    <div
      className="slot"
      data-over={over}
      data-filled={file !== null}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        accept(e.dataTransfer.files[0])
      }}
    >
      <label htmlFor={id} className="slot-label">
        {label}
      </label>
      {preview ? (
        <img className="slot-preview" src={preview} alt={`${label} you chose`} />
      ) : (
        <p className="slot-hint">{hint}</p>
      )}
      <input
        ref={input}
        id={id}
        className="visually-hidden"
        type="file"
        accept={ACCEPT}
        disabled={disabled}
        onChange={(e) => {
          accept(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      <div className="slot-actions">
        <button type="button" className="btn btn-small" disabled={disabled} onClick={() => input.current?.click()}>
          {file ? 'Replace' : 'Choose file'}
        </button>
        {file && (
          <button type="button" className="btn btn-small" disabled={disabled} onClick={() => onFile(null)}>
            Clear
          </button>
        )}
        {file && (
          <span className="slot-size">
            <Num>{bytes(file.size)}</Num>
          </span>
        )}
      </div>
    </div>
  )
}

interface UploadPanelProps {
  busy: boolean
  editModel: string | null
  onPeel: (original: Blob, edited: Blob, instruction: string) => void
  onEdit: (original: Blob, instruction: string) => Promise<Blob | null>
  onClose: () => void
}

export function UploadPanel({ busy, editModel, onPeel, onEdit, onClose }: UploadPanelProps) {
  const [original, setOriginal] = useState<Blob | null>(null)
  const [edited, setEdited] = useState<Blob | null>(null)
  const [instruction, setInstruction] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const instructionId = useId()

  const ready = original !== null && edited !== null && instruction.trim().length > 0
  const canEdit = editModel !== null && original !== null && instruction.trim().length > 0
  const locked = busy || editing

  const missing = [
    original ? null : 'the original',
    edited ? null : 'the AI edit',
    instruction.trim() ? null : 'the instruction',
  ].filter(Boolean)

  const runEdit = async () => {
    if (!original) return
    setProblem(null)
    setEditing(true)
    const result = await onEdit(original, instruction.trim())
    setEditing(false)
    if (result) {
      setEdited(result)
      onPeel(original, result, instruction.trim())
    }
  }

  return (
    <form
      className="upload"
      aria-label="Use your own images"
      onSubmit={(e) => {
        e.preventDefault()
        setProblem(null)
        if (ready && original && edited) onPeel(original, edited, instruction.trim())
      }}
    >
      <div className="upload-head">
        <h2>Use your own images</h2>
        <button type="button" className="btn btn-small" onClick={onClose}>
          Back to examples
        </button>
      </div>
      <div className="upload-slots">
        <Slot
          label="Original"
          hint="Drop the image you started from, before the AI touched it."
          file={original}
          onFile={(f) => {
            setProblem(null)
            setOriginal(f)
          }}
          onProblem={setProblem}
          disabled={locked}
        />
        <Slot
          label="AI edit"
          hint="Drop what the editor gave back. Any size; Peel aligns it."
          file={edited}
          onFile={(f) => {
            setProblem(null)
            setEdited(f)
          }}
          onProblem={setProblem}
          disabled={locked}
        />
      </div>
      <div className="upload-instruction">
        <label htmlFor={instructionId}>What you asked the editor to do</label>
        <input
          id={instructionId}
          type="text"
          value={instruction}
          maxLength={2000}
          placeholder="Make the jacket red"
          autoComplete="off"
          disabled={locked}
          onChange={(e) => setInstruction(e.target.value)}
        />
      </div>
      {problem && (
        <p className="upload-problem" role="alert">
          {problem}
        </p>
      )}
      <div className="upload-actions">
        <button type="submit" className="btn btn-solid" disabled={!ready || locked}>
          {busy ? 'Peeling' : 'Peel it'}
        </button>
        {editModel !== null && (
          <button type="button" className="btn" disabled={!canEdit || locked} onClick={runEdit}>
            {editing ? 'Editing with Nano Banana' : 'Edit with Nano Banana, then peel'}
          </button>
        )}
        <p className="upload-need" aria-live="polite">
          {ready ? 'PNG, JPEG or WebP, up to 15 MB each.' : `Still needed: ${missing.join(', ')}.`}
          {editModel !== null && !edited && ' No edit yet? Add the original and the instruction, and let Nano Banana make one.'}
        </p>
      </div>
    </form>
  )
}
