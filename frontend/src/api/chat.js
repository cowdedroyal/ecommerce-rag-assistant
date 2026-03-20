const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export async function sendMessage(message, options = {}) {
  const { model = 'base', retrieval_mode = 'hybrid', conversation_id = null } = options

  const body = {
    message,
    model,
    retrieval_mode,
    use_llm: true,
  }

  if (conversation_id) {
    body.conversation_id = conversation_id
  }

  const response = await fetch(`${API_BASE}/chat/send`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

export async function clearConversation(model = 'base') {
  const response = await fetch(
    `${API_BASE}/chat/clear?model=${encodeURIComponent(model)}`,
    {
      method: 'POST',
    },
  )

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

export async function fetchCatalogSummary() {
  const response = await fetch(`${API_BASE}/products/catalog-summary`)

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

function parseSSEBlocks(chunkBuffer) {
  const blocks = chunkBuffer.split('\n\n')
  return {
    blocks: blocks.slice(0, -1),
    rest: blocks.at(-1) || '',
  }
}

function parseSSEEvent(block) {
  const lines = block.split('\n')
  let event = 'message'
  const dataLines = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  const dataText = dataLines.join('\n')
  let data = dataText
  try {
    data = JSON.parse(dataText)
  } catch {
    // Keep raw string payload
  }

  return { type: event, data }
}

export async function* streamMessage(message, options = {}) {
  const { model = 'base', retrieval_mode = 'hybrid', conversation_id = null } = options

  const body = {
    message,
    model,
    retrieval_mode,
    use_llm: true,
    stream: true,
  }

  if (conversation_id) {
    body.conversation_id = conversation_id
  }

  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Stream request failed with status ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const parsed = parseSSEBlocks(buffer)
      buffer = parsed.rest

      for (const block of parsed.blocks) {
        const event = parseSSEEvent(block.trim())
        if (event.type) {
          yield event
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
