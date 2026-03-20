<template>
  <div
    class="message-enter flex w-full"
    :class="role === 'user' ? 'justify-end' : 'justify-start'"
  >
    <!-- Avatar -->
    <div
      v-if="role === 'assistant'"
      class="mr-2 mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary-500 text-white text-sm font-bold"
    >
      AI
    </div>

    <!-- Message bubble -->
    <div
      class="relative max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm"
      :class="bubbleClasses"
    >
      <!-- Loading state -->
      <div v-if="loading" class="flex items-center py-1">
        <span class="loading-dot"></span>
        <span class="loading-dot"></span>
        <span class="loading-dot"></span>
      </div>

      <!-- Content with markdown rendering -->
      <div
        v-else
        class="markdown-body"
        v-html="renderedContent"
      ></div>

      <!-- Timestamp -->
      <div
        v-if="timestamp && !loading"
        class="mt-1.5 text-[10px] opacity-50"
        :class="role === 'user' ? 'text-right' : 'text-left'"
      >
        {{ formattedTime }}
      </div>
    </div>

    <!-- User avatar -->
    <div
      v-if="role === 'user'"
      class="ml-2 mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gray-200 text-gray-600 text-sm font-bold"
    >
      你
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'

// Configure marked for safe rendering
marked.setOptions({
  breaks: true,
  gfm: true,
})

function escapeHtml(raw = '') {
  return raw
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

const props = defineProps({
  /** Message sender role */
  role: {
    type: String,
    required: true,
    validator: (v) => ['user', 'assistant'].includes(v),
  },
  /** Message text content (supports markdown) */
  content: {
    type: String,
    default: '',
  },
  /** ISO timestamp string */
  timestamp: {
    type: String,
    default: '',
  },
  /** Whether the message is in loading/typing state */
  loading: {
    type: Boolean,
    default: false,
  },
})

// Compute bubble color classes based on role
const bubbleClasses = computed(() => {
  if (props.role === 'user') {
    return 'bg-primary-50 text-gray-800 rounded-tr-sm'
  }
  return 'bg-white text-gray-800 rounded-tl-sm border border-gray-100'
})

// Render markdown content to HTML
const renderedContent = computed(() => {
  if (!props.content) return ''
  return marked.parse(escapeHtml(props.content))
})

// Format the timestamp for display
const formattedTime = computed(() => {
  if (!props.timestamp) return ''
  try {
    const date = new Date(props.timestamp)
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return props.timestamp
  }
})
</script>
