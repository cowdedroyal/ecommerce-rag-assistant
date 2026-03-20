<template>
  <div class="flex h-screen flex-col">
    <!-- Top bar -->
    <header class="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-3 shadow-sm">
      <div class="flex items-center gap-3">
        <!-- Logo / Title -->
        <div class="flex items-center gap-2">
          <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-500 text-white">
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <h1 class="text-lg font-bold text-gray-800">电商智能助手</h1>
        </div>
        <div
          v-if="catalogSummary"
          class="hidden rounded-full bg-slate-100 px-3 py-1 text-[11px] font-medium text-slate-600 md:block"
        >
          {{ catalogSummary.dataset_mode === 'synthetic-demo' ? '演示数据模式' : '目录已加载' }}
          · {{ catalogSummary.product_count }} 商品
        </div>
      </div>

      <!-- Selectors -->
      <div class="flex items-center gap-4">
        <ModelSelector
          v-model="selectedModel"
          label="模型"
          :options="modelOptions"
        />
        <ModelSelector
          v-model="selectedRetrieval"
          label="检索"
          :options="retrievalOptions"
        />
        <button
          @click="clearChat"
          class="rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-gray-500 transition hover:bg-gray-50 hover:text-gray-700"
          title="清空对话"
        >
          清空
        </button>
      </div>
    </header>

    <!-- Main content area -->
    <div class="flex flex-1 overflow-hidden">
      <!-- Left: Chat area -->
      <div class="flex flex-1 flex-col">
        <!-- Messages list -->
        <div
          ref="messagesContainer"
          class="flex-1 overflow-y-auto px-6 py-4 space-y-4"
        >
          <!-- Welcome message when empty -->
          <div v-if="messages.length === 0" class="flex h-full items-center justify-center">
            <div class="max-w-3xl text-center">
              <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary-50">
                <svg class="h-8 w-8 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                </svg>
              </div>
              <h2 class="mb-2 text-xl font-semibold text-gray-700">欢迎使用电商智能助手</h2>
              <p class="text-sm text-gray-400">当前默认使用稳定可复现的合成目录数据，适合演示检索、推荐解释与多轮问答能力。</p>

              <div v-if="catalogSummary" class="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm">
                  <div class="text-[11px] uppercase tracking-wide text-slate-400">商品数</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-800">{{ catalogSummary.product_count }}</div>
                </div>
                <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm">
                  <div class="text-[11px] uppercase tracking-wide text-slate-400">订单数</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-800">{{ catalogSummary.order_count }}</div>
                </div>
                <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm">
                  <div class="text-[11px] uppercase tracking-wide text-slate-400">平均评分</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-800">{{ catalogSummary.avg_rating }}</div>
                </div>
                <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm">
                  <div class="text-[11px] uppercase tracking-wide text-slate-400">主分类数</div>
                  <div class="mt-1 text-2xl font-semibold text-slate-800">{{ catalogSummary.category_count }}</div>
                </div>
              </div>

              <div class="mt-6 flex flex-wrap justify-center gap-2">
                <button
                  v-for="suggestion in suggestions"
                  :key="suggestion"
                  @click="sendFromSuggestion(suggestion)"
                  class="rounded-full border border-gray-200 bg-white px-4 py-2 text-sm text-gray-600 shadow-sm transition hover:border-primary-300 hover:text-primary-600"
                >
                  {{ suggestion }}
                </button>
              </div>
            </div>
          </div>

          <!-- Chat messages -->
          <ChatMessage
            v-for="(msg, index) in messages"
            :key="index"
            :role="msg.role"
            :content="msg.content"
            :timestamp="msg.timestamp"
            :loading="msg.loading"
          />
        </div>

        <!-- Input area -->
        <div class="border-t border-gray-200 bg-white px-6 py-4">
          <div class="flex items-end gap-3">
            <div class="relative flex-1">
              <textarea
                ref="inputRef"
                v-model="inputText"
                @keydown.enter.exact="handleEnter"
                @input="autoResize"
                placeholder="输入你的问题，例如：推荐一款适合跑步的运动鞋..."
                rows="1"
                class="w-full resize-none rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 pr-12 text-sm
                       text-gray-700 transition placeholder:text-gray-400
                       focus:border-primary-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-primary-400"
                :disabled="isLoading"
                style="max-height: 120px"
              ></textarea>
            </div>
            <button
              @click="handleSend"
              :disabled="!inputText.trim() || isLoading"
              class="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-primary-500 text-white
                     shadow-sm transition
                     hover:bg-primary-600
                     disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              <svg v-if="!isLoading" class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
              <svg v-else class="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </button>
          </div>
          <p class="mt-2 text-center text-[10px] text-gray-300">
            模型: {{ selectedModel }} | 检索: {{ selectedRetrieval }} | 按 Enter 发送，Shift+Enter 换行
          </p>
        </div>
      </div>

      <!-- Right: Retrieval panel -->
      <div class="hidden w-80 flex-shrink-0 border-l border-gray-200 bg-gray-50/50 lg:block">
        <div class="flex h-full flex-col overflow-hidden">
          <!-- Panel header -->
          <div class="border-b border-gray-200 px-4 py-3">
            <h3 class="text-sm font-semibold text-gray-700">检索过程</h3>
            <p class="text-[10px] text-gray-400">实时展示 RAG 检索与重排序结果</p>
          </div>

          <!-- Panel content - scrollable -->
          <div class="flex-1 overflow-y-auto px-4 py-3">
            <!-- Retrieval process -->
            <div v-if="retrievalProcess && hasRetrievalData" class="mb-4">
              <RetrievalPanel :retrievalProcess="retrievalProcess" />
            </div>
            <div v-else class="flex items-center justify-center py-8">
              <p class="text-xs text-gray-400">发送消息后将展示检索过程</p>
            </div>

            <!-- Recommended products -->
            <div v-if="recommendedProducts.length > 0" class="mt-4">
              <h4 class="mb-2 text-sm font-semibold text-gray-700">推荐商品</h4>
              <div class="space-y-3">
                <ProductCard
                  v-for="product in recommendedProducts"
                  :key="product.id"
                  :product="product"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted } from 'vue'
import ChatMessage from '../components/ChatMessage.vue'
import ProductCard from '../components/ProductCard.vue'
import RetrievalPanel from '../components/RetrievalPanel.vue'
import ModelSelector from '../components/ModelSelector.vue'
import { clearConversation, fetchCatalogSummary, sendMessage, streamMessage } from '../api/chat.js'

// --- State ---

const messages = ref([])
const inputText = ref('')
const isLoading = ref(false)
const selectedModel = ref('base')
const selectedRetrieval = ref('hybrid')
const retrievalProcess = ref(null)
const recommendedProducts = ref([])
const conversationId = ref(null)
const catalogSummary = ref(null)

// DOM refs
const messagesContainer = ref(null)
const inputRef = ref(null)

// --- Constants ---

const modelOptions = [
  { value: 'base', label: 'Base 基座模型' },
  { value: 'sft', label: 'SFT 微调模型' },
  { value: 'dpo', label: 'DPO 对齐模型' },
]

const retrievalOptions = [
  { value: 'hybrid', label: 'Hybrid 混合检索' },
  { value: 'dense', label: 'Dense 稠密检索' },
  { value: 'bm25', label: 'BM25 稀疏检索' },
]

const suggestions = [
  '推荐一款性价比高的蓝牙耳机',
  '有哪些适合送礼的护肤品？',
  '帮我对比几款热门手机',
  '推荐适合学生党的笔记本电脑',
]

// --- Computed ---

const hasRetrievalData = computed(() => {
  if (!retrievalProcess.value) return false
  const rp = retrievalProcess.value
  return (
    (rp.bm25_results && rp.bm25_results.length > 0) ||
    (rp.dense_results && rp.dense_results.length > 0) ||
    (rp.reranked_results && rp.reranked_results.length > 0)
  )
})

// --- Methods ---

// Auto-scroll to the latest message
function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// Auto-resize the textarea as the user types
function autoResize() {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
}

// Handle Enter key - send on Enter, newline on Shift+Enter
function handleEnter(e) {
  if (e.shiftKey) return // Allow Shift+Enter for newline
  e.preventDefault()
  handleSend()
}

// Send a message from suggestion chips
function sendFromSuggestion(text) {
  inputText.value = text
  handleSend()
}

// Clear the conversation
async function clearChat() {
  try {
    await clearConversation(selectedModel.value)
  } catch (err) {
    console.warn('Failed to clear server conversation:', err.message)
  }
  messages.value = []
  retrievalProcess.value = null
  recommendedProducts.value = []
  conversationId.value = null
}

async function loadCatalog() {
  try {
    catalogSummary.value = await fetchCatalogSummary()
  } catch (err) {
    console.warn('Failed to load catalog summary:', err.message)
  }
}

// Main send handler - tries streaming first, falls back to non-streaming
async function handleSend() {
  const text = inputText.value.trim()
  if (!text || isLoading.value) return

  // Add user message
  const userMsg = {
    role: 'user',
    content: text,
    timestamp: new Date().toISOString(),
    loading: false,
  }
  messages.value.push(userMsg)
  inputText.value = ''
  autoResize()
  scrollToBottom()

  // Add assistant loading placeholder
  const assistantMsg = {
    role: 'assistant',
    content: '',
    timestamp: '',
    loading: true,
  }
  messages.value.push(assistantMsg)
  scrollToBottom()

  isLoading.value = true
  const assistantIndex = messages.value.length - 1

  const options = {
    model: selectedModel.value,
    retrieval_mode: selectedRetrieval.value,
    conversation_id: conversationId.value,
  }

  try {
    // Try streaming first
    let streamed = false
    try {
      let accumulated = ''
      for await (const event of streamMessage(text, options)) {
        streamed = true
        if (event.type === 'token') {
          accumulated += event.data?.text || event.data || ''
          messages.value[assistantIndex].content = accumulated
          messages.value[assistantIndex].loading = false
          scrollToBottom()
        } else if (event.type === 'retrieval') {
          retrievalProcess.value = event.data?.retrieval_process || null
        } else if (event.type === 'products') {
          recommendedProducts.value = event.data?.recommended_products || []
        } else if (event.type === 'done') {
          if (event.data?.conversation_id) {
            conversationId.value = event.data.conversation_id
          }
          break
        } else if (event.type === 'error') {
          throw new Error(event.data || 'Stream error')
        }
      }

      if (streamed && !messages.value[assistantIndex].content) {
        // Stream connected but no content received
        streamed = false
      }
    } catch (streamErr) {
      // If streaming fails, fall back to regular request
      console.warn('Streaming not available, falling back to regular request:', streamErr.message)
      streamed = false
    }

    // Fallback: non-streaming request
    if (!streamed) {
      const result = await sendMessage(text, options)

      messages.value[assistantIndex].content = result.answer || '抱歉，我暂时无法回答这个问题。'
      messages.value[assistantIndex].loading = false

      if (result.retrieval_process) {
        retrievalProcess.value = result.retrieval_process
      }

      if (result.recommended_products) {
        recommendedProducts.value = result.recommended_products
      }

      if (result.conversation_id) {
        conversationId.value = result.conversation_id
      }
    }

    messages.value[assistantIndex].timestamp = new Date().toISOString()
  } catch (err) {
    console.error('Chat error:', err)
    messages.value[assistantIndex].content = `出错了：${err.message || '网络请求失败，请检查后端服务是否运行。'}`
    messages.value[assistantIndex].loading = false
    messages.value[assistantIndex].timestamp = new Date().toISOString()
  } finally {
    isLoading.value = false
    scrollToBottom()
    // Re-focus input
    nextTick(() => {
      inputRef.value?.focus()
    })
  }
}

// Focus input on mount
onMounted(() => {
  loadCatalog()
  inputRef.value?.focus()
})
</script>
