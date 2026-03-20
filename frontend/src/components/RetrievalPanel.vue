<template>
  <div class="space-y-3">
    <div
      v-if="summary"
      class="rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50 p-3"
    >
      <div class="mb-2 flex items-center justify-between">
        <h4 class="text-xs font-semibold tracking-wide text-slate-700">运行态摘要</h4>
        <span class="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-500">
          {{ summary.dataset_mode || 'unknown' }}
        </span>
      </div>
      <div class="grid grid-cols-2 gap-2 text-[11px] text-slate-500">
        <div class="rounded-lg bg-white px-2 py-1.5">
          请求模式：<span class="font-medium text-slate-700">{{ summary.requested_mode }}</span>
        </div>
        <div class="rounded-lg bg-white px-2 py-1.5">
          实际模式：<span class="font-medium text-slate-700">{{ summary.selected_mode }}</span>
        </div>
        <div class="rounded-lg bg-white px-2 py-1.5">
          Dense：<span class="font-medium text-slate-700">{{ summary.dense_available ? '可用' : '降级' }}</span>
        </div>
        <div class="rounded-lg bg-white px-2 py-1.5">
          Rerank：<span class="font-medium text-slate-700">{{ summary.reranker_enabled ? '开启' : '关闭' }}</span>
        </div>
      </div>
    </div>

    <!-- BM25 Results Panel -->
    <CollapsibleSection
      title="BM25 检索结果"
      icon="📝"
      :count="bm25Results.length"
      :defaultOpen="false"
    >
      <div class="space-y-2">
        <RetrievalItem
          v-for="(item, idx) in bm25Results"
          :key="'bm25-' + idx"
          :item="item"
          color="blue"
        />
        <div v-if="!bm25Results.length" class="py-2 text-center text-xs text-gray-400">
          暂无结果
        </div>
      </div>
    </CollapsibleSection>

    <!-- Dense Results Panel -->
    <CollapsibleSection
      title="Dense 检索结果"
      icon="🔍"
      :count="denseResults.length"
      :defaultOpen="false"
    >
      <div class="space-y-2">
        <RetrievalItem
          v-for="(item, idx) in denseResults"
          :key="'dense-' + idx"
          :item="item"
          color="purple"
        />
        <div v-if="!denseResults.length" class="py-2 text-center text-xs text-gray-400">
          暂无结果
        </div>
      </div>
    </CollapsibleSection>

    <!-- Reranked Results Panel -->
    <CollapsibleSection
      title="Rerank 最终结果"
      icon="✅"
      :count="rerankedResults.length"
      :defaultOpen="true"
    >
      <div class="space-y-2">
        <RetrievalItem
          v-for="(item, idx) in rerankedResults"
          :key="'rerank-' + idx"
          :item="item"
          color="green"
          :highlight="idx === 0"
        />
        <div v-if="!rerankedResults.length" class="py-2 text-center text-xs text-gray-400">
          暂无结果
        </div>
      </div>
    </CollapsibleSection>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  /** Retrieval process data with bm25_results, dense_results, reranked_results */
  retrievalProcess: {
    type: Object,
    default: () => ({}),
  },
})

// Extract result arrays with safe fallback
const bm25Results = computed(() => props.retrievalProcess?.bm25_results || [])
const denseResults = computed(() => props.retrievalProcess?.dense_results || [])
const rerankedResults = computed(() => props.retrievalProcess?.reranked_results || [])
const summary = computed(() => props.retrievalProcess?.summary || null)

// --- Inline sub-components ---

// CollapsibleSection: a foldable section with title and toggle
const CollapsibleSection = {
  props: {
    title: String,
    icon: { type: String, default: '' },
    count: { type: Number, default: 0 },
    defaultOpen: { type: Boolean, default: false },
  },
  setup(props, { slots }) {
    const isOpen = ref(props.defaultOpen)
    const toggle = () => { isOpen.value = !isOpen.value }
    return { isOpen, toggle, slots }
  },
  template: `
    <div class="rounded-lg border border-gray-100 bg-white overflow-hidden">
      <button
        @click="toggle"
        class="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
      >
        <span class="flex items-center gap-1.5">
          <span>{{ icon }}</span>
          <span>{{ title }}</span>
          <span class="ml-1 rounded-full bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500">
            {{ count }}
          </span>
        </span>
        <svg
          class="h-4 w-4 text-gray-400 transition-transform duration-200"
          :class="{ 'rotate-180': isOpen }"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div v-show="isOpen" class="border-t border-gray-50 px-3 py-2">
        <slot />
      </div>
    </div>
  `,
}

// RetrievalItem: a single retrieval result with score bar
const RetrievalItem = {
  props: {
    item: { type: Object, required: true },
    color: { type: String, default: 'blue' },
    highlight: { type: Boolean, default: false },
  },
  setup(props) {
    const scorePercent = computed(() => {
      const score = props.item.score ?? props.item.relevance_score ?? 0
      // Scores may be 0-1 or 0-100; normalize to percentage
      return Math.min(100, Math.max(0, score > 1 ? score : score * 100))
    })

    const barColorClass = computed(() => {
      const map = {
        blue: 'bg-blue-400',
        purple: 'bg-amber-400',
        green: 'bg-green-500',
      }
      return map[props.color] || 'bg-blue-400'
    })

    const scoreValue = computed(() => {
      const score = props.item.score ?? props.item.relevance_score ?? 0
      return score > 1 ? score.toFixed(1) : score.toFixed(4)
    })

    return { scorePercent, barColorClass, scoreValue }
  },
  template: `
    <div
      class="rounded-lg p-2 text-xs transition"
      :class="highlight ? 'bg-green-50 border border-green-200' : 'bg-gray-50'"
    >
      <div class="mb-1 flex items-start justify-between gap-2">
        <span class="font-medium text-gray-700 line-clamp-2 flex-1">
          {{ item.title || item.doc_title || item.text?.slice(0, 50) || '未知文档' }}
        </span>
        <span class="flex-shrink-0 text-[10px] text-gray-400">
          {{ scoreValue }}
        </span>
      </div>
      <div class="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          class="h-full rounded-full transition-all duration-500"
          :class="barColorClass"
          :style="{ width: scorePercent + '%' }"
        />
      </div>
      <div v-if="item.text" class="mt-1 text-[10px] text-gray-400 line-clamp-2">
        {{ item.text }}
      </div>
    </div>
  `,
}
</script>
