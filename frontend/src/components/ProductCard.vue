<template>
  <div class="rounded-2xl border border-slate-200 bg-white/95 p-3 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md">
    <!-- Product image placeholder -->
    <div class="mb-2 flex h-28 items-center justify-center rounded-xl bg-slate-50 text-slate-300">
      <img
        v-if="product.image"
        :src="product.image"
        :alt="product.title"
        class="h-full w-full rounded-xl object-cover"
      />
      <svg
        v-else
        class="h-10 w-10"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="1.5"
          d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
        />
      </svg>
    </div>

    <!-- Product title -->
    <div class="mb-2 flex items-start justify-between gap-2">
      <h4 class="line-clamp-2 text-sm font-medium text-slate-800" :title="product.title">
        {{ product.title }}
      </h4>
      <span
        v-if="product.confidence"
        class="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-600"
      >
        匹配度 {{ product.confidence }}%
      </span>
    </div>

    <p v-if="product.description" class="mb-2 line-clamp-2 text-xs leading-5 text-slate-500">
      {{ product.description }}
    </p>

    <div
      v-if="product.match_reason"
      class="mb-2 rounded-xl border border-amber-100 bg-amber-50 px-2.5 py-2 text-[11px] leading-5 text-amber-700"
    >
      {{ product.match_reason }}
    </div>

    <!-- Price -->
    <div class="mb-1.5 flex items-baseline gap-1">
      <span class="text-lg font-bold text-rose-500">¥{{ formattedPrice }}</span>
      <span v-if="product.brand" class="text-[11px] text-slate-400">{{ product.brand }}</span>
    </div>

    <!-- Rating stars -->
    <div class="mb-2 flex items-center gap-1">
      <div class="flex">
        <svg
          v-for="i in 5"
          :key="i"
          class="h-3.5 w-3.5"
          :class="i <= Math.round(product.rating || 0) ? 'text-yellow-400' : 'text-gray-200'"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
          />
        </svg>
      </div>
      <span class="text-xs text-slate-400">{{ product.rating?.toFixed(1) || 'N/A' }}</span>
    </div>

    <!-- Category and source tags -->
    <div class="flex flex-wrap gap-1">
      <span
        v-if="product.category"
        class="inline-block rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-500"
      >
        {{ product.category }}
      </span>
      <span
        v-if="product.source"
        class="inline-block rounded-full bg-primary-50 px-2 py-0.5 text-[10px] text-primary-600"
      >
        {{ product.source }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  /** Product data object */
  product: {
    type: Object,
    required: true,
    validator: (p) => 'title' in p && 'price' in p,
  },
})

// Format price with two decimal places
const formattedPrice = computed(() => {
  const price = Number(props.product.price)
  if (isNaN(price)) return '0.00'
  return price.toFixed(2)
})
</script>
