<template>
  <div class="flex items-center gap-2">
    <label class="text-sm text-gray-500 whitespace-nowrap">{{ label }}</label>
    <select
      :value="modelValue"
      @change="$emit('update:modelValue', $event.target.value)"
      class="block w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm
             text-gray-700 shadow-sm transition
             focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500
             hover:border-gray-300 cursor-pointer"
    >
      <option v-for="opt in options" :key="opt.value" :value="opt.value">
        {{ opt.label }}
      </option>
    </select>
  </div>
</template>

<script setup>
// ModelSelector - a compact dropdown for selecting model variant or retrieval mode

defineProps({
  /** Currently selected value (v-model) */
  modelValue: {
    type: String,
    required: true,
  },
  /** Dropdown label text */
  label: {
    type: String,
    default: '',
  },
  /** Available options: [{ value, label }] */
  options: {
    type: Array,
    required: true,
    validator: (opts) => opts.every((o) => 'value' in o && 'label' in o),
  },
})

defineEmits(['update:modelValue'])
</script>
