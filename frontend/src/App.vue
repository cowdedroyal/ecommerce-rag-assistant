<template>
  <div class="h-screen w-screen overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.12),_transparent_28%),linear-gradient(180deg,_#f8fafc_0%,_#eef4ff_100%)]">
    <ChatView />
  </div>
</template>

<script setup>
// Root application component - renders ChatView directly
import ChatView from './views/ChatView.vue'
</script>
