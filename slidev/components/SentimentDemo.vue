<script setup lang="ts">
import { computed, ref } from 'vue'

interface PredictionResult {
  score: number
  label: string
  confidence: number
}

const results: PredictionResult[] = [
  { score: 5, label: '极端正面', confidence: 93 },
  { score: 4, label: '略微正面', confidence: 88 },
  { score: 0, label: '极端负面', confidence: 91 },
  { score: 3, label: '中性', confidence: 76 },
]

const score = ref<number | null>(null)
const label = ref('点击按钮开始')
const confidence = ref(0)

const scoreText = computed(() => score.value ?? '—')
const barWidth = computed(() => `${confidence.value}%`)

function predict() {
  const next = results[Math.floor(Math.random() * results.length)]
  score.value = next.score
  label.value = next.label
  confidence.value = next.confidence
}
</script>

<template>
  <div class="sentiment-demo">
    <p class="sentiment-demo__eyebrow">
      模拟情感预测请求
    </p>

    <button
      class="sentiment-demo__button"
      type="button"
      @click.stop="predict"
    >
      开始预测
    </button>

    <div class="sentiment-demo__result">
      <div class="sentiment-demo__score">
        {{ scoreText }}
      </div>
      <div class="sentiment-demo__label">
        {{ label }}
      </div>
      <p class="sentiment-demo__confidence">
        置信度 {{ confidence }}%
      </p>
      <div class="sentiment-demo__track" aria-hidden="true">
        <div
          class="sentiment-demo__bar"
          :style="{ width: barWidth }"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.sentiment-demo {
  width: min(100%, 24rem);
  margin: 2rem auto 0;
  padding: 1.5rem 1.4rem 1.35rem;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 12px 28px -24px rgba(15, 23, 42, 0.36);
  text-align: center;
}

.sentiment-demo__eyebrow {
  margin: 0 0 1rem;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--c-text-muted);
}

.sentiment-demo__button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 9999px;
  background: var(--c-primary);
  color: #fff;
  font-size: 0.88rem;
  font-weight: 600;
  line-height: 1;
  padding: 0.78rem 1.35rem;
  cursor: pointer;
  transition: transform 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease;
  box-shadow: 0 10px 24px -18px rgba(99, 102, 241, 0.6);
}

.sentiment-demo__button:hover {
  background: #5156ea;
  transform: translateY(-1px);
}

.sentiment-demo__button:active {
  transform: translateY(0);
}

.sentiment-demo__result {
  margin-top: 1.4rem;
}

.sentiment-demo__score {
  font-size: 2.75rem;
  font-weight: 700;
  line-height: 1;
  color: var(--c-text);
}

.sentiment-demo__label {
  margin-top: 0.55rem;
  font-size: 1rem;
  line-height: 1.5;
  color: var(--c-text);
}

.sentiment-demo__confidence {
  margin: 0.65rem 0 0.65rem;
  font-size: 0.84rem;
  line-height: 1.4;
  color: var(--c-text-muted);
}

.sentiment-demo__track {
  width: 11.5rem;
  height: 4px;
  margin: 0 auto;
  border-radius: 9999px;
  overflow: hidden;
  background: rgba(148, 163, 184, 0.2);
}

.sentiment-demo__bar {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--c-primary) 0%, var(--c-teal) 100%);
  transition: width 0.35s ease;
}
</style>
