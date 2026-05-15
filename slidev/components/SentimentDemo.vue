<script setup lang="ts">
import { computed, ref } from 'vue'

interface DemoResult {
  score: number
  labelZh: string
  labelEn: string
  confidence: number
  probabilities: number[]
  reasoning: string
}

interface DemoCase {
  name: string
  text: string
  result: DemoResult
}

const scoreLabels = ['极端负面', '明显负面', '略微负面', '中性', '略微正面', '极端正面']

const cases: DemoCase[] = [
  {
    name: '好评',
    text: '这次客服响应很快，问题也解决得很彻底，下次还会继续使用。',
    result: {
      score: 5,
      labelZh: '极端正面',
      labelEn: 'extremely_positive',
      confidence: 0.86,
      probabilities: [0.01, 0.01, 0.02, 0.03, 0.07, 0.86],
      reasoning: '文本包含“响应很快”“解决彻底”“继续使用”等强正向表达，模型将主要概率集中在 5 分档。',
    },
  },
  {
    name: '差评',
    text: '物流比预计晚了两天，包装也有破损，整体比较失望。',
    result: {
      score: 1,
      labelZh: '明显负面',
      labelEn: 'clearly_negative',
      confidence: 0.74,
      probabilities: [0.18, 0.74, 0.05, 0.02, 0.01, 0],
      reasoning: '“晚了两天”“破损”“失望”构成连续负向证据，但语气没有达到极端抱怨，因此落在 1 分档。',
    },
  },
  {
    name: '中性',
    text: '功能基本能用，但界面响应偶尔变慢，整体表现一般。',
    result: {
      score: 3,
      labelZh: '中性',
      labelEn: 'neutral',
      confidence: 0.57,
      probabilities: [0.03, 0.08, 0.18, 0.57, 0.11, 0.03],
      reasoning: '文本同时包含“基本能用”和“偶尔变慢”，正负证据相互抵消，概率峰值落在中性 3 分。',
    },
  },
  {
    name: '轻微正面',
    text: '更新后终于稳定了，虽然还有小瑕疵，但体验明显提升。',
    result: {
      score: 4,
      labelZh: '略微正面',
      labelEn: 'slightly_positive',
      confidence: 0.68,
      probabilities: [0.01, 0.03, 0.06, 0.16, 0.68, 0.06],
      reasoning: '“稳定”“明显提升”给出正向判断，“小瑕疵”降低极端正面概率，所以最终为 4 分。',
    },
  },
]

const activeIndex = ref(0)
const isPredicting = ref(false)

const activeCase = computed(() => cases[activeIndex.value])
const result = computed(() => activeCase.value.result)
const confidencePercent = computed(() => Math.round(result.value.confidence * 100))
const maxProbability = computed(() => Math.max(...result.value.probabilities))

function selectCase(index: number) {
  activeIndex.value = index
  isPredicting.value = true
  window.setTimeout(() => {
    isPredicting.value = false
  }, 260)
}

function nextCase() {
  selectCase((activeIndex.value + 1) % cases.length)
}

function toPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function barWidth(value: number) {
  const safeMax = maxProbability.value || 1
  return `${Math.max(6, (value / safeMax) * 100)}%`
}
</script>

<template>
  <div class="sentiment-demo" :class="{ 'is-loading': isPredicting }">
    <section class="sentiment-demo__input" aria-label="待预测文本">
      <div class="sentiment-demo__section-head">
        <span class="sentiment-demo__kicker">Input Text</span>
        <span class="sentiment-demo__model">BERT Student · 0-5</span>
      </div>

      <p class="sentiment-demo__text">
        “{{ activeCase.text }}”
      </p>

      <div class="sentiment-demo__chips" aria-label="示例文本">
        <button
          v-for="(item, index) in cases"
          :key="item.name"
          class="sentiment-demo__chip"
          :class="{ 'is-active': index === activeIndex }"
          type="button"
          @click.stop="selectCase(index)"
        >
          {{ item.name }}
        </button>
      </div>

      <button
        class="sentiment-demo__button"
        type="button"
        @click.stop="nextCase"
      >
        {{ isPredicting ? '预测中...' : '切换样例' }}
      </button>
    </section>

    <section class="sentiment-demo__result" aria-label="预测结果">
      <div class="sentiment-demo__summary">
        <div class="sentiment-demo__score">
          {{ result.score }}
        </div>
        <div>
          <div class="sentiment-demo__label">
            {{ result.labelZh }}
          </div>
          <div class="sentiment-demo__label-en">
            {{ result.labelEn }}
          </div>
        </div>
        <div class="sentiment-demo__confidence">
          <span>主置信度</span>
          <strong>{{ confidencePercent }}%</strong>
        </div>
      </div>

      <div class="sentiment-demo__probabilities">
        <div
          v-for="(probability, score) in result.probabilities"
          :key="score"
          class="sentiment-demo__prob-row"
          :class="{ 'is-winner': score === result.score }"
        >
          <span class="sentiment-demo__prob-label">{{ score }} · {{ scoreLabels[score] }}</span>
          <span class="sentiment-demo__prob-track">
            <span
              class="sentiment-demo__prob-bar"
              :style="{ width: barWidth(probability) }"
            />
          </span>
          <span class="sentiment-demo__prob-value">{{ toPercent(probability) }}</span>
        </div>
      </div>

      <p class="sentiment-demo__reasoning">
        {{ result.reasoning }}
      </p>
    </section>
  </div>
</template>

<style scoped>
.sentiment-demo {
  display: grid;
  grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr);
  gap: 1rem;
  width: min(100%, 50rem);
  margin: 1.5rem auto 0;
  color: var(--c-text);
}

.sentiment-demo__input,
.sentiment-demo__result {
  min-width: 0;
  border: 1px solid rgba(226, 232, 240, 0.92);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 14px 32px -28px rgba(15, 23, 42, 0.38);
}

.sentiment-demo__input {
  display: flex;
  flex-direction: column;
  padding: 1.1rem;
}

.sentiment-demo__result {
  padding: 1rem 1.1rem;
}

.sentiment-demo__section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  margin-bottom: 0.8rem;
}

.sentiment-demo__kicker,
.sentiment-demo__model {
  font-size: 0.64rem;
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.sentiment-demo__kicker {
  color: var(--c-primary);
}

.sentiment-demo__model {
  color: var(--c-text-muted);
}

.sentiment-demo__text {
  min-height: 7.2rem;
  margin: 0;
  padding: 0.9rem 0.95rem;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 7px;
  background: rgba(248, 250, 252, 0.86);
  color: var(--c-text);
  font-size: 1.02rem;
  font-weight: 600;
  line-height: 1.72;
  text-align: left;
}

.sentiment-demo__chips {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.42rem;
  margin-top: 0.8rem;
}

.sentiment-demo__chip,
.sentiment-demo__button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  cursor: pointer;
  transition: background-color 0.15s ease, color 0.15s ease, transform 0.15s ease;
}

.sentiment-demo__chip {
  min-height: 1.9rem;
  border-radius: 7px;
  background: rgba(241, 245, 249, 0.95);
  color: var(--c-text-muted);
  font-size: 0.72rem;
  font-weight: 700;
}

.sentiment-demo__chip.is-active {
  background: rgba(99, 102, 241, 0.12);
  color: var(--c-primary);
}

.sentiment-demo__button {
  width: 100%;
  min-height: 2.35rem;
  margin-top: auto;
  border-radius: 7px;
  background: var(--c-primary);
  color: #fff;
  font-size: 0.82rem;
  font-weight: 700;
}

.sentiment-demo__button:hover,
.sentiment-demo__chip:hover {
  transform: translateY(-1px);
}

.sentiment-demo__summary {
  display: grid;
  grid-template-columns: 4.2rem minmax(0, 1fr) 4.8rem;
  align-items: center;
  gap: 0.85rem;
}

.sentiment-demo__score {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 4.2rem;
  aspect-ratio: 1;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--c-primary) 0%, var(--c-teal) 100%);
  color: #fff;
  font-size: 2.35rem;
  font-weight: 800;
  line-height: 1;
}

.sentiment-demo__label {
  color: var(--c-text);
  font-size: 1.16rem;
  font-weight: 800;
  line-height: 1.2;
}

.sentiment-demo__label-en {
  margin-top: 0.25rem;
  color: var(--c-text-muted);
  font-size: 0.72rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  line-height: 1.2;
}

.sentiment-demo__confidence {
  text-align: right;
}

.sentiment-demo__confidence span {
  display: block;
  color: var(--c-text-muted);
  font-size: 0.66rem;
  line-height: 1.2;
}

.sentiment-demo__confidence strong {
  display: block;
  margin-top: 0.15rem;
  color: var(--c-text);
  font-size: 1.18rem;
  line-height: 1;
}

.sentiment-demo__probabilities {
  display: grid;
  gap: 0.42rem;
  margin-top: 0.9rem;
}

.sentiment-demo__prob-row {
  display: grid;
  grid-template-columns: 5.8rem minmax(0, 1fr) 3rem;
  align-items: center;
  gap: 0.55rem;
  min-height: 1.35rem;
  color: var(--c-text-muted);
  font-size: 0.68rem;
  line-height: 1;
}

.sentiment-demo__prob-row.is-winner {
  color: var(--c-text);
  font-weight: 800;
}

.sentiment-demo__prob-label,
.sentiment-demo__prob-value {
  white-space: nowrap;
}

.sentiment-demo__prob-track {
  height: 0.52rem;
  border-radius: 9999px;
  background: rgba(226, 232, 240, 0.85);
  overflow: hidden;
}

.sentiment-demo__prob-bar {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--c-primary) 0%, var(--c-teal) 100%);
  transition: width 0.32s ease;
}

.sentiment-demo__prob-value {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.sentiment-demo__reasoning {
  margin: 0.9rem 0 0;
  padding: 0.72rem 0.82rem;
  border-radius: 7px;
  background: rgba(99, 102, 241, 0.07);
  color: var(--c-text);
  font-size: 0.73rem;
  line-height: 1.58;
  text-align: left;
}

.sentiment-demo.is-loading .sentiment-demo__result {
  opacity: 0.62;
}
</style>
