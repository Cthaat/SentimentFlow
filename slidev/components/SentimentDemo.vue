<script setup lang="ts">
import { computed, ref } from 'vue'

interface DemoResult {
  score: number
  labelZh: string
  labelEn: string
  confidence: number
  probabilities: number[]
  reasoning: string
  source: string
  modelName: string
}

interface DemoCase {
  name: string
  text: string
  result: DemoResult
}

interface PredictResponse {
  text?: string
  score?: number
  label?: string
  label_zh?: string
  confidence?: number
  probabilities?: number[]
  reasoning?: string
  source?: string
  model_name?: string
  detail?: string
}

type DemoStatus = 'sample' | 'loading' | 'live' | 'error'

const scoreLabels = ['极端负面', '明显负面', '略微负面', '中性', '略微正面', '极端正面']
const directBackendCandidates = ['http://127.0.0.1:8846', 'http://localhost:8846']

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
      source: 'sample',
      modelName: 'demo_static',
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
      source: 'sample',
      modelName: 'demo_static',
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
      source: 'sample',
      modelName: 'demo_static',
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
      source: 'sample',
      modelName: 'demo_static',
    },
  },
]

const activeIndex = ref(0)
const status = ref<DemoStatus>('sample')
const errorMessage = ref('')
const customText = ref('我刚试了新版本，速度比以前快很多，但训练日志还可以再清晰一点。')
const currentText = ref(cases[0].text)
const result = ref<DemoResult>(cloneResult(cases[0].result))

const isPredicting = computed(() => status.value === 'loading')
const canPredict = computed(() => customText.value.trim().length > 0 && !isPredicting.value)
const confidencePercent = computed(() => Math.round(result.value.confidence * 100))
const maxProbability = computed(() => Math.max(...result.value.probabilities))
const resultSourceLabel = computed(() => {
  if (status.value === 'live') return '后端实时预测'
  if (status.value === 'loading') return '请求后端中'
  if (status.value === 'error') return '保留上次结果'
  return '演示样例'
})

function cloneResult(value: DemoResult): DemoResult {
  return {
    ...value,
    probabilities: [...value.probabilities],
  }
}

function selectCase(index: number) {
  const next = cases[index]
  activeIndex.value = index
  currentText.value = next.text
  result.value = cloneResult(next.result)
  status.value = 'sample'
  errorMessage.value = ''
}

function nextCase() {
  selectCase((activeIndex.value + 1) % cases.length)
}

async function predictCustomText() {
  const text = customText.value.trim()
  if (!text || isPredicting.value) return

  activeIndex.value = -1
  currentText.value = text
  status.value = 'loading'
  errorMessage.value = ''

  try {
    result.value = resultFromResponse(await requestPrediction(text))
    status.value = 'live'
  }
  catch (error) {
    status.value = 'error'
    errorMessage.value = error instanceof Error ? error.message : '后端预测请求失败'
  }
}

async function requestPrediction(text: string): Promise<PredictResponse> {
  const errors: string[] = []
  for (const endpoint of getPredictEndpoints()) {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const data = await readJsonResponse(response)
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`)
      }
      return data
    }
    catch (error) {
      const message = error instanceof Error ? error.message : 'request failed'
      errors.push(`${endpoint} -> ${message}`)
    }
  }

  throw new Error(`后端预测不可用：${errors.join('；')}`)
}

async function readJsonResponse(response: Response): Promise<PredictResponse> {
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    throw new Error(`返回不是 JSON：${response.status}`)
  }
  return await response.json() as PredictResponse
}

function resultFromResponse(data: PredictResponse): DemoResult {
  const score = clampScore(data.score)
  const confidence = clampProbability(data.confidence)
  const probabilities = normalizeProbabilities(data.probabilities, score, confidence)

  return {
    score,
    labelZh: data.label_zh || scoreLabels[score],
    labelEn: data.label || 'unknown',
    confidence: confidence || probabilities[score] || Math.max(...probabilities),
    probabilities,
    reasoning: data.reasoning || '后端返回了评分与概率分布，本次响应未提供额外解释。',
    source: data.source || 'backend',
    modelName: data.model_name || 'active_model',
  }
}

function normalizeProbabilities(values: number[] | undefined, score: number, confidence: number) {
  const probabilities = scoreLabels.map((_label, index) => clampProbability(values?.[index]))
  if (!probabilities.some(value => value > 0)) {
    probabilities[score] = confidence || 0.5
  }
  return probabilities
}

function clampScore(value: number | undefined) {
  if (!Number.isFinite(value)) return 3
  return Math.max(0, Math.min(5, Math.round(value as number)))
}

function clampProbability(value: number | undefined) {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value as number))
}

function getPredictEndpoints() {
  const endpoints = ['/api/predict/']
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location
    const directHost = normalizeLocalHost(hostname)
    endpoints.push(`${protocol}//${directHost}:8846/api/predict/`)
  }
  for (const baseUrl of directBackendCandidates) {
    endpoints.push(`${baseUrl}/api/predict/`)
  }
  return [...new Set(endpoints)]
}

function normalizeLocalHost(hostname: string) {
  if (!hostname || hostname === '0.0.0.0' || hostname === '::' || hostname === '[::]') {
    return '127.0.0.1'
  }
  return hostname
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
        “{{ currentText }}”
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

      <div class="sentiment-demo__custom">
        <label class="sentiment-demo__custom-label" for="live-sentiment-input">
          现场输入
        </label>
        <textarea
          id="live-sentiment-input"
          v-model="customText"
          class="sentiment-demo__textarea"
          rows="3"
          maxlength="200"
          placeholder="输入一段中文文本，现场请求后端预测"
          @click.stop
          @keydown.stop
          @keydown.ctrl.enter.prevent="predictCustomText"
        />
      </div>

      <div class="sentiment-demo__actions">
        <button
          class="sentiment-demo__button sentiment-demo__button--secondary"
          type="button"
          @click.stop="nextCase"
        >
          切换样例
        </button>
        <button
          class="sentiment-demo__button"
          type="button"
          :disabled="!canPredict"
          @click.stop="predictCustomText"
        >
          {{ isPredicting ? '预测中...' : '现场预测' }}
        </button>
      </div>

      <p v-if="errorMessage" class="sentiment-demo__error">
        {{ errorMessage }}
      </p>
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

      <div class="sentiment-demo__source-row">
        <span>{{ resultSourceLabel }}</span>
        <span>{{ result.source }}</span>
        <span>{{ result.modelName }}</span>
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
  grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
  gap: 0.9rem;
  width: min(100%, 51rem);
  margin: 1.2rem auto 0;
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
  padding: 0.95rem;
}

.sentiment-demo__result {
  padding: 0.95rem 1rem;
}

.sentiment-demo__section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  margin-bottom: 0.65rem;
}

.sentiment-demo__kicker,
.sentiment-demo__model {
  font-size: 0.61rem;
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
  min-height: 4.6rem;
  margin: 0;
  padding: 0.72rem 0.8rem;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 7px;
  background: rgba(248, 250, 252, 0.86);
  color: var(--c-text);
  font-size: 0.86rem;
  font-weight: 650;
  line-height: 1.62;
  text-align: left;
}

.sentiment-demo__chips {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.38rem;
  margin-top: 0.58rem;
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
  min-height: 1.62rem;
  border-radius: 7px;
  background: rgba(241, 245, 249, 0.95);
  color: var(--c-text-muted);
  font-size: 0.66rem;
  font-weight: 700;
}

.sentiment-demo__chip.is-active {
  background: rgba(99, 102, 241, 0.12);
  color: var(--c-primary);
}

.sentiment-demo__custom {
  margin-top: 0.7rem;
}

.sentiment-demo__custom-label {
  display: block;
  margin-bottom: 0.34rem;
  color: var(--c-text);
  font-size: 0.68rem;
  font-weight: 800;
}

.sentiment-demo__textarea {
  width: 100%;
  min-height: 4.1rem;
  max-height: 4.1rem;
  resize: none;
  border: 1px solid rgba(203, 213, 225, 0.95);
  border-radius: 7px;
  background: #fff;
  color: var(--c-text);
  font-size: 0.72rem;
  line-height: 1.48;
  padding: 0.55rem 0.65rem;
  outline: none;
}

.sentiment-demo__textarea:focus {
  border-color: rgba(99, 102, 241, 0.55);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.sentiment-demo__actions {
  display: grid;
  grid-template-columns: 0.82fr 1fr;
  gap: 0.5rem;
  margin-top: 0.62rem;
}

.sentiment-demo__button {
  min-height: 2.05rem;
  border-radius: 7px;
  background: var(--c-primary);
  color: #fff;
  font-size: 0.76rem;
  font-weight: 800;
}

.sentiment-demo__button--secondary {
  background: rgba(241, 245, 249, 0.95);
  color: var(--c-text);
}

.sentiment-demo__button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.sentiment-demo__button:not(:disabled):hover,
.sentiment-demo__chip:hover {
  transform: translateY(-1px);
}

.sentiment-demo__error {
  margin: 0.48rem 0 0;
  padding: 0.42rem 0.55rem;
  border-radius: 6px;
  background: rgba(244, 63, 94, 0.08);
  color: #be123c;
  font-size: 0.66rem;
  line-height: 1.42;
  text-align: left;
}

.sentiment-demo__summary {
  display: grid;
  grid-template-columns: 3.8rem minmax(0, 1fr) 4.4rem;
  align-items: center;
  gap: 0.72rem;
}

.sentiment-demo__score {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 3.8rem;
  aspect-ratio: 1;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--c-primary) 0%, var(--c-teal) 100%);
  color: #fff;
  font-size: 2.08rem;
  font-weight: 800;
  line-height: 1;
}

.sentiment-demo__label {
  color: var(--c-text);
  font-size: 1.05rem;
  font-weight: 800;
  line-height: 1.2;
}

.sentiment-demo__label-en {
  margin-top: 0.22rem;
  color: var(--c-text-muted);
  font-size: 0.66rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  line-height: 1.2;
}

.sentiment-demo__confidence {
  text-align: right;
}

.sentiment-demo__confidence span {
  display: block;
  color: var(--c-text-muted);
  font-size: 0.62rem;
  line-height: 1.2;
}

.sentiment-demo__confidence strong {
  display: block;
  margin-top: 0.15rem;
  color: var(--c-text);
  font-size: 1.08rem;
  line-height: 1;
}

.sentiment-demo__source-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.34rem;
  margin-top: 0.62rem;
}

.sentiment-demo__source-row span {
  display: inline-flex;
  align-items: center;
  min-height: 1.25rem;
  border-radius: 9999px;
  background: rgba(99, 102, 241, 0.08);
  color: var(--c-text-muted);
  font-size: 0.58rem;
  font-weight: 700;
  line-height: 1;
  padding: 0 0.48rem;
}

.sentiment-demo__probabilities {
  display: grid;
  gap: 0.36rem;
  margin-top: 0.72rem;
}

.sentiment-demo__prob-row {
  display: grid;
  grid-template-columns: 5.8rem minmax(0, 1fr) 3rem;
  align-items: center;
  gap: 0.5rem;
  min-height: 1.18rem;
  color: var(--c-text-muted);
  font-size: 0.65rem;
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
  height: 0.46rem;
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
  margin: 0.74rem 0 0;
  padding: 0.62rem 0.72rem;
  border-radius: 7px;
  background: rgba(99, 102, 241, 0.07);
  color: var(--c-text);
  font-size: 0.68rem;
  line-height: 1.5;
  text-align: left;
}

.sentiment-demo.is-loading .sentiment-demo__result {
  opacity: 0.64;
}
</style>
