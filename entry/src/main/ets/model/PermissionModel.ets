import { systemDateTime } from '@kit.BasicServicesKit';
import { devices, load as loadHiAI, predict as predictHiAI, HiAISession } from 'libpermission_hiai.so';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { resourceManager } from '@kit.LocalizationKit';
import { util } from '@kit.ArkTS';
import { SEQUENCE_LENGTH, TokenizedInput, WordPiece } from './WordPiece';

const MODEL_FILE: string = 'permission.ms';
const BUCKETS: number[] = [16, 32, 64];
const LABEL_COUNT: number = 5;

interface ReferenceCase {
  text: string;
  logits: number[];
}

interface ModelConfig {
  thresholds: number[];
}

export class InferenceResult {
  scores: number[];
  thresholds: number[];
  elapsedMs: number;
  backend: string;
  constructor(scores: number[], thresholds: number[], elapsedMs: number, backend: string) {
    this.scores = scores;
    this.thresholds = thresholds;
    this.elapsedMs = elapsedMs;
    this.backend = backend;
  }
}

class BenchmarkRunner {
  name: string;
  run: () => Promise<number[]>;
  constructor(name: string, run: () => Promise<number[]>) {
    this.name = name;
    this.run = run;
  }
}

export class PermissionModel {
  private model?: mindSporeLite.Model;
  private tokenizer?: WordPiece;
  private hiaiSessions: Map<number, HiAISession> = new Map<number, HiAISession>();
  private unavailable: Set<number> = new Set<number>();
  private hasHiAI: boolean = false;
  private resources?: resourceManager.ResourceManager;
  diagnostic: string = '';
  private thresholds: number[] = [];

  async load(resources: resourceManager.ResourceManager): Promise<void> {
    if (this.tokenizer !== undefined) {
      return;
    }
    const decoder: util.TextDecoder = util.TextDecoder.create('utf-8');
    const vocabBytes: Uint8Array = await resources.getRawFileContent('vocab.txt');
    const configBytes: Uint8Array = await resources.getRawFileContent('model_config.json');
    const config: ModelConfig = JSON.parse(decoder.decodeToString(configBytes)) as ModelConfig;
    if (config.thresholds.length !== LABEL_COUNT ||
        config.thresholds.some((threshold: number): boolean => !(threshold > 0 && threshold < 1))) {
      throw new Error('模型阈值配置无效');
    }
    const tokenizer: WordPiece = new WordPiece(decoder.decodeToString(vocabBytes));
    this.resources = resources;
    this.thresholds = config.thresholds;
    try {
      const names: string[] = devices();
      this.diagnostic = 'NNRT 设备 ' + (names.length > 0 ? names.join('、') : '无');
      if (!names.some((name: string): boolean => name.startsWith('HIAI_F (ID '))) {
        throw new Error('未枚举到 HIAI_F');
      }
      this.hasHiAI = true;
      this.diagnostic += '\nHiAI 按有效 token 选择 16 / 32 / 64，首次使用时加载';
    } catch (error) {
      this.recordHiAIError(error as Error);
      await this.loadCpu(resources);
    }
    this.tokenizer = tokenizer;
  }

  private async loadSession(resources: resourceManager.ResourceManager, tokenizer: WordPiece,
    bucket: number): Promise<HiAISession> {
    const cached: HiAISession | undefined = this.hiaiSessions.get(bucket);
    if (cached !== undefined) {
      return cached;
    }
    const suffix: string = bucket === 0 ? 'baseline' : bucket.toString();
    const raw: Uint8Array = await resources.getRawFileContent('permission_' + suffix + '.om');
    const buffer: ArrayBuffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const session: HiAISession = await loadHiAI(buffer);
    const report: string = await this.checkReference(resources, tokenizer, session, bucket === 0 ? 64 : bucket);
    this.hiaiSessions.set(bucket, session);
    this.diagnostic += '\n' + (bucket === 0 ? '基线' : bucket.toString() + ' token') + ' ' + report;
    return session;
  }

  private async checkReference(resources: resourceManager.ResourceManager, tokenizer: WordPiece,
    session: HiAISession, length: number): Promise<string> {
    const bytes: Uint8Array = await resources.getRawFileContent('hiai_reference.json');
    const decoder: util.TextDecoder = util.TextDecoder.create('utf-8');
    const cases: ReferenceCase[] = JSON.parse(decoder.decodeToString(bytes)) as ReferenceCase[];
    let matched: number = 0;
    let checked: number = 0;
    let maxError: number = 0;
    for (const item of cases) {
      const tokens: TokenizedInput = tokenizer.encode(item.text);
      if (this.tokenCount(tokens) > length) {
        continue;
      }
      checked++;
      const actual: number[] = await predictHiAI(session, tokens.ids, tokens.mask);
      let sameLabels: boolean = true;
      for (let i = 0; i < LABEL_COUNT; i++) {
        const score: number = 1 / (1 + Math.exp(-actual[i]));
        const reference: number = 1 / (1 + Math.exp(-item.logits[i]));
        if ((score >= this.thresholds[i]) !== (reference >= this.thresholds[i])) {
          sameLabels = false;
        }
        maxError = Math.max(maxError, Math.abs(actual[i] - item.logits[i]));
      }
      if (sameLabels) {
        matched++;
      }
    }
    const report: string = 'HiAI 自检 ' + matched.toString() + '/' + checked.toString() +
      ' · 最大 logit 差 ' + maxError.toFixed(4);
    if (checked === 0 || matched !== checked) {
      throw new Error(report);
    }
    return report;
  }

  private recordHiAIError(error: Error): void {
    const reason: string = error.message || '设备或模型不可用';
    this.diagnostic += (this.diagnostic.length > 0 ? '\n' : '') + 'HiAI 回退原因 ' + reason;
    console.warn('PermissionHiAI fallback: ' + reason);
  }

  private async loadCpu(resources: resourceManager.ResourceManager): Promise<void> {
    if (this.model !== undefined) {
      return;
    }
    if (!canIUse('SystemCapability.AI.MindSporeLite')) {
      throw new Error('当前设备不支持 MindSpore Lite CPU 回退');
    }
    const bytes: Uint8Array = await resources.getRawFileContent(MODEL_FILE);
    const buffer: ArrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const context: mindSporeLite.Context = {
      target: ['cpu'],
      cpu: { threadNum: 2, precisionMode: 'enforce_fp32' }
    };
    const model: mindSporeLite.Model = await mindSporeLite.loadModelFromBuffer(buffer, context);
    const inputs: mindSporeLite.MSTensor[] = model.getInputs();
    if (inputs.length !== 2 || !inputs.some((tensor: mindSporeLite.MSTensor): boolean => tensor.name === 'input_ids') ||
        !inputs.some((tensor: mindSporeLite.MSTensor): boolean => tensor.name === 'attention_mask')) {
      throw new Error('模型输入名称不符合预期');
    }
    for (const tensor of inputs) {
      if (tensor.shape.length !== 2 || tensor.shape[0] !== 1 || tensor.shape[1] !== SEQUENCE_LENGTH ||
          tensor.dtype !== mindSporeLite.DataType.NUMBER_TYPE_INT32) {
        throw new Error('模型输入形状或类型不符合预期：' + tensor.name);
      }
    }
    this.model = model;
  }

  async predict(text: string): Promise<InferenceResult> {
    const tokenizer: WordPiece | undefined = this.tokenizer;
    if (tokenizer === undefined || this.resources === undefined) {
      throw new Error('模型尚未加载');
    }
    const started: number = Date.now();
    const tokens: TokenizedInput = tokenizer.encode(text);
    if (tokens.tooLong) {
      throw new Error('指令过长，请缩短后重试');
    }
    if (tokens.mask[2] === 0) {
      throw new Error('请输入有效指令');
    }
    if (this.hasHiAI) {
      const count: number = this.tokenCount(tokens);
      for (const bucket of BUCKETS) {
        if (bucket < count || this.unavailable.has(bucket)) {
          continue;
        }
        try {
          const hiai: HiAISession = await this.loadSession(this.resources, tokenizer, bucket);
          const logits: number[] = await predictHiAI(hiai, tokens.ids, tokens.mask);
          return this.result(logits, started, hiai.device + ' · ' + bucket.toString() + ' token');
        } catch (error) {
          this.hiaiSessions.delete(bucket);
          this.unavailable.add(bucket);
          this.recordHiAIError(new Error(bucket.toString() + ' token ' + (error as Error).message));
        }
      }
    }
    const logits: number[] = await this.predictCpu(tokens, this.resources);
    return this.result(logits, started, 'CPU · MindSpore Lite');
  }

  private tokenCount(tokens: TokenizedInput): number {
    return tokens.mask.reduce((sum: number, value: number): number => sum + value, 0);
  }

  private async predictCpu(tokens: TokenizedInput, resources: resourceManager.ResourceManager): Promise<number[]> {
    await this.loadCpu(resources);
    const model: mindSporeLite.Model | undefined = this.model;
    if (model === undefined) {
      throw new Error('CPU 模型未加载');
    }
    const inputs: mindSporeLite.MSTensor[] = model.getInputs();
    for (const tensor of inputs) {
      tensor.setData(tensor.name === 'input_ids' ? tokens.ids.buffer : tokens.mask.buffer);
    }
    const outputs: mindSporeLite.MSTensor[] = await model.predict(inputs);
    if (outputs.length !== 1 || outputs[0].elementNum !== LABEL_COUNT ||
        outputs[0].dtype !== mindSporeLite.DataType.NUMBER_TYPE_FLOAT32) {
      throw new Error('模型输出不符合预期');
    }
    const logits: Float32Array = new Float32Array(outputs[0].getData());
    return Array.from(logits);
  }

  async benchmark(text: string): Promise<string> {
    const tokenizer: WordPiece | undefined = this.tokenizer;
    const resources: resourceManager.ResourceManager | undefined = this.resources;
    if (tokenizer === undefined || resources === undefined) {
      throw new Error('模型尚未加载');
    }
    const tokens: TokenizedInput = tokenizer.encode(text);
    if (tokens.tooLong || tokens.mask[2] === 0) {
      throw new Error('请输入有效指令');
    }
    const count: number = this.tokenCount(tokens);
    const runners: BenchmarkRunner[] = [];
    const warnings: string[] = [];
    if (this.hasHiAI) {
      for (const bucket of [0, 16, 32, 64]) {
        if (bucket > 0 && bucket < count) {
          continue;
        }
        try {
          const session: HiAISession = await this.loadSession(resources, tokenizer, bucket);
          const name: string = bucket === 0 ? '旧版 NPU 64 token' : '优化版 NPU ' + bucket.toString() + ' token';
          runners.push(new BenchmarkRunner(name, (): Promise<number[]> => predictHiAI(session, tokens.ids, tokens.mask)));
        } catch (error) {
          warnings.push('HiAI ' + (bucket === 0 ? '基线' : bucket.toString()) + '：' + (error as Error).message);
        }
      }
    }
    try {
      await this.loadCpu(resources);
      runners.push(new BenchmarkRunner('CPU 64 token', (): Promise<number[]> => this.predictCpu(tokens, resources)));
    } catch (error) {
      warnings.push('CPU 64 token：' + (error as Error).message);
    }
    const result: string[] = ['有效 token ' + count.toString() + ' · 每组预热 5 次，记录 50 次'];
    for (const runner of runners) {
      try {
        for (let i: number = 0; i < 5; i++) {
          await runner.run();
        }
        const times: number[] = [];
        for (let i: number = 0; i < 50; i++) {
          const start: number = systemDateTime.getUptime(systemDateTime.TimeType.ACTIVE, true);
          await runner.run();
          times.push((systemDateTime.getUptime(systemDateTime.TimeType.ACTIVE, true) - start) / 1000000);
        }
        times.sort((a: number, b: number): number => a - b);
        result.push(runner.name + ' · P50 ' + ((times[24] + times[25]) / 2).toFixed(1) +
          ' ms · P95 ' + times[47].toFixed(1) + ' ms');
      } catch (error) {
        warnings.push(runner.name + '：' + (error as Error).message);
      }
    }
    return result.concat(warnings).join('\n');
  }

  private result(logits: number[], started: number, backend: string): InferenceResult {
    if (logits.length !== LABEL_COUNT || logits.some((logit: number): boolean => !Number.isFinite(logit))) {
      throw new Error('模型输出不符合预期');
    }
    const scores: number[] = logits.map((logit: number): number => 1 / (1 + Math.exp(-logit)));
    return new InferenceResult(scores, this.thresholds, Date.now() - started, backend);
  }
}
