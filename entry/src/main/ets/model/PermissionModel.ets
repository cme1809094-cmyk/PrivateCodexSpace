import { systemDateTime } from '@kit.BasicServicesKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { devices, accelerator, load as loadHiAI, predict as predictHiAI, dispose as disposeHiAI, HiAISession } from 'libpermission_hiai.so';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { resourceManager } from '@kit.LocalizationKit';
import { util } from '@kit.ArkTS';
import { SEQUENCE_LENGTH, TokenizedInput, WordPiece } from './WordPiece';

const BUCKETS: number[] = [16, 32, 64];
const LABEL_COUNT: number = 5;
const APP_VERSION: string = '1.1.0';

interface ReferenceCase {
  text: string;
  logits: number[];
}
interface ModelConfig {
  thresholds: number[];
}
export interface ModelAsset {
  bucket: number;
  bytes: number;
  sha256: string;
  referenceCases: number;
}
export interface ModelProfile {
  id: string;
  name: string;
  assetPrefix: string;
  maxLogitError: number;
  assets: ModelAsset[];
}
interface ModelCatalog {
  defaultModel: string;
  profiles: ModelProfile[];
}
class ReferenceResult {
  checked: number = 0;
  matched: number = 0;
  maxLogitError: number = 0;
}
class TimingResult {
  warmupRuns: number = 5;
  sampleRuns: number = 50;
  p50Ms: number = 0;
  p95Ms: number = 0;
}
export class BucketAcceptance {
  bucket: number = 0;
  expectedBytes: number = 0;
  manifestSha256: string = '';
  bytesVerified: boolean = false;
  expectedCases: number = 0;
  device: string = '';
  checked: number = 0;
  matched: number = 0;
  maxLogitError: number = 0;
  timing?: TimingResult;
  passed: boolean = false;
  error: string = '';
}
export class AcceptanceReport {
  schemaVersion: number = 1;
  appVersion: string = APP_VERSION;
  startedAt: string = new Date().toISOString();
  finishedAt: string = '';
  modelId: string = '';
  modelName: string = '';
  runtimeBackend: string = '';
  thresholds: number[] = [];
  maxLogitErrorLimit: number = 0;
  cpuFallbackAllowed: boolean = false;
  scope: string = 'NPU conversion parity and warm inference latency; not task accuracy or sustained performance';
  buckets: BucketAcceptance[] = [];
  passed: boolean = false;
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
export class PermissionModel {
  private model?: mindSporeLite.Model;
  private tokenizer?: WordPiece;
  private profile?: ModelProfile;
  private references: ReferenceCase[] = [];
  private thresholds: number[] = [];
  private hiaiSessions: Map<number, HiAISession> = new Map<number, HiAISession>();
  private selfTests: Map<number, ReferenceResult> = new Map<number, ReferenceResult>();
  private unavailable: Set<number> = new Set<number>();
  private hasHiAI: boolean = false;
  private runtimeBackend: string = '';
  private resources?: resourceManager.ResourceManager;
  private working: boolean = false;
  profiles: ModelProfile[] = [];
  modelId: string = '';
  modelName: string = '';
  diagnostic: string = '';

  isReady(): boolean {
    return this.tokenizer !== undefined;
  }

  private begin(): void {
    if (this.working) {
      throw new Error('模型正在工作，请稍后重试');
    }
    this.working = true;
  }

  private async json(resources: resourceManager.ResourceManager, name: string): Promise<string> {
    const bytes: Uint8Array = await resources.getRawFileContent(name);
    return util.TextDecoder.create('utf-8').decodeToString(bytes);
  }

  async load(resources: resourceManager.ResourceManager): Promise<void> {
    this.begin();
    try {
      if (this.isReady()) {
        return;
      }
      const catalog: ModelCatalog = JSON.parse(await this.json(resources, 'model_catalog.json')) as ModelCatalog;
      if (catalog.profiles.length === 0 ||
          new Set<string>(catalog.profiles.map((item: ModelProfile): string => item.id)).size !== catalog.profiles.length) {
        throw new Error('模型目录无效');
      }
      for (const item of catalog.profiles) {
        if (!item.id || !item.name || item.assetPrefix.includes('..') || item.assetPrefix.startsWith('/') ||
            (item.assetPrefix.length > 0 && !item.assetPrefix.endsWith('/')) ||
            !Number.isFinite(item.maxLogitError) || item.maxLogitError <= 0 ||
            !BUCKETS.every((bucket: number): boolean => item.assets.filter((asset: ModelAsset): boolean => asset.bucket === bucket).length === 1) ||
            item.assets.some((asset: ModelAsset): boolean => ![0, 16, 32, 64].includes(asset.bucket) ||
              !Number.isInteger(asset.bytes) || asset.bytes <= 0 || !/^[a-f0-9]{64}$/.test(asset.sha256) ||
              !Number.isInteger(asset.referenceCases) || asset.referenceCases <= 0)) {
          throw new Error('模型配置无效：' + item.id);
        }
      }
      const selected: ModelProfile | undefined = catalog.profiles.find((item: ModelProfile): boolean => item.id === catalog.defaultModel);
      if (selected === undefined) {
        throw new Error('默认模型不存在');
      }
      this.profiles = catalog.profiles;
      this.resources = resources;
      await this.activate(selected, resources);
    } finally {
      this.working = false;
    }
  }

  async switchModel(id: string): Promise<void> {
    this.begin();
    try {
      const resources: resourceManager.ResourceManager | undefined = this.resources;
      const profile: ModelProfile | undefined = this.profiles.find((item: ModelProfile): boolean => item.id === id);
      if (resources === undefined || profile === undefined) {
        throw new Error('模型不存在：' + id);
      }
      if (id !== this.modelId || !this.isReady()) {
        await this.activate(profile, resources);
      }
    } finally {
      this.working = false;
    }
  }

  private async activate(profile: ModelProfile, resources: resourceManager.ResourceManager): Promise<void> {
    // Validate paired metadata before replacing the current profile.
    const vocab: string = await this.json(resources, profile.assetPrefix + 'vocab.txt');
    const config: ModelConfig = JSON.parse(await this.json(resources, profile.assetPrefix + 'model_config.json')) as ModelConfig;
    const references: ReferenceCase[] = JSON.parse(await this.json(resources, profile.assetPrefix + 'hiai_reference.json')) as ReferenceCase[];
    if (config.thresholds.length !== LABEL_COUNT ||
        config.thresholds.some((threshold: number): boolean => !(threshold > 0 && threshold < 1)) || references.length === 0) {
      throw new Error('模型阈值或自检配置无效');
    }
    for (const item of references) {
      this.validateLogits(item.logits);
    }
    const tokenizer: WordPiece = new WordPiece(vocab);
    await this.releaseSessions();
    // The public MindSpore Lite JS Model has no dispose API; drop its reference.
    this.model = undefined;
    this.tokenizer = undefined;
    this.profile = profile;
    this.modelId = profile.id;
    this.modelName = profile.name;
    this.references = references;
    this.thresholds = config.thresholds.slice();
    this.selfTests.clear();
    this.unavailable.clear();
    this.hasHiAI = false;
    this.runtimeBackend = '';
    this.diagnostic = profile.name;
    hilog.info(0xFF00, 'PermissionNPU', '[PNPU][MODEL] model=%{public}s version=%{public}s selected', profile.id, APP_VERSION);
    try {
      const names: string[] = devices();
      this.diagnostic += '\nNNRT 设备 ' + (names.length > 0 ? names.join('、') : '无');
      const selected: string = accelerator();
      this.runtimeBackend = selected;
      this.hasHiAI = true;
      this.diagnostic += '\nCANN 离线模型设备 ' + selected + '\nNPU 按档位首次使用时加载';
    } catch (error) {
      this.recordHiAIError(error as Error | string);
      await this.loadCpu(resources);
    }
    this.tokenizer = tokenizer;
  }

  private asset(bucket: number): ModelAsset {
    const asset: ModelAsset | undefined = this.profile?.assets.find((item: ModelAsset): boolean => item.bucket === bucket);
    if (asset === undefined) {
      throw new Error('当前模型没有此档位');
    }
    return asset;
  }

  private async releaseBucket(bucket: number): Promise<void> {
    const session: HiAISession | undefined = this.hiaiSessions.get(bucket);
    if (session !== undefined) {
      await disposeHiAI(session);
      this.hiaiSessions.delete(bucket);
    }
  }

  private async releaseSessions(): Promise<void> {
    for (const bucket of Array.from(this.hiaiSessions.keys())) {
      await this.releaseBucket(bucket);
    }
  }

  private async loadSession(bucket: number): Promise<HiAISession> {
    const cached: HiAISession | undefined = this.hiaiSessions.get(bucket);
    if (cached !== undefined) {
      return cached;
    }
    const resources: resourceManager.ResourceManager | undefined = this.resources;
    const profile: ModelProfile | undefined = this.profile;
    if (resources === undefined || profile === undefined) {
      throw new Error('模型尚未加载');
    }
    // Keep at most one NPU bucket resident, including during benchmarking.
    await this.releaseSessions();
    const asset: ModelAsset = this.asset(bucket);
    const suffix: string = bucket === 0 ? 'baseline' : bucket.toString();
    const file: string = profile.assetPrefix + 'permission_' + suffix + '.om';
    const raw: Uint8Array = await resources.getRawFileContent(file);
    if (raw.byteLength !== asset.bytes) {
      throw new Error(file + ' 字节数不符，请检查 Git LFS 和模型文件');
    }
    const buffer: ArrayBuffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    hilog.info(0xFF00, 'PermissionNPU', '[PNPU][LOAD] model=%{public}s bucket=%{public}d file=%{public}s bytes=%{public}d begin',
      this.modelId, bucket, file, buffer.byteLength);
    const session: HiAISession = await loadHiAI(buffer);
    try {
      const report: ReferenceResult = await this.checkReference(session, bucket);
      this.selfTests.set(bucket, report);
      if (report.checked !== asset.referenceCases || report.matched !== report.checked || report.maxLogitError > profile.maxLogitError) {
        throw new Error('NPU 自检 ' + report.matched.toString() + '/' + report.checked.toString() +
          '（应检查 ' + asset.referenceCases.toString() + '），最大 logit 差 ' + report.maxLogitError.toFixed(6) +
          '，上限 ' + profile.maxLogitError.toString());
      }
      this.hiaiSessions.set(bucket, session);
      hilog.info(0xFF00, 'PermissionNPU', '[PNPU][LOAD] model=%{public}s bucket=%{public}d ready device=%{public}s',
        this.modelId, bucket, session.device);
      this.diagnostic += '\n' + suffix + ' token 自检 ' + report.matched.toString() + '/' + report.checked.toString();
      return session;
    } catch (error) {
      await disposeHiAI(session);
      throw error;
    }
  }

  private async checkReference(session: HiAISession, bucket: number): Promise<ReferenceResult> {
    const tokenizer: WordPiece | undefined = this.tokenizer;
    if (tokenizer === undefined) {
      throw new Error('模型尚未加载');
    }
    const report: ReferenceResult = new ReferenceResult();
    for (const item of this.references) {
      const tokens: TokenizedInput = tokenizer.encode(item.text);
      if (tokens.tooLong || this.tokenCount(tokens) > (bucket === 0 ? 64 : bucket)) {
        continue;
      }
      const actual: number[] = await predictHiAI(session, tokens.ids, tokens.mask);
      this.validateLogits(actual);
      report.checked++;
      let sameLabels: boolean = true;
      for (let i: number = 0; i < LABEL_COUNT; i++) {
        const score: number = 1 / (1 + Math.exp(-actual[i]));
        const reference: number = 1 / (1 + Math.exp(-item.logits[i]));
        if ((score >= this.thresholds[i]) !== (reference >= this.thresholds[i])) {
          sameLabels = false;
        }
        report.maxLogitError = Math.max(report.maxLogitError, Math.abs(actual[i] - item.logits[i]));
      }
      if (sameLabels) {
        report.matched++;
      }
    }
    hilog.info(0xFF00, 'PermissionNPU', '[PNPU][SELFTEST] model=%{public}s bucket=%{public}d matched=%{public}d checked=%{public}d maxError=%{public}s',
      this.modelId, bucket, report.matched, report.checked, report.maxLogitError.toFixed(6));
    return report;
  }

  private errorMessage(error: Error | string): string {
    return typeof error === 'string' ? error : (error?.message || '设备或模型不可用');
  }

  private recordHiAIError(error: Error | string): void {
    const reason: string = this.errorMessage(error);
    this.diagnostic += '\nHiAI 回退原因 ' + reason;
    hilog.warn(0xFF00, 'PermissionNPU', '[PNPU][FALLBACK] model=%{public}s reason=%{public}s', this.modelId, reason);
  }

  private async loadCpu(resources: resourceManager.ResourceManager): Promise<void> {
    if (this.model !== undefined) {
      return;
    }
    if (!canIUse('SystemCapability.AI.MindSporeLite')) {
      throw new Error('当前设备不支持 MindSpore Lite CPU 回退');
    }
    const bytes: Uint8Array = await resources.getRawFileContent((this.profile?.assetPrefix || '') + 'permission.ms');
    const buffer: ArrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const context: mindSporeLite.Context = {
      target: ['cpu'],
      cpu: { threadNum: 2, precisionMode: 'enforce_fp32' }
    };
    hilog.info(0xFF00, 'PermissionNPU', '[PNPU][CPU] model=%{public}s load begin', this.modelId);
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

  private encode(text: string, bucket: number): TokenizedInput {
    if (this.tokenizer === undefined || this.resources === undefined) {
      throw new Error('模型尚未加载');
    }
    if (![0, 16, 32, 64].includes(bucket)) {
      throw new Error('token 档位无效');
    }
    const tokens: TokenizedInput = this.tokenizer.encode(text);
    if (tokens.tooLong) {
      throw new Error('指令过长，请缩短后重试');
    }
    if (tokens.mask[2] === 0) {
      throw new Error('请输入有效指令');
    }
    if (bucket > 0 && this.tokenCount(tokens) > bucket) {
      throw new Error('有效 token 超过 ' + bucket.toString() + '（含 CLS/SEP），请选择更大档位或自动');
    }
    return tokens;
  }

  async predict(text: string, selectedBucket: number = 0): Promise<InferenceResult> {
    this.begin();
    try {
      const started: number = Date.now();
      const tokens: TokenizedInput = this.encode(text, selectedBucket);
      if (this.hasHiAI) {
        for (const bucket of (selectedBucket === 0 ? BUCKETS : [selectedBucket])) {
          if (bucket < this.tokenCount(tokens) || this.unavailable.has(bucket)) {
            continue;
          }
          try {
            const hiai: HiAISession = await this.loadSession(bucket);
            const logits: number[] = await predictHiAI(hiai, tokens.ids, tokens.mask);
            return this.result(logits, started, hiai.device + ' · ' + bucket.toString() + ' token');
          } catch (error) {
            await this.releaseBucket(bucket);
            this.unavailable.add(bucket);
            this.recordHiAIError(bucket.toString() + ' token ' + this.errorMessage(error as Error | string));
          }
        }
      }
      const logits: number[] = await this.predictCpu(tokens);
      return this.result(logits, started, 'CPU · MindSpore Lite · 64 token');
    } finally {
      this.working = false;
    }
  }

  private tokenCount(tokens: TokenizedInput): number {
    return tokens.mask.reduce((sum: number, value: number): number => sum + value, 0);
  }

  private async predictCpu(tokens: TokenizedInput): Promise<number[]> {
    if (this.resources === undefined) {
      throw new Error('模型尚未加载');
    }
    await this.loadCpu(this.resources);
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
    const logits: number[] = Array.from(new Float32Array(outputs[0].getData()));
    this.validateLogits(logits);
    return logits;
  }

  private async measure(run: () => Promise<number[]>): Promise<TimingResult> {
    const result: TimingResult = new TimingResult();
    for (let i: number = 0; i < result.warmupRuns; i++) {
      this.validateLogits(await run());
    }
    const times: number[] = [];
    for (let i: number = 0; i < result.sampleRuns; i++) {
      const start: number = systemDateTime.getUptime(systemDateTime.TimeType.ACTIVE, true);
      this.validateLogits(await run());
      times.push((systemDateTime.getUptime(systemDateTime.TimeType.ACTIVE, true) - start) / 1000000);
    }
    times.sort((a: number, b: number): number => a - b);
    result.p50Ms = (times[24] + times[25]) / 2;
    result.p95Ms = times[47];
    return result;
  }

  async benchmark(text: string, selectedBucket: number = 0): Promise<string> {
    this.begin();
    try {
      const tokens: TokenizedInput = this.encode(text, selectedBucket);
      const result: string[] = [this.modelName + ' · 有效 token ' + this.tokenCount(tokens).toString() + ' · 预热 5 次，测量 50 次'];
      const buckets: number[] = selectedBucket > 0 ? [selectedBucket] : (this.profile?.assets.map((asset: ModelAsset): number => asset.bucket) || []);
      if (this.hasHiAI) {
        for (const bucket of buckets) {
          if (bucket > 0 && bucket < this.tokenCount(tokens)) {
            continue;
          }
          const name: string = bucket === 0 ? '旧版 NPU 64 token' : 'NPU ' + bucket.toString() + ' token';
          try {
            const session: HiAISession = await this.loadSession(bucket);
            const timing: TimingResult = await this.measure((): Promise<number[]> => predictHiAI(session, tokens.ids, tokens.mask));
            result.push(name + ' · P50 ' + timing.p50Ms.toFixed(2) + ' ms · P95 ' + timing.p95Ms.toFixed(2) + ' ms');
          } catch (error) {
            result.push(name + '：' + this.errorMessage(error as Error | string));
          } finally {
            await this.releaseBucket(bucket);
          }
        }
      } else {
        result.push('NPU 不可用，以下仅为 CPU 时延');
      }
      try {
        const timing: TimingResult = await this.measure((): Promise<number[]> => this.predictCpu(tokens));
        result.push('CPU 64 token · P50 ' + timing.p50Ms.toFixed(2) + ' ms · P95 ' + timing.p95Ms.toFixed(2) + ' ms');
      } catch (error) {
        result.push('CPU：' + this.errorMessage(error as Error | string));
      }
      return result.join('\n');
    } finally {
      this.working = false;
    }
  }

  async acceptNpu(progress: (message: string) => void): Promise<AcceptanceReport> {
    this.begin();
    try {
      if (!this.isReady() || this.profile === undefined) {
        throw new Error('模型尚未加载');
      }
      const report: AcceptanceReport = new AcceptanceReport();
      report.modelId = this.modelId;
      report.modelName = this.modelName;
      report.runtimeBackend = this.runtimeBackend;
      report.thresholds = this.thresholds.slice();
      report.maxLogitErrorLimit = this.profile.maxLogitError;
      await this.releaseSessions();
      this.selfTests.clear();
      this.unavailable.clear();
      for (const bucket of BUCKETS) {
        const item: BucketAcceptance = new BucketAcceptance();
        const asset: ModelAsset = this.asset(bucket);
        item.bucket = bucket;
        item.expectedBytes = asset.bytes;
        // This is a manifest identity, not a phone-side SHA-256 measurement.
        item.manifestSha256 = asset.sha256;
        item.expectedCases = asset.referenceCases;
        progress(this.modelName + ' · 正在验收 ' + bucket.toString() + ' token');
        try {
          if (!this.hasHiAI) {
            throw new Error('HIAI_F 不可用；NPU 验收禁止 CPU 回退。' + this.diagnostic);
          }
          const session: HiAISession = await this.loadSession(bucket);
          item.bytesVerified = true;
          item.device = session.device;
          const sample: ReferenceCase | undefined = this.references.find((reference: ReferenceCase): boolean => {
            const tokens: TokenizedInput = this.encode(reference.text, 0);
            return this.tokenCount(tokens) <= bucket;
          });
          if (sample === undefined) {
            throw new Error('此档没有可测指令');
          }
          const tokens: TokenizedInput = this.encode(sample.text, bucket);
          item.timing = await this.measure((): Promise<number[]> => predictHiAI(session, tokens.ids, tokens.mask));
          item.passed = true;
        } catch (error) {
          item.error = this.errorMessage(error as Error | string);
        } finally {
          const checked: ReferenceResult | undefined = this.selfTests.get(bucket);
          if (checked !== undefined) {
            item.checked = checked.checked;
            item.matched = checked.matched;
            item.maxLogitError = checked.maxLogitError;
          }
          await this.releaseBucket(bucket);
        }
        report.buckets.push(item);
        hilog.info(0xFF00, 'PermissionNPU', '[PNPU][ACCEPT] model=%{public}s bucket=%{public}d passed=%{public}s matched=%{public}d/%{public}d error=%{public}s',
          this.modelId, bucket, item.passed.toString(), item.matched, item.checked, item.error);
      }
      report.passed = report.buckets.length === 3 && report.buckets.every((item: BucketAcceptance): boolean => item.passed);
      report.finishedAt = new Date().toISOString();
      progress(this.modelName + (report.passed ? ' · 三档 NPU 转换一致性通过' : ' · 验收未通过，请查看逐档结果'));
      return report;
    } finally {
      this.working = false;
    }
  }

  private validateLogits(logits: number[]): void {
    if (logits.length !== LABEL_COUNT || logits.some((logit: number): boolean => !Number.isFinite(logit))) {
      throw new Error('模型输出应为 5 个有限 logit');
    }
  }

  private result(logits: number[], started: number, backend: string): InferenceResult {
    this.validateLogits(logits);
    const scores: number[] = logits.map((logit: number): number => 1 / (1 + Math.exp(-logit)));
    return new InferenceResult(scores, this.thresholds.slice(), Date.now() - started, this.modelName + ' · ' + backend);
  }
}
