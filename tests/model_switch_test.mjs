// Host simulation of the actual ArkTS router with real paired vocabularies and
// reference vectors. Native NPU and MindSpore Lite calls are explicitly mocked.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const raw = path.join(root, 'entry/src/main/resources/rawfile');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'permission-switch-'));
let state;
let clock = 0;
const logs = [];
const outcomes = [];
const productionCatalog = JSON.parse(fs.readFileSync(path.join(raw, 'model_catalog.json')));
const catalog = structuredClone(productionCatalog);
for (const p of catalog.profiles) for (const a of p.assets) a.bytes = 8;
const metadata = new Map();
const vectors = new Map();
const key = (ids, mask) => Array.from(ids).join(',') + ':' + Array.from(mask).join(',');
const codeFor = id => id === 'minirbt' ? 1 : 2;
const profileFor = value => catalog.profiles.find(p => codeFor(p.id) === value);

const native = {
  devices: () => ['Kirin9020', 'HIAI_F'],
  accelerator: () => {
    if (state.noNpu) throw new Error('HIAI_F unavailable');
    return 'HIAI_F · CANN mock';
  },
  load: async buffer => {
    const bytes = new Uint8Array(buffer);
    const profile = profileFor(bytes[0]);
    const bucket = bytes[1];
    const name = profile.id + ':' + bucket;
    state.loads.push(name);
    if (state.loadFailure.has(name)) throw new Error('Injected Build failure ' + name);
    const session = { device: 'HIAI_F · CANN mock', profile: profile.id, bucket, disposed: false };
    state.live.add(session);
    state.maxLive = Math.max(state.maxLive, state.live.size);
    return session;
  },
  predict: async (session, ids, mask) => {
    assert.ok(!session.disposed, 'Never predict through a disposed session');
    if (state.gate) await state.gate;
    const name = session.profile + ':' + session.bucket;
    const calls = (state.calls.get(name) || 0) + 1;
    state.calls.set(name, calls);
    if (state.failRun === name && calls === 40) throw new Error('Injected RunSync failure during timing');
    let logits = vectors.get(session.profile).get(key(ids, mask));
    if (!logits) logits = [2, -2, 2, -2, -2];
    logits = [...logits];
    if (session.bucket === 16 && state.corrupt === 'nan') logits[0] = NaN;
    if (session.bucket === 16 && state.corrupt === 'short') logits.pop();
    if (session.bucket === 16 && state.corrupt === 'labels') logits = logits.map(v => -v);
    if (session.bucket === 16 && state.corrupt === 'drift') {
      const i = logits.findIndex(v => Math.abs(v) > 1);
      assert.ok(i >= 0);
      logits[i] += Math.sign(logits[i]) * 0.2;
    }
    return logits;
  },
  dispose: async session => {
    assert.ok(!session.disposed, 'Release a native handle at most once');
    session.disposed = true;
    state.live.delete(session);
    state.disposed.push(session.profile + ':' + session.bucket);
    if (state.releaseFailure && session.bucket === 16) throw new Error('Injected disposal completion failure');
  }
};
globalThis.__permissionTest = {
  native,
  hilog: Object.fromEntries(['info', 'warn'].map(level => [level, (...args) => logs.push({ level, args })])),
  util: { TextDecoder: { create: () => ({ decodeToString: b => new TextDecoder().decode(b) }) } },
  systemDateTime: { TimeType: { ACTIVE: 1 }, getUptime: () => (clock += 1000000) },
  mindSporeLite: {
    DataType: { NUMBER_TYPE_INT32: 4, NUMBER_TYPE_FLOAT32: 11 },
    loadModelFromBuffer: async buffer => {
      const profile = profileFor(new Uint8Array(buffer)[0]);
      state.cpuLoads.push(profile.id);
      return {
        getInputs: () => ['input_ids', 'attention_mask'].map(name => ({name, shape: [1,64], dtype: 4, setData: () => {}})),
        predict: async () => [{dtype: 11, elementNum: 5, getData: () => new Float32Array([2,-2,2,-2,-2]).buffer}]
      };
    }
  }
};
const reset = () => {
  state = {loads: [], disposed: [], cpuLoads: [], calls: new Map(), live: new Set(), maxLive: 0, loadFailure: new Set()};
  logs.length = 0;
};
const resources = {
  getRawFileContent: async name => {
    state.reads ||= [];
    state.reads.push(name);
    if (state.missing === name) throw new Error('Missing rawfile ' + name);
    if (name === 'model_catalog.json') return new TextEncoder().encode(JSON.stringify(state.catalog || catalog));
    if (metadata.has(name)) return metadata.get(name);
    const p = catalog.profiles.find(p => name.startsWith(p.assetPrefix) &&
      (p.assetPrefix.length > 0 || !name.startsWith('models/')));
    assert.ok(p, 'Known profile asset ' + name);
    assert.match(name, /permission(?:_(?:16|32|64|baseline))?\.(om|ms)$/);
    const bucket = name.endsWith('.ms') ? 64 : name.includes('baseline') ? 0 : Number(name.match(/_(\d+)\.om$/)[1]);
    const bytes = new Uint8Array(state.truncate === name ? 7 : 8);
    bytes[0] = codeFor(p.id);
    bytes[1] = bucket;
    return bytes;
  }
};

try {
  const source = path.join(root, 'entry/src/main/ets/model');
  fs.copyFileSync(path.join(source, 'WordPiece.ets'), path.join(temporary, 'WordPiece.mts'));
  let code = fs.readFileSync(path.join(source, 'PermissionModel.ets'), 'utf8');
  code = code.replace(/^import .*from '@kit\.[^']+';\r?\n/gm, '')
    .replace(/^import .*from 'libpermission_hiai.so';\r?\n/gm, '')
    .replace("from './WordPiece'", "from './WordPiece.mts'");
  code = 'const {native,util,mindSporeLite,hilog,systemDateTime}=globalThis.__permissionTest;\n' +
    'const {devices,accelerator,load:loadHiAI,predict:predictHiAI,dispose:disposeHiAI}=native;\n' +
    'const canIUse=()=>true;\n' + code;
  fs.writeFileSync(path.join(temporary, 'PermissionModel.mts'), code);
  const { PermissionModel } = await import(pathToFileURL(path.join(temporary, 'PermissionModel.mts')));
  const { WordPiece } = await import(pathToFileURL(path.join(temporary, 'WordPiece.mts')));
  for (const p of productionCatalog.profiles) {
    for (const file of ['vocab.txt', 'model_config.json', 'hiai_reference.json']) {
      metadata.set(p.assetPrefix + file, fs.readFileSync(path.join(raw, p.assetPrefix, file)));
    }
    const tokenizer = new WordPiece(metadata.get(p.assetPrefix + 'vocab.txt').toString());
    const vectorFile = p.id === 'minirbt' ? 'ml/test_vectors.json' : 'variants/tinybert4l/test_vectors.json';
    for (const vector of JSON.parse(fs.readFileSync(path.join(root, vectorFile)))) {
      const actual = tokenizer.encode(vector.text);
      assert.deepEqual(Array.from(actual.ids), vector.input_ids, p.id + ': paired input_ids');
      assert.deepEqual(Array.from(actual.mask), vector.attention_mask, p.id + ': paired attention_mask');
    }
    const refs = JSON.parse(metadata.get(p.assetPrefix + 'hiai_reference.json'));
    vectors.set(p.id, new Map(refs.map(r => { const t = tokenizer.encode(r.text); return [key(t.ids,t.mask),r.logits]; })));
    for (const asset of p.assets) {
      const checked = refs.filter(r => {
        const t = tokenizer.encode(r.text);
        return !t.tooLong && t.mask.reduce((a,b) => a+b,0) <= (asset.bucket || 64);
      }).length;
      assert.equal(checked, asset.referenceCases, p.id + ':' + asset.bucket + ' complete reference count');
    }
  }

  async function scenario(name, body) {
    reset();
    await body();
    outcomes.push({name, passed: true});
    console.log('PASS:', name);
  }
  const start = async () => { const m = new PermissionModel(); await m.load(resources); return m; };

  await scenario('TinyBERT default is lazy, with its own thresholds and vocabulary', async () => {
    const m = await start();
    assert.equal(m.modelId, 'tinybert4l');
    assert.equal(state.loads.length, 0);
    assert.equal(state.cpuLoads.length, 0);
    const r = await m.predict('录一段视频，别录声音', 16);
    assert.deepEqual(r.thresholds, JSON.parse(metadata.get('models/tinybert4l/model_config.json')).thresholds);
    assert.ok(r.backend.includes('TinyBERT') && r.backend.includes('16 token'));
    assert.deepEqual(state.loads, ['tinybert4l:16']);
    assert.ok(state.reads.includes('models/tinybert4l/vocab.txt'));
  });
  await scenario('Every manual bucket is isolated and switching models releases old NPU handles', async () => {
    const m = await start();
    for (const id of ['tinybert4l', 'minirbt', 'tinybert4l']) {
      await m.switchModel(id);
      for (const bucket of [16,32,64]) {
        const r = await m.predict('拍照', bucket);
        assert.ok(r.backend.includes(bucket + ' token'));
        const p = catalog.profiles.find(p => p.id === id);
        assert.deepEqual(r.thresholds, JSON.parse(metadata.get(p.assetPrefix + 'model_config.json')).thresholds);
      }
    }
    assert.equal(state.loads.length, 9);
    assert.equal(state.disposed.length, 8);
    assert.equal(state.maxLive, 1);
    assert.ok(!state.reads.includes('permission_baseline.om'));
  });
  await scenario('Automatic routing chooses the smallest fitting bucket; manual overflow is rejected', async () => {
    const m = await start();
    const text = '测'.repeat(20);
    await assert.rejects(m.predict(text,16), /超过 16/);
    assert.equal(state.loads.length, 0);
    assert.ok((await m.predict(text)).backend.includes('32 token'));
    await assert.rejects(m.predict('测'.repeat(80)), /指令过长/);
    await assert.rejects(m.predict('  '), /有效指令/);
    await assert.rejects(m.predict('拍照',17), /档位无效/);
  });
  await scenario('Failed forced bucket uses same-model CPU, without silently trying a larger NPU bucket', async () => {
    const m = await start();
    state.loadFailure.add('tinybert4l:16');
    const r = await m.predict('拍照',16);
    assert.ok(r.backend.includes('TinyBERT') && r.backend.includes('CPU'));
    assert.deepEqual(state.loads, ['tinybert4l:16']);
    assert.deepEqual(state.cpuLoads, ['tinybert4l']);
    await m.switchModel('minirbt');
    state.loadFailure.add('minirbt:16');
    await m.predict('拍照',16);
    assert.deepEqual(state.cpuLoads, ['tinybert4l','minirbt']);
  });
  await scenario('Automatic route can recover from a failed small NPU bucket', async () => {
    const m = await start();
    state.loadFailure.add('tinybert4l:16');
    assert.ok((await m.predict('拍照')).backend.includes('32 token'));
    assert.deepEqual(state.loads, ['tinybert4l:16','tinybert4l:32']);
    assert.equal(state.cpuLoads.length, 0);
  });
  await scenario('Switch metadata failure preserves the previous working model', async () => {
    const m = await start();
    state.missing = 'vocab.txt';
    await assert.rejects(m.switchModel('minirbt'), /Missing rawfile/);
    assert.equal(m.modelId, 'tinybert4l');
    assert.ok(m.isReady());
    assert.ok((await m.predict('拍照',16)).backend.includes('TinyBERT'));
  });
  await scenario('NPU acceptance covers 27/33/33 actual paired vectors, fresh sessions, and no CPU calls', async () => {
    const m = await start();
    await m.predict('拍照',16);
    const progress = [];
    const report = await m.acceptNpu(message => progress.push(message));
    assert.equal(report.passed, true);
    assert.equal(report.appVersion, '1.1.1');
    assert.equal(report.modelId, 'tinybert4l');
    assert.ok(report.runtimeBackend.includes('CANN mock'));
    assert.equal(report.cpuFallbackAllowed, false);
    assert.deepEqual(report.buckets.map(b => b.checked), [27,33,33]);
    for (const b of report.buckets) {
      assert.equal(b.checked, b.matched);
      assert.ok(b.bytesVerified && b.passed && b.device.includes('HIAI_F'));
      assert.equal(b.timing.warmupRuns, 5);
      assert.equal(b.timing.sampleRuns, 50);
      assert.ok(Number.isFinite(b.timing.p95Ms));
      assert.equal(b.manifestSha256, productionCatalog.profiles[1].assets.find(a => a.bucket === b.bucket).sha256);
    }
    assert.equal(state.cpuLoads.length, 0);
    assert.equal(state.live.size, 0);
    assert.equal(state.maxLive, 1);
    assert.equal(progress.length, 4);
    assert.equal(state.loads.filter(n => n === 'tinybert4l:16').length, 2);
    await m.switchModel('minirbt');
    const mini = await m.acceptNpu(() => {});
    assert.equal(mini.passed, true);
    assert.equal(mini.modelId, 'minirbt');
  });
  for (const corrupt of ['nan','short','labels','drift']) {
    await scenario('Acceptance rejects ' + corrupt + ' outputs and still checks the other buckets', async () => {
      const m = await start();
      state.corrupt = corrupt;
      const report = await m.acceptNpu(() => {});
      assert.equal(report.passed, false);
      assert.equal(report.buckets[0].passed, false);
      assert.ok(report.buckets[0].error);
      assert.equal(report.buckets[1].passed, true);
      assert.equal(report.buckets[2].passed, true);
      assert.equal(state.cpuLoads.length, 0);
      assert.equal(state.live.size, 0);
      if (corrupt === 'drift') {
        assert.equal(report.buckets[0].matched, 27);
        assert.ok(report.buckets[0].maxLogitError > report.maxLogitErrorLimit);
      }
    });
  }
  await scenario('A RunSync failure during timed runs cannot yield acceptance pass', async () => {
    const m = await start();
    state.failRun = 'tinybert4l:16';
    const report = await m.acceptNpu(() => {});
    assert.equal(report.passed, false);
    assert.match(report.buckets[0].error, /RunSync failure during timing/);
    assert.equal(state.live.size, 0);
  });
  await scenario('Cleanup failure preserves the original inference validation error', async () => {
    const m = await start();
    state.corrupt = 'nan';
    state.releaseFailure = true;
    const report = await m.acceptNpu(() => {});
    assert.equal(report.passed, false);
    assert.match(report.buckets[0].error, /5 个有限 logit/);
    assert.ok(!report.buckets[0].error.includes('disposal'));
    assert.ok(logs.some(({args}) => args[2].startsWith('[PNPU][CLEANUP]')));
    assert.equal(report.buckets[1].passed, true);
    assert.equal(report.buckets[2].passed, true);
  });
  await scenario('Missing NPU cannot pass acceptance through CPU fallback', async () => {
    state.noNpu = true;
    const m = await start();
    assert.deepEqual(state.cpuLoads, ['tinybert4l']); // Normal app startup may fall back.
    const before = state.cpuLoads.length;
    const report = await m.acceptNpu(() => {});
    assert.equal(report.passed, false);
    assert.equal(report.buckets.length, 3);
    assert.ok(report.buckets.every(b => !b.passed && b.error.includes('禁止 CPU')));
    assert.equal(state.cpuLoads.length, before);
    assert.equal(state.loads.length, 0);
  });
  await scenario('Truncated or LFS-pointer model asset fails before native load', async () => {
    const m = await start();
    state.truncate = 'models/tinybert4l/permission_16.om';
    const report = await m.acceptNpu(() => {});
    assert.equal(report.buckets[0].passed, false);
    assert.match(report.buckets[0].error, /字节数不符/);
    assert.ok(!state.loads.includes('tinybert4l:16'));
    assert.equal(state.cpuLoads.length, 0);
  });
  await scenario('Incomplete reference coverage cannot pass even when all remaining labels match', async () => {
    state.catalog = structuredClone(catalog);
    state.catalog.profiles[1].assets[0].referenceCases++;
    const m = await start();
    const report = await m.acceptNpu(() => {});
    assert.equal(report.buckets[0].matched, 27);
    assert.equal(report.buckets[0].passed, false);
  });
  await scenario('Invalid numeric tolerance is rejected at catalog load', async () => {
    state.catalog = structuredClone(catalog);
    state.catalog.profiles[1].maxLogitError = 0;
    await assert.rejects(start(), /模型配置无效/);
  });
  await scenario('Concurrent switch, acceptance and benchmark cannot race an active inference', async () => {
    const m = await start();
    let release;
    state.gate = new Promise(resolve => { release = resolve; });
    const pending = m.predict('拍照',16);
    await assert.rejects(m.switchModel('minirbt'), /正在工作/);
    await assert.rejects(m.acceptNpu(() => {}), /正在工作/);
    await assert.rejects(m.benchmark('拍照'), /正在工作/);
    release();
    await pending;
    state.gate = undefined;
    await m.switchModel('minirbt');
    assert.equal(m.modelId, 'minirbt');
  });
  await scenario('Single-profile generated catalog needs no MiniRBT or baseline assets', async () => {
    state.catalog = structuredClone(catalog);
    state.catalog.profiles = [state.catalog.profiles[1]];
    const m = await start();
    const report = await m.acceptNpu(() => {});
    assert.equal(report.passed, true);
    await assert.rejects(m.switchModel('minirbt'), /不存在/);
    assert.ok(state.reads.every(p => p === 'model_catalog.json' || p.startsWith('models/tinybert4l/')));
  });

  assert.ok(logs.every(({args}) => args[1] === 'PermissionNPU' && args[2].startsWith('[PNPU]')));
  const output = {kind: 'host simulation; NPU/CPU bindings mocked, real paired vocabularies and reference vectors',
    device_tested: false, scenarios: outcomes.length, passed: true, outcomes};
  const reportIndex = process.argv.indexOf('--report');
  if (reportIndex >= 0) fs.writeFileSync(process.argv[reportIndex+1], JSON.stringify(output,null,2) + '\n');
  console.log('PASS: ' + outcomes.length + ' model switching / acceptance host scenarios; no real NPU execution');
} finally {
  fs.rmSync(temporary, {recursive: true, force: true});
  delete globalThis.__permissionTest;
}
