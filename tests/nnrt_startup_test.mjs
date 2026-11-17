// Run with Node 22+ and --experimental-strip-types. This exercises the actual
// ArkTS model loader with mocked platform bindings; it does not execute an NPU.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = fs.mkdtempSync(path.join(process.env.NNRT_TEST_WORK || os.tmpdir(), 'nnrt-startup-'));
const source = path.join(root, 'entry/src/main/ets/model');
let state;
const logs = [];
const hilog = Object.fromEntries(['info', 'warn'].map(level => [level, (...args) => logs.push({level, args})]));
const reported = 'NPU_ohos.boot.hardware.Kirin9020_v2_0 (ID 42, type ACCELERATOR)';
const native = {
  devices: () => state.devices,
  accelerator: () => {
    state.selected++;
    if (state.error) throw new Error(state.error);
    return state.backend;
  },
  load: async () => {
    if (!state.allowLoad) throw new Error('Startup must not load an OMC before its bucket is needed');
    state.nativeLoads++;
    throw state.loadError;
  },
  predict: () => { throw new Error('Unexpected inference during startup'); },
  dispose: async () => {}
};
globalThis.__permissionTest = {
  native,
  hilog,
  util: { TextDecoder: { create: () => ({ decodeToString: b => new TextDecoder().decode(b) }) } },
  mindSporeLite: {
    DataType: { NUMBER_TYPE_INT32: 4, NUMBER_TYPE_FLOAT32: 11 },
    loadModelFromBuffer: async () => {
      state.cpuLoads++;
      return {
        getInputs: () => ['input_ids', 'attention_mask'].map(name => ({ name, shape: [1, 64], dtype: 4, setData: () => {} })),
        predict: async () => [{ dtype: 11, elementNum: 5, getData: () => new Float32Array(5).buffer }]
      };
    }
  }
};
const resources = {
  getRawFileContent: async name => new TextEncoder().encode(name === 'vocab.txt'
    ? '[PAD]\n[UNK]\n[CLS]\n[SEP]\n'
    : name === 'model_catalog.json' ? JSON.stringify({defaultModel: 'minirbt', profiles: [{id: 'minirbt', name: 'MiniRBT', assetPrefix: '', maxLogitError: 0.1,
        assets: [16,32,64].map(bucket => ({bucket, bytes: 14, sha256: 'a'.repeat(64), referenceCases: 1}))}]})
    : name === 'hiai_reference.json' ? '[{"text":"测试","logits":[0,0,0,0,0]}]'
    : name === 'model_config.json' ? '{"thresholds":[0.5,0.5,0.5,0.5,0.5]}' : 'mock CPU model')
};

try {
  fs.copyFileSync(path.join(source, 'WordPiece.ets'), path.join(temporary, 'WordPiece.mts'));
  let code = fs.readFileSync(path.join(source, 'PermissionModel.ets'), 'utf8');
  code = code.replace(/^import .*from '@kit\.[^']+';\r?\n/gm, '')
    .replace(/^import .*from 'libpermission_hiai.so';\r?\n/gm, '')
    .replace("from './WordPiece'", "from './WordPiece.mts'");
  code = 'const {native,util,mindSporeLite,hilog}=globalThis.__permissionTest;\n' +
    'const {devices,accelerator,load:loadHiAI,predict:predictHiAI,dispose:disposeHiAI}=native;\n' +
    'const canIUse=()=>true;\n' + code;
  fs.writeFileSync(path.join(temporary, 'PermissionModel.mts'), code);
  const { PermissionModel } = await import(pathToFileURL(path.join(temporary, 'PermissionModel.mts')).href);
  for (const backend of ['HIAI_F (ID 9, type ACCELERATOR) · CANN test-version']) {
    state = { devices: [reported, backend], backend, selected: 0, cpuLoads: 0 };
    const model = new PermissionModel();
    await model.load(resources);
    assert.equal(state.selected, 1);
    assert.equal(state.cpuLoads, 0);
    assert.ok(model.diagnostic.includes(backend));
    assert.ok(!model.diagnostic.includes('回退原因'));
  }
  for (const error of ['CANN offline-model backend HIAI_F is unavailable; generic NNRT accelerators cannot load OMC',
                      'Cannot load CANN Foundation', 'OH_NNDevice_GetType(42) failed (1)']) {
    state = { devices: [reported], error, selected: 0, cpuLoads: 0 };
    const model = new PermissionModel();
    await model.load(resources);
    assert.equal(state.cpuLoads, 1);
    assert.ok(model.diagnostic.includes(error));
  }
  const loadError = 'Load OMC failed at OH_NNCompilation_Build (not returned); exception_type=i; CANN=test; bridge=1.0.9; detail=Native exception int=17';
  state = { devices: ['HIAI_F'], backend: 'HIAI_F', allowLoad: true, loadError, nativeLoads: 0, selected: 0, cpuLoads: 0 };
  const model = new PermissionModel();
  await model.load(resources);
  const result = await model.predict('测试');
  assert.equal(state.nativeLoads, 3);
  assert.equal(state.cpuLoads, 1);
  assert.ok(result.backend.includes('CPU'));
  assert.ok(model.diagnostic.includes(loadError));
  assert.ok(logs.every(({args}) => args[1] === 'PermissionNPU' && args[2].startsWith('[PNPU]')));
  assert.equal(logs.filter(({args}) => args[2].startsWith('[PNPU][FALLBACK]') && args[4].includes(loadError)).length, 3);
  console.log('PASS: ArkTS startup selects CANN for OMC and preserves CPU fallback for generic NNRT/missing Foundation/query errors');
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
  delete globalThis.__permissionTest;
}
