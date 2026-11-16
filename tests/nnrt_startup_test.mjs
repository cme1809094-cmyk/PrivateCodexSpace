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
const reported = 'NPU_ohos.boot.hardware.Kirin9020_v2_0 (ID 42, type ACCELERATOR)';
const native = {
  devices: () => state.devices,
  accelerator: () => {
    state.selected++;
    if (state.error) throw new Error(state.error);
    return state.backend;
  },
  load: () => { throw new Error('Startup must not load an OMC before its bucket is needed'); },
  predict: () => { throw new Error('Unexpected inference during startup'); }
};
globalThis.__permissionTest = {
  native,
  util: { TextDecoder: { create: () => ({ decodeToString: b => new TextDecoder().decode(b) }) } },
  mindSporeLite: {
    DataType: { NUMBER_TYPE_INT32: 4 },
    loadModelFromBuffer: async () => {
      state.cpuLoads++;
      return { getInputs: () => ['input_ids', 'attention_mask'].map(name => ({ name, shape: [1, 64], dtype: 4 })) };
    }
  }
};
const resources = {
  getRawFileContent: async name => new TextEncoder().encode(name === 'vocab.txt'
    ? '[PAD]\n[UNK]\n[CLS]\n[SEP]\n'
    : name === 'model_config.json' ? '{"thresholds":[0.5,0.5,0.5,0.5,0.5]}' : 'mock CPU model')
};

try {
  fs.copyFileSync(path.join(source, 'WordPiece.ets'), path.join(temporary, 'WordPiece.mts'));
  let code = fs.readFileSync(path.join(source, 'PermissionModel.ets'), 'utf8');
  code = code.replace(/^import .*from '@kit\.[^']+';\r?\n/gm, '')
    .replace(/^import .*from 'libpermission_hiai.so';\r?\n/gm, '')
    .replace("from './WordPiece'", "from './WordPiece.mts'");
  code = 'const {native,util,mindSporeLite}=globalThis.__permissionTest;\n' +
    'const {devices,accelerator,load:loadHiAI,predict:predictHiAI}=native;\n' +
    'const canIUse=()=>true;\n' + code;
  fs.writeFileSync(path.join(temporary, 'PermissionModel.mts'), code);
  const { PermissionModel } = await import(pathToFileURL(path.join(temporary, 'PermissionModel.mts')).href);
  for (const backend of [reported, 'HIAI_F (ID 9, type ACCELERATOR)', 'Future vendor (ID 19, type ACCELERATOR)']) {
    state = { devices: [backend], backend, selected: 0, cpuLoads: 0 };
    const model = new PermissionModel();
    await model.load(resources);
    assert.equal(state.selected, 1);
    assert.equal(state.cpuLoads, 0);
    assert.ok(model.diagnostic.includes(backend));
    assert.ok(!model.diagnostic.includes('回退原因'));
  }
  for (const error of ['No NNRT accelerator is available; enumerated: CPU', 'OH_NNDevice_GetType(42) failed (1)']) {
    state = { devices: ['CPU (ID 4, type CPU)'], error, selected: 0, cpuLoads: 0 };
    const model = new PermissionModel();
    await model.load(resources);
    assert.equal(state.cpuLoads, 1);
    assert.ok(model.diagnostic.includes(error));
  }
  console.log('PASS: ArkTS startup accepts reported/legacy/future accelerator names and preserves CPU fallback/query diagnostics');
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
  delete globalThis.__permissionTest;
}
