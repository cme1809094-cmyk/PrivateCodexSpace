export interface HiAISession {
  readonly device: string;
}

export const devices: () => string[];
export const accelerator: () => string;
export const load: (model: ArrayBuffer) => Promise<HiAISession>;
export const predict: (session: HiAISession, ids: Int32Array, mask: Int32Array) => Promise<number[]>;
/** Release this handle. Queued predictions retain ownership until completion. */
export const dispose: (session: HiAISession) => Promise<void>;
