/**
 * What the synthetic-photo page (datagen.html) offers its driver, tools/gen_dataset.py: one puzzle's
 * renderer behind the same small API. Playwright calls `datagen.run` / `datagen.runScans`, and every
 * picture comes back through `__save`; opened by hand the page is a viewer with the labels drawn over
 * each sample.
 */

export interface DatagenPage {
  readonly params: URLSearchParams;
  /** Picture width and height in pixels. */
  readonly size: number;
  readonly canvas: HTMLCanvasElement;
  readonly overlay: HTMLCanvasElement;
  readonly info: HTMLElement;
}

export interface DatagenReady {
  readonly renderer: string;
  readonly size: number;
  readonly puzzle: string;
  /** Pictures in one scan (six faces, or the two turning corners). Scan s's pictures have seeds s * 8 + shot. */
  readonly shotsPerScan: number;
}

export interface DatagenApi<M = unknown> {
  readonly ready: Promise<DatagenReady>;
  one(seed: number, quality?: number): { meta: M; dataUrl: string };
  run(seeds: number[], quality?: number): Promise<number>;
  runScans(scans: number[], quality?: number): Promise<number>;
}

declare global {
  interface Window {
    __save?: (meta: string, jpegBase64: string) => Promise<void>;
    datagen: DatagenApi;
  }
}

/** Send one picture to the driver. */
export async function save(canvas: HTMLCanvasElement, meta: unknown, quality: number): Promise<void> {
  if (!window.__save) throw new Error('saving needs the __save binding (tools/gen_dataset.py provides it)');
  const dataUrl = canvas.toDataURL('image/jpeg', quality);
  await window.__save(JSON.stringify(meta), dataUrl.slice(dataUrl.indexOf(',') + 1));
}
