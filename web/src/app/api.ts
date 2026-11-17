/**
 * Talking to the local Python server (scanner/server.py). In development Vite proxies /api to it.
 *
 * Every puzzle's endpoints are under api/<puzzle id>/ (scanner/puzzles/<key>/api.py). This module has
 * what all puzzles share: the fetch wrapper, errors, the server's health, and the photo endpoints
 * (locate, live check, scan, confirm) every puzzle offers. Each puzzle's own client (src/app/<puzzle>/api.ts)
 * adds its decode and solve calls and types.
 */

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export const NO_SERVER =
  'The scanner server is not answering. Start it in the Cube_Scanner folder with:  python -m scanner serve  (then reload this page).';

export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`api/${path}`, init);
  } catch {
    throw new ApiError(NO_SERVER, 0);
  }
  if (res.status === 502 || res.status === 504) throw new ApiError(NO_SERVER, res.status);
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* not JSON */
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ')
          : `The server answered ${res.status} ${res.statusText}.`;
    throw new ApiError(message, res.status);
  }
  return body as T;
}

export const json = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export type SolverState = 'not-loaded' | 'loading' | 'ready' | 'missing' | 'error';

/** One puzzle's part of the server's health. */
export interface PuzzleHealth {
  readonly puzzle: string;
  readonly solver: { readonly state: SolverState; readonly detail: string };
  readonly vision: { readonly locator: boolean; readonly reader: boolean; readonly loaded: boolean; readonly device: string | null; readonly error: string | null };
  readonly labelledScans: number;
}

export interface Health extends PuzzleHealth {
  readonly ok: boolean;
  readonly version: string;
  /** Every puzzle's status, by id. */
  readonly puzzles?: Readonly<Record<string, PuzzleHealth>>;
}

export interface Located {
  readonly width: number;
  readonly height: number;
  /** The face's four corners as fractions of the picture (the puzzle's corner order). */
  readonly quad: number[][];
  readonly confidence: number;
  readonly preview?: string;
}

/** A live webcam frame checked by the networks (POST …/vision/check). */
export interface Checked extends Located {
  /** Null when no face was found. */
  readonly reading: {
    /** Places on the face the reader was unsure of. */
    readonly unsure: number;
    readonly minProb: number;
    /** Maltese: the 16 cells as read, put back k quarter turns (k = 0-3). */
    readonly cellTurns?: number[][];
    /** Tangram: what each of the face's 29 positions shows (the face's own order, so no turns are needed). */
    readonly labels?: number[];
  } | null;
}

export interface Confirmed {
  readonly saved: string;
  readonly scans: number;
  readonly readErrors: number | Record<string, number> | null;
  readonly decodedRight: boolean | null;
}

/** The calls every puzzle offers, under api/<id>/. */
export function puzzleApi(id: string) {
  const p = (path: string) => `${id}/${path}`;
  return {
    health: () => call<PuzzleHealth & { ok: boolean }>(p('health')),
    loadSolver: () => call<PuzzleHealth['solver']>(p('solver/load'), { method: 'POST' }),
    loadVision: () => call<{ started: boolean }>(p('vision/load'), { method: 'POST' }),
    locate(photo: Blob, name: string): Promise<Located> {
      const form = new FormData();
      form.append('photo', photo, name);
      return call<Located>(p('vision/locate?preview=true'), { method: 'POST', body: form });
    },
    check(frame: Blob): Promise<Checked> {
      const form = new FormData();
      form.append('photo', frame, 'frame.jpg');
      // A live check is worthless late: give up on one that hangs (the page simply sends the next frame).
      return call<Checked>(p('vision/check'), { method: 'POST', body: form, signal: AbortSignal.timeout(3000) });
    },
    /** Six photos to a decoded puzzle. `colours`: the cube's colour layout when the person chose one (palette.ts). */
    scan<T>(photos: { blob: Blob; name: string }[], quads?: (number[][] | null)[], colours?: readonly number[]): Promise<T> {
      const form = new FormData();
      for (const ph of photos) form.append('photos', ph.blob, ph.name);
      if (quads) form.append('quads', JSON.stringify(quads));
      if (colours) form.append('colours', JSON.stringify(colours));
      return call<T>(p('vision/scan'), { method: 'POST', body: form });
    },
    confirm: (scanId: string, body: unknown) => call<Confirmed>(p(`scans/${encodeURIComponent(scanId)}/confirm`), json(body)),
  };
}

export const server = {
  health: () => call<Health>('health'),
  puzzles: () => call<{ default: string; puzzles: { id: string; key: string; title: string; photos: number }[] }>('puzzles'),
};
