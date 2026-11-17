import { ApiError, type Checked, type Health, type Located, type PuzzleHealth, server } from './api';

/**
 * The app shell every puzzle shares: three steps (Photos, Check, Solve), the photos taken, the side
 * panel's header with the puzzle switcher, toasts, and the server's health. What is puzzle-specific
 * lives in the puzzle's `PuzzleUI` (src/app/<puzzle>/ui.ts): its Check and Solve screens, how its
 * photos are taken and read, and its own state.
 */

/** The three steps of the tool. */
export type Step = 'photos' | 'check' | 'solve';

export const STEPS: readonly { readonly id: Step; readonly label: string }[] = [
  { id: 'photos', label: 'Photos' },
  { id: 'check', label: 'Check' },
  { id: 'solve', label: 'Solve' },
];

/** One photo waiting to be scanned. */
export interface Shot {
  readonly id: number;
  readonly blob: Blob;
  readonly name: string;
  /** Shown at once when the browser can decode the file (not HEIC); the server's preview replaces it. */
  readonly localUrl: string | null;
  status: 'locating' | 'ready' | 'error';
  preview?: string;
  width?: number;
  height?: number;
  quad?: number[][];
  confidence?: number;
  error?: string;
  /** For a webcam photo: what the live check read (to tell when the same face is shown again). */
  readonly signature?: unknown;
}

export interface Screen {
  /** Fill the work area and the panel. */
  enter(app: App): void;
  /** Leaving the screen (stop cameras, timers). */
  leave?(): void;
}

/** How a puzzle's photos are taken (the Photos step is shared; this is what differs). */
export interface PhotoProtocol {
  /** Photos in one scan. */
  readonly count: number;
  /** The "Take the photos" card: a short list (HTML). */
  readonly instructions: string;
  /** One line under the Upload / Webcam switch. */
  readonly hint: string;
  /** The button that sends the photos. */
  readonly readLabel: string;
  /** Which of the located face's corners to mark on each photo, and what the mark means. */
  readonly markCorner: number;
  readonly markTitle: string;
  /** The live check's limit: at most this many unsure places on a face. */
  readonly maxUnsure: number;
  locate(photo: Blob, name: string): Promise<Located>;
  check(frame: Blob): Promise<Checked>;
  /** What to keep of a live check with the photo taken, to recognise the same face later. */
  signature(c: Checked): unknown;
  /** The photo (index into `taken`) showing the same face as this check, or -1. */
  sameAs(c: Checked, taken: readonly unknown[]): number;
  loadVision(): Promise<unknown>;
}

export interface PuzzleUI {
  /** As the server and the URL name it (?puzzle=…). */
  readonly id: string;
  readonly title: string;
  /** A line under the title in the panel. */
  readonly tagline: string;
  readonly photos: PhotoProtocol;
  readonly check: Screen;
  readonly solve: Screen;
  /** Can the app go to Check / Solve now? */
  canGo(step: 'check' | 'solve'): boolean;
  /** Extra cards for the Photos panel (a practice scramble, another way in); `rebuild` redraws the panel. */
  photoCards(app: App, rebuild: () => void): HTMLElement[];
  /** Send the photos, set up the Check step and go there. Throws ApiError with a message for the person. */
  read(app: App, shots: Shot[]): Promise<void>;
  /** Forget the current puzzle state (a new scan, or switching puzzles). */
  reset(): void;
  /** A one-line status of this puzzle's networks and solver, for the bottom of the panel. */
  statusLine(h: PuzzleHealth): string;
}

let nextShotId = 1;
export const newShotId = () => nextShotId++;

const PUZZLE_KEY = 'scanner-puzzle';

export class App {
  step: Step = 'photos';
  shots: Shot[] = [];
  health: Health | null = null;
  puzzle: PuzzleUI;

  readonly work: HTMLElement;
  readonly panel: HTMLElement;
  private readonly stepsEl: HTMLElement;
  private readonly toastEl: HTMLElement;
  private current: Screen | null = null;
  private toastTimer = 0;

  constructor(
    readonly puzzles: readonly PuzzleUI[],
    private readonly photosScreen: Screen,
  ) {
    this.work = document.getElementById('work')!;
    this.panel = document.getElementById('panel')!;
    this.stepsEl = document.getElementById('steps')!;
    this.toastEl = document.getElementById('toast')!;
    this.puzzle = this.initialPuzzle();
    this.stepsEl.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-step]');
      if (b && !b.disabled) this.go(b.dataset.step as Step);
    });
    this.applyTitle();
  }

  /** ?puzzle=… in the URL, else the one used last, else the first. */
  private initialPuzzle(): PuzzleUI {
    const fromUrl = new URLSearchParams(location.search).get('puzzle');
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(PUZZLE_KEY);
    } catch {
      /* private mode */
    }
    return this.puzzles.find((p) => p.id === fromUrl) ?? this.puzzles.find((p) => p.id === saved) ?? this.puzzles[0];
  }

  private applyTitle(): void {
    document.title = `Cube Scanner · ${this.puzzle.title}`;
  }

  /** Switch to another puzzle: the photos and everything read from them are dropped. */
  setPuzzle(id: string): void {
    const next = this.puzzles.find((p) => p.id === id);
    if (!next || next === this.puzzle) return;
    this.current?.leave?.();
    this.current = null;
    this.shots.forEach((s) => s.localUrl && URL.revokeObjectURL(s.localUrl));
    this.shots = [];
    this.puzzle.reset();
    this.puzzle = next;
    try {
      localStorage.setItem(PUZZLE_KEY, id);
    } catch {
      /* private mode */
    }
    const url = new URL(location.href);
    url.searchParams.set('puzzle', id);
    history.replaceState(null, '', url);
    this.applyTitle();
    this.go('photos');
    void this.refreshHealth().then((h) => {
      if (h && !this.puzzleHealth()?.vision.loaded) void this.puzzle.photos.loadVision().catch(() => {});
    });
  }

  canGo(step: Step): boolean {
    return step === 'photos' || this.puzzle.canGo(step);
  }

  go(step: Step): void {
    if (!this.canGo(step)) return;
    this.current?.leave?.();
    this.step = step;
    this.work.replaceChildren();
    this.panel.replaceChildren();
    this.work.dataset.step = step;
    this.work.dataset.puzzle = this.puzzle.id;
    this.panel.scrollTop = 0;
    this.current = step === 'photos' ? this.photosScreen : this.puzzle[step];
    this.current.enter(this);
    this.renderSteps();
  }

  renderSteps(): void {
    this.stepsEl.replaceChildren(
      ...STEPS.map((s, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'steps__tab';
        b.dataset.step = s.id;
        b.disabled = !this.canGo(s.id);
        if (s.id === this.step) b.setAttribute('aria-current', 'step');
        b.innerHTML = `<span class="steps__n">${i + 1}</span><span>${s.label}</span>`;
        return b;
      }),
    );
  }

  toast(message: string, kind: 'info' | 'error' | 'good' = 'info', ms = 5200): void {
    this.toastEl.textContent = message;
    this.toastEl.className = `toast toast--${kind}`;
    this.toastEl.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => (this.toastEl.hidden = true), ms);
  }

  /** The panel's brand header, with the puzzle switcher. */
  header(): HTMLElement {
    const head = document.createElement('header');
    head.className = 'panel__header';
    head.innerHTML = `
      <p class="eyebrow eyebrow--brand">Cube Scanner</p>
      <h1 class="brand__title">${this.puzzle.title}</h1>
      <p class="brand__meta">${this.puzzle.tagline}</p>`;
    if (this.puzzles.length > 1) {
      const seg = h('div', 'seg seg--sm puzzle-switch');
      seg.setAttribute('role', 'tablist');
      seg.setAttribute('aria-label', 'Puzzle');
      for (const p of this.puzzles) {
        const b = button(p.title, p === this.puzzle ? 'is-on' : '', () => {
          if (p === this.puzzle) return;
          const busy = this.shots.length > 0;
          if (busy && !window.confirm(`Switch to the ${p.title}? The photos taken so far are dropped.`)) return;
          this.setPuzzle(p.id);
        });
        b.setAttribute('role', 'tab');
        b.setAttribute('aria-selected', String(p === this.puzzle));
        seg.append(b);
      }
      head.append(seg);
    }
    return head;
  }

  puzzleHealth(): PuzzleHealth | null {
    const h = this.health;
    if (!h) return null;
    return h.puzzles?.[this.puzzle.id] ?? (h.puzzle === this.puzzle.id ? h : null);
  }

  /** A small line about the server at the bottom of the panel. */
  serverLine(): HTMLElement {
    const p = document.createElement('p');
    p.className = 'about server-line';
    const h = this.health;
    const ph = this.puzzleHealth();
    if (!h) p.textContent = 'Connecting to the scanner server…';
    else if (!ph) p.textContent = `Server ${h.version} does not know the ${this.puzzle.title} yet: restart python -m scanner serve.`;
    else p.textContent = `Server ${h.version}: ${this.puzzle.statusLine(ph)}`;
    return p;
  }

  async refreshHealth(): Promise<Health | null> {
    try {
      this.health = await server.health();
    } catch {
      this.health = null;
    }
    document.querySelectorAll('.server-line').forEach((el) => el.replaceWith(this.serverLine()));
    return this.health;
  }

  errorText(err: unknown, fallback = 'Something went wrong'): string {
    return err instanceof ApiError || err instanceof Error ? err.message : `${fallback}: ${String(err)}`;
  }
}

/** The usual status words for a puzzle's networks and solver. */
export function statusWords(h: PuzzleHealth): { vision: string; solver: string } {
  const vision = h.vision.locator && h.vision.reader ? (h.vision.loaded ? `vision ready (${h.vision.device})` : 'vision loading') : 'no trained vision models';
  const solver = { 'not-loaded': 'solver not loaded yet', loading: 'solver loading', ready: 'solver ready', missing: 'solver not connected', error: 'solver failed to load' }[h.solver.state];
  return { vision, solver };
}

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, html?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

export function button(label: string, cls: string, onClick: () => void, title?: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = cls;
  b.innerHTML = label;
  if (title) b.title = title;
  b.addEventListener('click', onClick);
  return b;
}

/** A camera preset button for `viewButtons`. */
export interface ViewOption {
  readonly view: string;
  readonly label: string;
  readonly title?: string;
}

/**
 * A row of camera presets in one segmented control, like the 3D simulator site's View card. The one
 * last chosen is lit; `set(null)` unlights them all (when something else turned the model).
 */
export function viewButtons(options: readonly ViewOption[], onPick: (view: string) => void): { el: HTMLElement; set(view: string | null): void } {
  const el = h('div', 'seg view-seg');
  el.setAttribute('role', 'group');
  el.setAttribute('aria-label', 'Camera');
  const set = (view: string | null) => {
    el.querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
      const on = b.dataset.view === view;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-pressed', String(on));
    });
  };
  for (const o of options) {
    const b = button(
      o.label,
      '',
      () => {
        onPick(o.view);
        set(o.view);
      },
      o.title,
    );
    b.dataset.view = o.view;
    el.append(b);
  }
  set(null);
  return { el, set };
}

export function card(title?: string, cls = 'card'): HTMLElement {
  const c = h('section', cls);
  if (title) c.append(h('p', 'eyebrow', title));
  return c;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

/** Download a text or JSON file. */
export function download(name: string, text: string, type = 'text/plain'): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
