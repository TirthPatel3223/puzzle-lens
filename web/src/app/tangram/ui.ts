import type { Move } from '../../vendor/tangram/moves';
import type { PuzzleState } from '../../vendor/tangram/state';
import { mulberry32 } from '../../vendor/puzzle/rng';
import { TIMING } from '../../vendor/tangram/config';
import { MODEL } from '../../tangram/model';
import { PuzzleState as State } from '../../vendor/tangram/state';
import type { Checked, PuzzleHealth } from '../api';
import { type App, type PhotoProtocol, type PuzzleUI, type Shot, button, card, escapeHtml, h, statusWords } from '../app';
import { ApiError, ID, type TangramDecode, type TangramScan, type TangramSolution, api } from './api';
import { TangramCheck } from './check';
import { LANDMARKS, PALETTE, linesOf, moveToken, observeState, parseMoves, practiceScramble } from './model';
import { TangramSolve } from './solve';
import { TangramView } from './view3d';

/** A practice round: the puzzle was scrambled from solved with these moves, so its state is known. */
export interface TangramPractice {
  readonly moves: Move[];
  readonly truth: PuzzleState;
}

/** Everything the Tangram Cube Extreme's steps share. */
export class TangramSession {
  practice: TangramPractice | null = null;
  scan: TangramScan | null = null;
  /** The photos the scan was read from (kept to read them again in another colour layout). */
  shots: Shot[] = [];
  /** The puzzle being checked: per position a colour 0-5 or 6 (no sticker), core frame. */
  obs: number[] | null = null;
  /** As first decoded (before any hand edits). */
  original: number[] | null = null;
  /** The legality check of `obs`. */
  decode: TangramDecode | null = null;
  source: 'scan' | 'moves' | null = null;
  solution: { info: TangramSolution; moves: Move[]; forState: string } | null = null;
  kept = false;
  private view3d: TangramView | null = null;

  /** The 3D model, shared by every screen (built on first use). */
  get view(): TangramView {
    this.view3d ??= new TangramView();
    return this.view3d;
  }

  setPuzzle(obs: number[], decode: TangramDecode, source: 'scan' | 'moves'): void {
    // A scan says how its cube is stickered; anything else is shown in the model's own colours.
    PALETTE.set(source === 'scan' ? this.scan?.colours?.paint : null);
    this.obs = obs.slice();
    this.original = obs.slice();
    this.decode = decode;
    this.source = source;
    this.solution = null;
    this.kept = false;
  }

  reset(): void {
    PALETTE.set(null);
    this.practice = null;
    this.scan = null;
    this.shots = [];
    this.obs = null;
    this.original = null;
    this.decode = null;
    this.source = null;
    this.solution = null;
    this.kept = false;
  }
}

/** A face shown again: at least this many of its 29 positions read the same. */
const SAME_FACE = 26;

const photos: PhotoProtocol = {
  count: 6,
  instructions: `
    <ol class="steps-list">
      <li><b>One photo of each face, straight on</b>, whatever shape the puzzle is in. A face is a side of the middle layer, which never changes shape: square the camera to the face's flat stickers.</li>
      <li><b>Pieces that stick out are fine.</b> A piece a 60° turn has tilted out of a face is seen at an angle in two photos; that is how the scanner tells it from the flat one it looks like from a corner. Do not photograph corner-on.</li>
      <li><b>Any order, any way up.</b> Every face has one turning corner (a three-coloured corner piece that only ever twists in place); the scanner finds it to tell which way up a photo is.</li>
      <li><b>Even light</b>, no strong glare, and fingers off the face.</li>
    </ol>`,
  hint: 'Six faces, straight on, any order, any way up, any shape: the scanner works out which face is which.',
  readLabel: 'Read the puzzle',
  markCorner: 2,
  markTitle: "The face's turning corner (the corner piece that only twists in place)",
  maxUnsure: 6,
  locate: (blob, name) => api.locate(blob, name),
  check: (frame) => api.check(frame),
  signature: (c: Checked) => c.reading?.labels,
  sameAs: (c: Checked, taken) => {
    const labels = c.reading?.labels;
    if (!labels) return -1;
    return taken.findIndex((t) => Array.isArray(t) && t.length === labels.length && t.filter((v, i) => v === labels[i]).length >= SAME_FACE);
  },
  loadVision: () => api.loadVision(),
};

export class TangramUI implements PuzzleUI {
  readonly id = ID;
  readonly title = 'Tangram Cube Extreme';
  readonly tagline = 'Six face-on photos to a checked 3D model to a solution, whatever its shape.';
  readonly photos = photos;
  readonly session = new TangramSession();
  readonly check = new TangramCheck(this.session, (app, paint) => this.read(app, this.session.shots, paint));
  readonly solve = new TangramSolve(this.session);

  canGo(step: 'check' | 'solve'): boolean {
    const s = this.session;
    if (step === 'check') return s.obs !== null;
    return !!s.decode?.legal && !!s.decode.orient;
  }

  reset(): void {
    this.session.reset();
  }

  statusLine(hl: PuzzleHealth): string {
    const w = statusWords(hl);
    return `${w.vision}, ${w.solver}. Labelled real scans kept: ${hl.labelledScans}.`;
  }

  async read(app: App, shots: Shot[], colours?: readonly number[]): Promise<void> {
    const scan = await api.scan(
      shots.map((s) => ({ blob: s.blob, name: s.name })),
      shots.map((s) => s.quad ?? null),
      colours,
    );
    const obs = scan.decode.observation;
    if (!obs) throw new ApiError('The photos could not be put together into a puzzle.', 422);
    this.session.scan = scan;
    this.session.shots = shots.slice();
    this.session.setPuzzle(obs, scan.decode, 'scan');
    app.go('check');
  }

  photoCards(app: App, rebuild: () => void): HTMLElement[] {
    return [this.practiceCard(rebuild), this.movesCard(app)];
  }

  private practiceCard(rebuild: () => void): HTMLElement {
    const s = this.session;
    const c = card('Practice with a known scramble');
    const p = s.practice;
    if (!p) {
      c.append(
        h(
          'p',
          'mode-note',
          'Scramble the puzzle with moves the app gives you. The app then knows the exact state, so it can show how well the scan did, and the photos become labelled data for training the model on real pictures.',
        ),
        button('Give me a scramble', 'btn btn--ghost', () => {
          const { moves, truth } = practiceScramble(mulberry32((Math.random() * 2 ** 32) >>> 0), 12);
          s.practice = { moves, truth };
          rebuild();
        }),
      );
      return c;
    }
    c.append(
      h(
        'p',
        'mode-note',
        `Start from a solved puzzle and do these moves (the 3D simulator's notation). <b>U</b> and <b>D</b> turn ${LANDMARKS.U} and ${LANDMARKS.D}, one step is 60° clockwise seen from that corner, ′ is counterclockwise. <b>F</b>, <b>R</b>, <b>L</b> flip ${LANDMARKS.F}, ${LANDMARKS.R} and ${LANDMARKS.L} half a turn.`,
      ),
    );
    const text = h('code', 'scramble__text');
    text.innerHTML = linesOf(p.moves.map(moveToken)).map(escapeHtml).join('<br>');
    c.append(text);
    const box = h('div', 'cube-box cube-box--small');
    c.append(box);
    const view = s.view;
    view.mount(box);
    view.setBadges([]);
    view.showLabels(true);
    view.show(p.truth);
    view.setView('corner', false);
    const row = h('div', 'actions');
    row.append(
      button('Play it on the model', 'btn btn--ghost', () => {
        view.show(State.solved(MODEL));
        for (const m of p.moves) view.play(m, TIMING.moveMs * 1.4);
      }),
      button('Cancel', 'btn btn--ghost', () => {
        s.practice = null;
        rebuild();
      }),
    );
    c.append(row, h('p', 'hint hint--tight', 'When your puzzle matches the model, take the six photos.'));
    return c;
  }

  private movesCard(app: App): HTMLElement {
    const c = card('Or type moves from solved');
    c.append(h('p', 'mode-note', "A state given as the moves that make it from solved, in the 3D simulator's notation (U U2 U3 U2' U' D … F R L), to check it on the 3D model or solve it."));
    const area = document.createElement('textarea');
    area.className = 'array-input';
    area.rows = 2;
    area.placeholder = "U F D2' R U3 L";
    area.spellcheck = false;
    const err = h('p', 'alg__error');
    err.hidden = true;
    const go = button('Check it', 'btn btn--ghost', () => void this.fromMoves(app, area.value, err));
    const row = h('div', 'actions');
    row.append(go);
    c.append(area, row, err);
    return c;
  }

  private async fromMoves(app: App, text: string, err: HTMLElement): Promise<void> {
    err.hidden = true;
    try {
      const moves = parseMoves(text);
      if (!moves.length) throw new Error('Type some moves first, e.g. U F D2′ R.');
      const truth = State.solved(MODEL);
      truth.applyAll(moves);
      const obs = observeState(truth);
      const decode = await api.decode(obs);
      this.session.scan = null;
      this.session.setPuzzle(obs, decode, 'moves');
      app.go('check');
    } catch (e) {
      err.textContent = e instanceof ApiError || e instanceof Error ? e.message : String(e);
      err.hidden = false;
    }
  }
}
