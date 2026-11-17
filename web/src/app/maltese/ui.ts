import solverMapJson from '../../../../shared/maltese/solver_map.json';
import { TIMING } from '../../vendor/maltese/config';
import { FACES } from '../../vendor/maltese/geometry';
import { MalteseState } from '../../vendor/maltese/state';
import { mulberry32 } from '../../vendor/puzzle/rng';
import { type SolverMap, fromSolver144 } from '../../maltese/solverMap';
import type { Checked, PuzzleHealth } from '../api';
import { type App, type PhotoProtocol, type PuzzleUI, type Shot, button, card, h, statusWords } from '../app';
import { sameFaceAs } from '../liveCheck';
import { ApiError, type Decode, ID, type Scan, type Solution, api } from './api';
import { CheckScreen } from './check';
import { CubeView } from './cube3d';
import { COLOUR_NAMES, HOLD_POSITION, PALETTE, type PlainObservation, SOLVER_VIEW, type SolverMove, linesOfFour, solverScramble, toPlain } from './model';
import { SolveScreen } from './solve';

const SOLVER_MAP = solverMapJson as unknown as SolverMap;

/** A practice round: the cube was scrambled from solved with these moves, so its state is known. */
export interface Practice {
  readonly moves: SolverMove[];
  /** The scrambled cube in the solver's frame. */
  readonly truth: MalteseState;
  /** You said the real cube now matches. */
  applied: boolean;
}

/** How each face of the cube being checked is shown (photos, or a drawing when there are none). */
export interface FaceInfo {
  readonly title: string;
  readonly fileName?: string;
  readonly photo?: string;
  readonly chip?: { text: string; colour: number };
}

/** Everything the Maltese Gear Cube's steps share. */
export class MalteseSession {
  practice: Practice | null = null;
  /** The last scan, as the server answered it. */
  scan: Scan | null = null;
  /** The photos it was read from (kept to read them again in another colour layout). */
  shots: Shot[] = [];
  /** The cube being checked, in the frame it was photographed in (or the solver's frame for a pasted array). */
  obs: PlainObservation | null = null;
  /** The same as first decoded (before any hand edits). */
  original: PlainObservation | null = null;
  /** Legality check of `obs` (decode of the hand-edited state). */
  decode: Decode | null = null;
  /** Per face of `obs` (FACES order): how to show it. */
  faces: FaceInfo[] = [];
  /** Where `obs` came from. */
  source: 'scan' | 'array' | null = null;
  /** The solution for the current cube. */
  solution: { info: Solution; moves: SolverMove[]; start: MalteseState; forState: string } | null = null;
  /** The scan was kept as labelled real data. */
  kept = false;
  private view: CubeView | null = null;

  /** The 3D model, shared by every screen (built on first use: about a second of mesh booleans). */
  get cube(): CubeView {
    this.view ??= new CubeView();
    return this.view;
  }

  /** Start a new cube to check (from a scan or a pasted array). */
  setCube(obs: PlainObservation, decode: Decode, faces: FaceInfo[], source: 'scan' | 'array'): void {
    // A scan says how its cube is stickered; a pasted array is shown in the model's own colours.
    PALETTE.set(source === 'scan' ? this.scan?.colours?.paint : null);
    this.obs = obs;
    this.original = { cells: obs.cells.slice(), gearColour: obs.gearColour.slice(), gearQuarter: obs.gearQuarter.slice() };
    this.decode = decode;
    this.faces = faces;
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
    this.faces = [];
    this.source = null;
    this.solution = null;
    this.kept = false;
  }
}

const photos: PhotoProtocol = {
  count: 6,
  instructions: `
    <ol class="steps-list">
      <li><b>One photo of each face</b>, square-on, the face filling most of the picture with its edges in view.</li>
      <li><b>Any order, any way up.</b> The cube can rest on a table or be held; keep fingers off the face's edges.</li>
      <li><b>Even light.</b> Avoid strong glare on the stickers.</li>
    </ol>`,
  hint: 'Any order, any way up: the scanner works out which face is which.',
  readLabel: 'Read the cube',
  markCorner: 0,
  markTitle: "The face's top-left corner as the photo shows it",
  maxUnsure: 4,
  locate: (blob, name) => api.locate(blob, name),
  check: (frame) => api.check(frame),
  signature: (c: Checked) => c.reading?.cellTurns?.[0],
  sameAs: (c: Checked, taken) => sameFaceAs(c.reading?.cellTurns ?? [], taken),
  loadVision: () => api.loadVision(),
};

export class MalteseUI implements PuzzleUI {
  readonly id = ID;
  readonly title = 'Maltese Gear Cube';
  readonly tagline = 'Photos to a checked 3D model to a solution.';
  readonly photos = photos;
  readonly session = new MalteseSession();
  readonly check = new CheckScreen(this.session, (app, paint) => this.read(app, this.session.shots, paint));
  readonly solve = new SolveScreen(this.session);

  canGo(step: 'check' | 'solve'): boolean {
    const s = this.session;
    if (step === 'check') return s.obs !== null;
    return !!s.decode?.legal && !!s.decode.state144;
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
    if (!scan.observation) throw new ApiError('The photos could not be put together into a cube.', 422);
    PALETTE.set(scan.colours?.paint);                      // before the faces are named
    const faces: FaceInfo[] = FACES.map((f) => {
      const shot = scan.faces.find((x) => x.face === f)!;
      const colour = COLOUR_NAMES.indexOf(shot.solvedColour as (typeof COLOUR_NAMES)[number]);
      return {
        title: `Photo ${shot.shot + 1}`,
        fileName: shot.name,
        photo: shot.flat,
        chip: { text: `${PALETTE.nameOf(colour)} face · ${shot.holdPosition} when solving`, colour },
      };
    });
    this.session.scan = scan;
    this.session.shots = shots.slice();
    this.session.setCube(scan.observation, scan.decode, faces, 'scan');
    app.go('check');
  }

  photoCards(app: App, rebuild: () => void): HTMLElement[] {
    return [this.practiceCard(rebuild), this.arrayCard(app)];
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
          'Scramble the cube with moves the app gives you. The app then knows the exact state, so it can show how well the scan did, and the photos become labelled data for training the model on real pictures.',
        ),
      );
      c.append(
        button('Give me a scramble', 'btn btn--ghost', () => {
          const seed = (Math.random() * 2 ** 32) >>> 0;
          const moves = solverScramble(mulberry32(seed), 30);
          const truth = MalteseState.solved();
          truth.applyAll(moves.map((m) => m.engine));
          s.practice = { moves, truth, applied: false };
          rebuild();
        }),
      );
      return c;
    }
    c.append(
      h(
        'p',
        'mode-note',
        PALETTE.say('Start from a solved cube. Hold it <b>blue on top, yellow in front, purple on the left</b> (the white-green-red corner is at the back, bottom, right) and do these moves. U, L, F turn the top, left and front faces; a number repeats the quarter turn, ′ means counterclockwise.'),
      ),
    );
    const text = h('code', 'scramble__text');
    text.innerHTML = linesOfFour(p.moves.map((m) => m.token)).join('<br>');
    c.append(text);
    const box = h('div', 'cube-box cube-box--small');
    c.append(box);
    const cube = s.cube;
    cube.defineView('solver', SOLVER_VIEW);
    cube.mount(box);
    cube.setBadges({ B: 'U', R: 'L', D: 'F' }, { B: PALETTE.say('U: top (blue)'), R: PALETTE.say('L: left (purple)'), D: PALETTE.say('F: front (yellow)') });
    void cube.ready.then(() => {
      cube.show(p.truth);
      cube.setView('solver', false);
    });
    const row = h('div', 'actions');
    row.append(
      button('Play it on the model', 'btn btn--ghost', () => {
        void cube.ready.then(() => {
          cube.show(MalteseState.solved());
          for (const m of p.moves) cube.play(m.engine, TIMING.moveMs * 0.75);
        });
      }),
      button('Cancel', 'btn btn--ghost', () => {
        s.practice = null;
        rebuild();
      }),
    );
    c.append(row);
    c.append(h('p', 'hint hint--tight', 'When your cube matches the model, take the six photos.'));
    return c;
  }

  private arrayCard(app: App): HTMLElement {
    const c = card('Or paste a 144 array');
    c.append(h('p', 'mode-note', "The solver's array (96 stickers, 24 gear colours, 24 gear turns), to check it on the 3D model or solve it."));
    const area = document.createElement('textarea');
    area.className = 'array-input';
    area.rows = 3;
    area.placeholder = '4 4 4 0 3 4 2 1 …';
    area.spellcheck = false;
    c.append(area);
    const err = h('p', 'alg__error');
    err.hidden = true;
    const go = button('Check it', 'btn btn--ghost', () => void this.fromArray(app, area.value, err));
    const row = h('div', 'actions');
    row.append(go);
    c.append(row, err);
    return c;
  }

  private async fromArray(app: App, text: string, err: HTMLElement): Promise<void> {
    err.hidden = true;
    const nums = text
      .split(/[\s,;[\]]+/)
      .filter(Boolean)
      .map(Number);
    if (nums.length !== 144 || nums.some((n) => !Number.isInteger(n))) {
      err.textContent = `The array needs 144 whole numbers; this has ${nums.length}.`;
      err.hidden = false;
      return;
    }
    let obs;
    try {
      obs = toPlain(fromSolver144(nums, SOLVER_MAP));
    } catch (e) {
      err.textContent = e instanceof Error ? e.message : String(e);
      err.hidden = false;
      return;
    }
    try {
      const decode = await api.decode(obs);
      // A pasted array is in the solver's frame: name the faces by their solved colours.
      const faces: FaceInfo[] = FACES.map((f, i) => ({
        title: `${COLOUR_NAMES[i][0].toUpperCase()}${COLOUR_NAMES[i].slice(1)} face`,         // a pasted array: the usual colours
        chip: { text: `${HOLD_POSITION[f]} when solving`, colour: i },
      }));
      this.session.scan = null;
      this.session.setCube(obs, decode, faces, 'array');
      app.go('check');
    } catch (e) {
      err.textContent = e instanceof ApiError ? e.message : String(e);
      err.hidden = false;
    }
  }
}
