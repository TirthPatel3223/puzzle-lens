import { TIMING } from '../../vendor/maltese/config';
import { FACES, FACE_NORMAL, faceOf } from '../../vendor/maltese/geometry';
import { CUBE_GROUP } from '../../vendor/maltese/model';
import { type Move, invertMove, normalizeAmount } from '../../vendor/maltese/notation';
import type { MalteseState } from '../../vendor/maltese/state';
import { mat3Vec } from '../../vendor/puzzle/vec';
import { ApiError, api } from './api';
import { type App, type Screen, button, card, h, viewButtons } from '../app';
import type { MalteseSession } from './ui';
import { HOLD_POSITION, PALETTE, SOLVER_VIEW, SOLVER_VIEWS, type SolverMove, describeSolverMove, linesOfFour, parseSolverWord, stateFromPoses } from './model';

const SPEEDS = [0.5, 0.75, 1, 1.5, 2, 3, 4];

/** Step 3: solve the checked cube, then follow the solution move by move on the 3D model. */
export class SolveScreen implements Screen {
  constructor(private readonly ses: MalteseSession) {}

  private app!: App;
  private moves: SolverMove[] = [];
  private start: MalteseState | null = null;
  /** Moves done so far (the next one to do is moves[index]). */
  private index = 0;
  private playing = false;
  private speedIdx = 2;
  private alive = false;
  private moveCard!: HTMLElement;
  private listCard!: HTMLElement;
  private infoCard!: HTMLElement;
  private overlay!: HTMLElement;
  private viewSeg: ReturnType<typeof viewButtons> | null = null;

  enter(app: App): void {
    this.app = app;
    this.alive = true;
    this.playing = false;
    const wrap = h('div', 'solve');
    const box = h('div', 'cube-box cube-box--big');
    box.id = 'solve-cube';
    this.overlay = h('div', 'now-move');
    box.append(this.overlay);
    wrap.append(box);
    app.work.append(wrap);

    app.panel.append(app.header());
    app.panel.append(this.holdCard());
    this.moveCard = card('Solution');
    this.listCard = card('All moves');
    this.infoCard = card();
    this.listCard.hidden = true;
    this.infoCard.hidden = true;
    app.panel.append(this.moveCard, this.listCard, this.infoCard, app.serverLine());

    const cube = this.ses.cube;
    cube.mount(box);
    cube.defineView('solver', SOLVER_VIEW);
    cube.defineView('solver-corner', SOLVER_VIEWS.corner);
    cube.defineView('solver-face', SOLVER_VIEWS.face);
    cube.setBadges({ B: 'U', R: 'L', D: 'F' }, { B: PALETTE.say('U: top (blue)'), R: PALETTE.say('L: left (purple)'), D: PALETTE.say('F: front (yellow)') });
    cube.onApplied = () => this.afterMove();
    document.addEventListener('keydown', this.onKey);

    const d = this.ses.decode!;
    this.start = stateFromPoses(d.slot!, d.ori!);
    void cube.ready.then(() => {
      if (!this.alive) return;
      cube.show(this.start!);
      cube.setView('solver', false);
    });
    const key = d.state144!.join(',');
    if (this.ses.solution && this.ses.solution.forState === key) {
      this.begin(this.ses.solution.moves);
    } else {
      void this.solve(key);
    }
  }

  leave(): void {
    this.alive = false;
    this.playing = false;
    this.ses.cube.onApplied = () => {};
    this.ses.cube.highlight(null);
    document.removeEventListener('keydown', this.onKey);
  }

  // ------------------------------------------------------------------ solving

  private async solve(key: string): Promise<void> {
    const app = this.app;
    const say = (html: string) => {
      this.moveCard.replaceChildren(h('p', 'eyebrow', 'Solution'), h('p', 'status', html));
      this.overlay.innerHTML = `<span class="now-move__words">${html}</span>`;
    };
    try {
      let health = await app.refreshHealth();
      if (health && health.solver.state !== 'ready') {
        await api.loadSolver();
        const t0 = performance.now();
        while (this.alive) {
          health = await app.refreshHealth();
          const st = health?.solver.state;
          if (st === 'ready') break;
          if (st === 'missing' || st === 'error') throw new ApiError(health!.solver.detail || 'The solver could not be loaded.', 503);
          say(`<span class="spinner"></span> Loading the solver (the first time takes a while)… ${Math.round((performance.now() - t0) / 1000)} s`);
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
      if (!this.alive) return;
      say('<span class="spinner"></span> Searching for a solution (usually 10 to 20 seconds)…');
      const info = await api.solve(this.ses.decode!.state144!);
      if (!this.alive) return;
      const moves = parseSolverWord(info.word);
      // The server says the same thing in the 3D model's moves; they must agree.
      const agree =
        moves.length === info.engineMoves.length &&
        moves.every((m, i) => m.engine.face === info.engineMoves[i].face && m.engine.amount === normalizeAmount(info.engineMoves[i].amount));
      if (!agree) throw new Error('The solution could not be translated to the 3D model (the move dictionary disagrees).');
      const end = this.start!.clone();
      end.applyAll(moves.map((m) => m.engine));
      if (!end.isSolved()) app.toast('Careful: the solution does not solve the 3D model. Please report this scan.', 'error', 12000);
      this.ses.solution = { info, moves, start: this.start!, forState: key };
      this.begin(moves);
      void app.refreshHealth();
    } catch (err) {
      if (!this.alive) return;
      const msg = err instanceof ApiError || err instanceof Error ? err.message : String(err);
      say(`<span class="t-problem">${escapeHtml(msg)}</span>`);
      this.moveCard.append(button('Try again', 'btn btn--ghost', () => void this.solve(key)));
    }
  }

  // ------------------------------------------------------------------ following along

  private begin(moves: SolverMove[]): void {
    this.moves = moves;
    this.index = 0;
    this.buildMoveCard();
    this.buildList();
    this.buildInfo();
    void this.ses.cube.ready.then(() => {
      if (!this.alive) return;
      this.ses.cube.show(this.start!);
      this.refresh();
    });
  }

  private stateAt(k: number): MalteseState {
    const s = this.start!.clone();
    for (let i = 0; i < k; i++) s.apply(this.moves[i].engine);
    return s;
  }

  private duration(): number {
    return TIMING.moveMs / SPEEDS[this.speedIdx];
  }

  next(): void {
    if (this.index >= this.moves.length) return;
    if (this.ses.cube.play(this.moves[this.index].engine, this.duration())) {
      this.index++;
      this.refresh();
    }
  }

  prev(): void {
    if (this.index <= 0) return;
    this.playing = false;
    const m: Move = invertMove(this.moves[this.index - 1].engine);
    if (this.ses.cube.play(m, this.duration())) {
      this.index--;
      this.refresh();
    }
  }

  jump(k: number): void {
    this.playing = false;
    this.index = Math.max(0, Math.min(this.moves.length, k));
    this.ses.cube.show(this.stateAt(this.index));
    this.refresh();
  }

  togglePlay(): void {
    if (this.index >= this.moves.length) this.jump(0);
    this.playing = !this.playing;
    if (this.playing && !this.ses.cube.busy) this.next();
    this.refresh();
  }

  private afterMove(): void {
    if (!this.alive) return;
    if (this.playing && !this.ses.cube.busy) {
      if (this.index < this.moves.length) setTimeout(() => this.playing && this.alive && !this.ses.cube.busy && this.next(), 140);
      else this.playing = false;
    }
    this.refresh();
  }

  private onKey = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
    if (!this.moves.length) return;
    if (e.key === 'ArrowRight' || e.code === 'Space') {
      e.preventDefault();
      this.playing = false;
      this.next();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      this.prev();
    } else if (e.key === 'p' || e.key === 'P') {
      this.togglePlay();
    } else if (e.key === 'Home') {
      this.jump(0);
    } else if (e.key === 'End') {
      this.jump(this.moves.length);
    }
  };

  private buildMoveCard(): void {
    const c = this.moveCard;
    c.replaceChildren(h('p', 'eyebrow', 'Follow along'));
    c.append(h('div', 'move-now'));
    c.append(h('div', 'progress', '<div class="progress__bar"></div>'));
    const controls = h('div', 'player');
    const icon = (d: string) => `<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true"><path d="${d}"/></svg>`;
    controls.append(
      button(icon('M6 5h2v14H6zM20 5v14L9 12z'), 'btn btn--ghost btn--icon', () => this.jump(0), 'Back to the start (Home)'),
      button(icon('M17 5v14L6 12z'), 'btn btn--ghost btn--icon', () => this.prev(), 'Undo the last move (←)'),
      button('', 'btn btn--pill btn--play', () => this.togglePlay(), 'Play / pause (P)'),
      button(icon('M7 5v14l11-7z'), 'btn btn--ghost btn--icon', () => {
        this.playing = false;
        this.next();
      }, 'Next move (→ or Space)'),
      button(icon('M4 5v14l11-7zM16 5h2v14h-2z'), 'btn btn--ghost btn--icon', () => this.jump(this.moves.length), 'To the end (End)'),
    );
    c.append(controls);
    const speed = h('label', 'range');
    speed.innerHTML = `<span>Speed</span><input type="range" min="0" max="${SPEEDS.length - 1}" step="1" value="${this.speedIdx}" /><span class="speed-val"></span>`;
    const input = speed.querySelector('input')!;
    const val = speed.querySelector('.speed-val')!;
    const setVal = () => (val.textContent = `${SPEEDS[this.speedIdx]}×`);
    setVal();
    input.addEventListener('input', () => {
      this.speedIdx = parseInt(input.value, 10);
      setVal();
    });
    c.append(speed);
    c.append(h('p', 'hint hint--tight', 'Keys: <kbd>→</kbd> or <kbd>Space</kbd> next, <kbd>←</kbd> back, <kbd>P</kbd> play / pause.'));
  }

  private buildList(): void {
    const c = this.listCard;
    c.hidden = false;
    c.replaceChildren(h('p', 'eyebrow', `All moves (${this.moves.length})`));
    const list = h('ol', 'move-list');
    const lines = linesOfFour(this.moves.map((m) => m.token));
    lines.forEach((_, li) => {
      const row = h('li', 'move-list__line');
      for (let k = li * 4; k < Math.min(this.moves.length, li * 4 + 4); k++) {
        const b = button(this.moves[k].token, 'move-tok', () => this.jump(k), `Go to move ${k + 1}`);
        b.dataset.k = String(k);
        row.append(b);
      }
      list.append(row);
    });
    c.append(list);
  }

  private buildInfo(): void {
    const app = this.app;
    const s = this.ses.solution!.info;
    const c = this.infoCard;
    c.hidden = false;
    c.replaceChildren(h('p', 'eyebrow', 'About this solution'));
    c.append(
      h(
        'p',
        'mode-note',
        `${s.faceTurns} moves (${s.quarterTurns} quarter turns), found in ${s.seconds.toFixed(1)} s. Checked on the solver's own simulator and on this 3D model.`,
      ),
    );
    const row = h('div', 'actions');
    row.append(
      button('Copy the moves', 'btn btn--ghost btn--sm', () => {
        void navigator.clipboard.writeText(linesOfFour(this.moves.map((m) => m.token)).join('\n')).then(
          () => app.toast('The moves are on the clipboard, four to a line.', 'good'),
          () => app.toast('Copying was blocked by the browser.', 'error'),
        );
      }),
      button('Download (.txt)', 'btn btn--ghost btn--sm', () => {
        const text = [
          'Maltese Gear Cube solution',
          PALETTE.say('Hold the cube blue on top, yellow in front, purple on the left (the white-green-red corner at the back, bottom, right; it never moves).'),
          "U, L, F turn the top, left and front faces a quarter turn clockwise; ' is counterclockwise; a number repeats the quarter turn.",
          `144 array: ${this.ses.decode?.state144?.join(' ')}`,
          '',
          ...linesOfFour(this.moves.map((m) => m.token)),
        ].join('\n');
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
        a.download = 'maltese-solution.txt';
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      }),
      button('Scan another cube', 'btn btn--ghost btn--sm', () => {
        app.shots.forEach((x) => x.localUrl && URL.revokeObjectURL(x.localUrl));
        app.shots = [];
        this.ses.practice = null;
        app.go('photos');
      }),
    );
    c.append(row);
  }

  private refresh(): void {
    if (!this.moves.length) return;
    const n = this.moves.length;
    const i = this.index;
    const done = i >= n;
    const m = done ? null : this.moves[i];
    const solved = done && this.stateAt(n).isSolved();
    const now = this.moveCard.querySelector('.move-now');
    if (now) {
      now.innerHTML = m
        ? `<span class="move-now__token">${m.token}</span><span class="move-now__words">${describeSolverMove(m)}</span><span class="move-now__count">Move ${i + 1} of ${n}</span>`
        : `<span class="move-now__token move-now__token--done">${solved ? 'Solved' : 'Done'}</span><span class="move-now__words">${solved ? 'All moves done: the cube is solved.' : 'All moves done.'}</span><span class="move-now__count">${n} of ${n}</span>`;
    }
    this.overlay.innerHTML = m
      ? `<span class="now-move__token">${m.token}</span><span class="now-move__words">${describeSolverMove(m)}</span><span class="now-move__count">${i + 1} / ${n}</span>`
      : `<span class="now-move__token now-move__token--done">${solved ? 'Solved' : 'Done'}</span>`;
    const bar = this.moveCard.querySelector<HTMLElement>('.progress__bar');
    if (bar) bar.style.width = `${(100 * i) / n}%`;
    const play = this.moveCard.querySelector<HTMLButtonElement>('.btn--play');
    if (play) {
      play.innerHTML = this.playing
        ? '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg> Pause'
        : '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true"><path d="M7 5v14l12-7z"/></svg> Play';
    }
    this.listCard.querySelectorAll<HTMLElement>('.move-tok').forEach((b) => {
      const k = parseInt(b.dataset.k!, 10);
      b.classList.toggle('is-done', k < i);
      b.classList.toggle('is-now', k === i);
    });
    const cur = this.listCard.querySelector<HTMLElement>('.move-tok.is-now');
    if (cur) {
      const list = cur.closest('.move-list') as HTMLElement;
      const top = cur.offsetTop - list.offsetTop;
      if (top < list.scrollTop || top > list.scrollTop + list.clientHeight - 30) list.scrollTop = top - 40;
    }
    // Light up the layer of the next move while nothing is moving.
    if (!this.ses.cube.busy) this.ses.cube.highlight(m ? m.engine.face : null);
  }

  // ------------------------------------------------------------------ holding

  private holdCard(): HTMLElement {
    const app = this.app;
    const c = card('Hold the cube like this');
    const d = this.ses.decode!;
    // Which photo is on top, in front and on the left while solving (the decoder's rotation takes the
    // frame photographed to the solver's frame).
    let photos = '';
    if (this.ses.source === 'scan' && this.ses.scan) {
      const R = CUBE_GROUP.elements[d.rotation];
      const where: Record<string, string> = {};
      for (const f of this.ses.scan.faces) {
        const g = faceOf(mat3Vec(R, FACE_NORMAL[f.face]));
        if (g) where[HOLD_POSITION[g]] = `photo ${f.shot + 1}`;
      }
      photos = ` In your photos: top is ${where.top}, front is ${where.front}, left is ${where.left}.`;
    }
    c.append(
      h(
        'p',
        'mode-note',
        `${PALETTE.say('<b>Blue face on top, yellow in front, purple on the left.</b> The white-green-red corner is then at the back, bottom, right, and it never moves.')}${photos}`,
      ),
    );
    c.append(
      h(
        'p',
        'hint hint--tight',
        `On the model, U, L and F mark the three faces the solution turns: top (${PALETTE.nameOf(FACES.indexOf('B'))}), left (${PALETTE.nameOf(FACES.indexOf('R'))}) and front (${PALETTE.nameOf(FACES.indexOf('D'))}).`,
      ),
    );
    // Turn the model to match the cube in your hands (blue stays on top in both).
    c.append(h('p', 'view-label', 'Turn the model'));
    this.viewSeg = viewButtons(
      [
        { view: 'solver-corner', label: 'Corner', title: PALETTE.say('The blue-yellow-red corner towards you, blue on top') },
        { view: 'solver-face', label: 'Face-on', title: PALETTE.say('The yellow face towards you, blue on top') },
      ],
      (view) => this.ses.cube.setView(view),
    );
    c.append(this.viewSeg.el);
    const row = h('div', 'actions');
    row.append(button('Back to the check', 'btn btn--ghost btn--sm', () => app.go('check')));
    c.append(row);
    return c;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
