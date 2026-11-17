import { TIMING } from '../../vendor/tangram/config';
import { type Move, invertMove } from '../../vendor/tangram/moves';
import type { PuzzleState } from '../../vendor/tangram/state';
import { stateFromOrient } from '../../tangram/model';
import { type App, type Screen, button, card, download, escapeHtml, h, viewButtons } from '../app';
import { ApiError, type TangramSolution, api } from './api';
import { LANDMARKS, PALETTE, countDifferences, describeMove, linesOf, moveToken, observeState, parseMoves, stateAfter, undoMoves } from './model';
import type { TangramSession } from './ui';

const SPEEDS = [0.5, 0.75, 1, 1.5, 2, 3, 4];

/** Step 3: solve the checked puzzle, then follow the solution move by move on the 3D model. */
export class TangramSolve implements Screen {
  constructor(private readonly ses: TangramSession) {}

  private app!: App;
  private moves: Move[] = [];
  private start: PuzzleState | null = null;
  private index = 0;
  private playing = false;
  private speedIdx = 2;
  private alive = false;
  private moveCard!: HTMLElement;
  private listCard!: HTMLElement;
  private infoCard!: HTMLElement;
  private overlay!: HTMLElement;

  enter(app: App): void {
    this.app = app;
    this.alive = true;
    this.playing = false;
    const s = this.ses;
    const wrap = h('div', 'solve');
    const box = h('div', 'cube-box cube-box--big');
    box.id = 'solve-cube';
    this.overlay = h('div', 'now-move');
    box.append(this.overlay);
    wrap.append(box);
    app.work.append(wrap);

    app.panel.append(app.header(), this.holdCard());
    this.moveCard = card('Solution');
    this.listCard = card('All moves');
    this.infoCard = card();
    this.listCard.hidden = true;
    this.infoCard.hidden = true;
    app.panel.append(this.moveCard, this.listCard, this.infoCard, app.serverLine());

    const view = s.view;
    view.mount(box);
    view.showLabels(true);
    view.setBadges([]);
    view.onApplied = () => this.afterMove();
    document.addEventListener('keydown', this.onKey);
    this.start = stateFromOrient(s.decode!.orient!);
    view.show(this.start);
    view.setView('corner', false);
    const key = s.decode!.orient!.join(',');
    if (s.solution && s.solution.forState === key) this.begin(s.solution.moves);
    else void this.solve(key);
  }

  leave(): void {
    this.alive = false;
    this.playing = false;
    this.ses.view.onApplied = () => {};
    this.ses.view.highlight(null);
    document.removeEventListener('keydown', this.onKey);
  }

  // ------------------------------------------------------------------ solving

  private say(html: string): void {
    this.moveCard.replaceChildren(h('p', 'eyebrow', 'Solution'), h('p', 'status', html));
    this.overlay.innerHTML = `<span class="now-move__words">${html}</span>`;
  }

  private async solve(key: string): Promise<void> {
    const app = this.app;
    const s = this.ses;
    try {
      let health = await app.refreshHealth();
      if (app.puzzleHealth()?.solver.state !== 'ready') {
        await api.loadSolver();
        const t0 = performance.now();
        while (this.alive) {
          health = await app.refreshHealth();
          const st = app.puzzleHealth()?.solver;
          if (!health || !st || st.state === 'ready') break;
          if (st.state === 'missing' || st.state === 'error') throw new ApiError(st.detail || 'The solver could not be loaded.', 503);
          this.say(`<span class="spinner"></span> Loading the solver… ${Math.round((performance.now() - t0) / 1000)} s`);
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
      if (!this.alive) return;
      this.say('<span class="spinner"></span> Searching for a solution…');
      const info = await api.solve(s.decode!.orient!);
      if (!this.alive) return;
      this.accept(info, key);
    } catch (err) {
      if (!this.alive) return;
      this.failed(err, key);
    }
  }

  /** The server's answer, checked once more on this page's own model before it is shown. */
  private accept(info: TangramSolution, key: string): void {
    const moves = parseMoves(info.word);
    const end = stateAfter(moves, this.start!);
    if (!end.isSolved()) this.app.toast('Careful: the solution does not solve the 3D model. Please report this scan.', 'error', 12000);
    this.ses.solution = { info, moves, forState: key };
    this.begin(moves);
    void this.app.refreshHealth();
  }

  private failed(err: unknown, key: string): void {
    const s = this.ses;
    const notConnected = err instanceof ApiError && err.status === 503;
    const msg = err instanceof ApiError || err instanceof Error ? err.message : String(err);
    this.moveCard.replaceChildren(h('p', 'eyebrow', 'Solution'));
    if (notConnected) {
      this.moveCard.append(
        h('p', 'status is-alert', 'The Tangram Cube Extreme solver is not available.'),
        h('p', 'hint hint--tight', escapeHtml(msg)),
      );
    } else {
      this.moveCard.append(h('p', 'status', `<span class="t-problem">${escapeHtml(msg)}</span>`));
    }
    this.overlay.innerHTML = `<span class="now-move__words">${notConnected ? 'No solver yet' : 'No solution'}</span>`;
    const row = h('div', 'actions');
    row.append(button('Try again', 'btn btn--ghost', () => void this.solve(key)));
    // With a practice scramble, its undo is a solution: enough to follow along until the solver is here.
    const p = s.practice;
    if (p && countDifferences(observeState(p.truth), observeState(this.start!)) === 0) {
      row.append(
        button('Follow the undo of my scramble', 'btn btn--primary', () => {
          const moves = undoMoves(p.moves);
          const info: TangramSolution = {
            word: moves.map(moveToken).join(' '),
            moves: moves.map((m) => ({ axis: m.axis, amount: m.amount, token: moveToken(m) })),
            turns: moves.length,
            steps: moves.reduce((n, m) => n + Math.abs(m.amount), 0),
            seconds: 0,
            verified: true,
            source: 'practice-undo',
          };
          this.accept(info, key);
        }),
      );
    }
    this.moveCard.append(row);
  }

  // ------------------------------------------------------------------ following along

  private begin(moves: Move[]): void {
    this.moves = moves;
    this.index = 0;
    this.buildMoveCard();
    this.buildList();
    this.buildInfo();
    if (!this.alive) return;
    this.ses.view.show(this.start!);
    this.refresh();
  }

  private stateAt(k: number): PuzzleState {
    return stateAfter(this.moves.slice(0, k), this.start!);
  }

  private duration(): number {
    return (TIMING.moveMs * 2.2) / SPEEDS[this.speedIdx];
  }

  next(): void {
    if (this.index >= this.moves.length) return;
    this.ses.view.play(this.moves[this.index], this.duration());
    this.index++;
    this.refresh();
  }

  prev(): void {
    if (this.index <= 0) return;
    this.playing = false;
    this.ses.view.play(invertMove(this.moves[this.index - 1]), this.duration());
    this.index--;
    this.refresh();
  }

  jump(k: number): void {
    this.playing = false;
    this.index = Math.max(0, Math.min(this.moves.length, k));
    this.ses.view.show(this.stateAt(this.index));
    this.refresh();
  }

  togglePlay(): void {
    if (this.index >= this.moves.length) this.jump(0);
    this.playing = !this.playing;
    if (this.playing && !this.ses.view.busy) this.next();
    this.refresh();
  }

  private afterMove(): void {
    if (!this.alive) return;
    const view = this.ses.view;
    if (this.playing && !view.busy) {
      if (this.index < this.moves.length) setTimeout(() => this.playing && this.alive && !view.busy && this.next(), 160);
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
    c.append(h('div', 'move-now'), h('div', 'progress', '<div class="progress__bar"></div>'));
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
    c.append(speed, h('p', 'hint hint--tight', 'Keys: <kbd>→</kbd> or <kbd>Space</kbd> next, <kbd>←</kbd> back, <kbd>P</kbd> play / pause.'));
  }

  private buildList(): void {
    const c = this.listCard;
    c.hidden = false;
    c.replaceChildren(h('p', 'eyebrow', `All moves (${this.moves.length})`));
    const list = h('ol', 'move-list move-list--6');
    const per = 6;
    for (let li = 0; li * per < this.moves.length; li++) {
      const row = h('li', 'move-list__line');
      for (let k = li * per; k < Math.min(this.moves.length, li * per + per); k++) {
        const b = button(escapeHtml(moveToken(this.moves[k])), 'move-tok', () => this.jump(k), `Go to move ${k + 1}`);
        b.dataset.k = String(k);
        row.append(b);
      }
      list.append(row);
    }
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
        s.source === 'practice-undo'
          ? `${s.turns} moves: your practice scramble undone (the solver is not available). Checked on this 3D model.`
          : `${s.turns} moves (${s.steps} steps of 60° or flips), found in ${s.seconds.toFixed(1)} s. Checked on the scanner's model of the puzzle and on this 3D model.`,
      ),
    );
    const text = () => linesOf(this.moves.map(moveToken)).join('\n');
    const row = h('div', 'actions');
    row.append(
      button('Copy the moves', 'btn btn--ghost btn--sm', () => {
        void navigator.clipboard.writeText(text()).then(
          () => app.toast('The moves are on the clipboard, six to a line.', 'good'),
          () => app.toast('Copying was blocked by the browser.', 'error'),
        );
      }),
      button('Download (.txt)', 'btn btn--ghost btn--sm', () =>
        download(
          'tangram-solution.txt',
          [
            'Tangram Cube Extreme solution (the 3D simulator\'s notation)',
            `U turns ${LANDMARKS.U}, D ${LANDMARKS.D}: one step is 60° clockwise seen from that corner, ' is counterclockwise, a number repeats the step.`,
            `F, R, L flip ${LANDMARKS.F}, ${LANDMARKS.R} and ${LANDMARKS.L} half a turn.`,
            `State: ${this.ses.obs?.join(' ')}`,
            '',
            text(),
          ].join('\n'),
        ),
      ),
      button('Scan another puzzle', 'btn btn--ghost btn--sm', () => {
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
    const tok = (x: Move) => escapeHtml(moveToken(x));
    const now = this.moveCard.querySelector('.move-now');
    if (now) {
      now.innerHTML = m
        ? `<span class="move-now__token">${tok(m)}</span><span class="move-now__words">${describeMove(m)}</span><span class="move-now__count">Move ${i + 1} of ${n}</span>`
        : `<span class="move-now__token move-now__token--done">${solved ? 'Solved' : 'Done'}</span><span class="move-now__words">${solved ? 'All moves done: the puzzle is solved.' : 'All moves done.'}</span><span class="move-now__count">${n} of ${n}</span>`;
    }
    this.overlay.innerHTML = m
      ? `<span class="now-move__token">${tok(m)}</span><span class="now-move__words">${describeMove(m)}</span><span class="now-move__count">${i + 1} / ${n}</span>`
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
    if (!this.ses.view.busy) this.ses.view.highlight(m ? m.axis : null);
  }

  // ------------------------------------------------------------------ holding

  private holdCard(): HTMLElement {
    const c = card('Hold the puzzle like this');
    c.append(
      h(
        'p',
        'mode-note',
        PALETTE.say(`<b>The orange-yellow-green corner up</b> (it is the T corner, the turning corner U turns; it only ever twists in place), whatever shape the puzzle is in. The moves are named by pieces that never leave their place:`),
      ),
    );
    const ul = h('ul', 'steps-list');
    ul.innerHTML = `
      <li><b>U</b>: ${LANDMARKS.U}; <b>D</b>: ${LANDMARKS.D}. One step is 60° clockwise seen from that corner; ′ is counterclockwise; U2 is two steps.</li>
      <li><b>F</b>: ${LANDMARKS.F}; <b>R</b>: ${LANDMARKS.R}; <b>L</b>: ${LANDMARKS.L}. A flip turns that edge's slab half a turn.</li>`;
    c.append(ul, h('p', 'hint hint--tight', 'On the model, the labels U D F R L mark those parts, and the part that turns next is lit up.'));
    c.append(h('p', 'view-label', 'Turn the model'));
    const seg = viewButtons(
      [
        { view: 'corner', label: 'Corner', title: PALETTE.say('The T corner (orange-yellow-green) towards you, green on top') },
        { view: 'face', label: 'Face-on', title: PALETTE.say('The orange face straight on') },
        { view: 'edge', label: 'Edge', title: 'The L flip edge towards you' },
      ],
      (view) => this.ses.view.setView(view),
    );
    c.append(seg.el);
    const row = h('div', 'actions');
    row.append(button('Back to the check', 'btn btn--ghost btn--sm', () => this.app.go('check')));
    c.append(row);
    return c;
  }
}
