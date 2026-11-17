import { FACES, type Face } from '../../vendor/maltese/geometry';
import { ApiError, api } from './api';
import { type App, type Screen, button, card, h, viewButtons } from '../app';
import { coloursNote } from '../coloursNote';
import type { MalteseSession } from './ui';
import { faceCard } from './faceCard';
import { HELD_VIEWS, PALETTE, type PlainObservation, countDifferences, faceView, heldTruth, stateOf } from './model';

/** Under this probability a value the scanner settled on is marked "to check". */
const DOUBT = 0.9;

/** Step 2: the six faces beside the 3D model; fix anything that is wrong, then solve. */
export class CheckScreen implements Screen {
  /**
   * @param reread read the scan's photos again with the cube taken to have colour layout `paint` (the
   *               person chose another one); it shows the result.
   */
  constructor(
    private readonly ses: MalteseSession,
    private readonly reread: (app: App, paint: readonly number[]) => Promise<void> = async () => {},
  ) {}

  private app!: App;
  private cardsEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private nextEl!: HTMLElement;
  private decodeTimer = 0;
  private decodeSeq = 0;
  private truthHeld: PlainObservation | null = null;
  private viewSeg: ReturnType<typeof viewButtons> | null = null;

  enter(app: App): void {
    this.app = app;
    const obs = this.ses.obs!;
    this.truthHeld = this.ses.practice && this.ses.source === 'scan' ? heldTruth(this.ses.practice.truth, this.ses.original ?? obs) : null;

    // Work area: the six faces.
    const wrap = h('div', 'check');
    const top = h('div', 'check__top');
    top.append(
      h(
        'p',
        'check__lead',
        this.ses.source === 'scan'
          ? 'Each ring shows what the scanner decided for that sticker or gear. Compare the faces (and the 3D model) with your cube, and select any ring that is wrong to fix it.'
          : 'The faces as the array describes them. Select any ring to change it.',
      ),
    );
    top.append(this.legend());
    wrap.append(top);
    this.cardsEl = h('div', 'face-grid-cards');
    wrap.append(this.cardsEl);
    app.work.append(wrap);

    // Panel.
    app.panel.append(app.header());
    this.statusEl = card('Is it a real cube?');
    app.panel.append(this.statusEl);
    app.panel.append(this.modelCard());
    this.nextEl = card('Next');
    app.panel.append(this.nextEl);
    app.panel.append(app.serverLine());

    this.renderAll();
    this.show3D();
  }

  leave(): void {
    clearTimeout(this.decodeTimer);
  }

  // ------------------------------------------------------------------ rendering

  private renderAll(): void {
    this.renderCards();
    this.renderStatus();
    this.renderNext();
  }

  /** Faces in the order they were photographed (or FACES order for an array). */
  private faceOrder(): number[] {
    const scan = this.ses.scan;
    if (this.ses.source === 'scan' && scan) return scan.faces.map((f) => FACES.indexOf(f.face));
    return [0, 1, 2, 3, 4, 5];
  }

  private badgeOf(f: number): string {
    const scan = this.ses.scan;
    if (this.ses.source === 'scan' && scan) return String(scan.faces.find((x) => x.face === FACES[f])!.shot + 1);
    return PALETTE.nameOf(f)[0].toUpperCase();
  }

  private renderCards(): void {
    const obs = this.ses.obs!;
    const orig = this.ses.original!;
    const scan = this.ses.source === 'scan' ? this.ses.scan : null;
    const problemCells = new Set<number>();
    const problemGears = new Set<number>();
    for (const p of this.ses.decode?.problems ?? []) {
      p.cells.forEach((c) => problemCells.add(c));
      p.gears.forEach((g) => problemGears.add(g));
    }
    const changedCells = new Set<number>();
    const changedGears = new Set<number>();
    for (const c of scan?.decode.corrections ?? []) {
      if (c.kind === 'cell') changedCells.add(c.index);
      else changedGears.add(c.index);
    }
    const truth = this.truthHeld;
    const range = (n: number, from: number) => Array.from({ length: n }, (_, i) => from + i);
    const cards = this.faceOrder().map((f) => {
      const cells = range(16, 16 * f);
      const gears = range(4, 4 * f);
      const editedCell = (i: number) => obs.cells[i] !== orig.cells[i];
      const editedGear = (i: number) => obs.gearColour[i] !== orig.gearColour[i] || obs.gearQuarter[i] !== orig.gearQuarter[i];
      const info = this.ses.faces[f];
      return faceCard(
        {
          title: info.title,
          fileName: info.fileName,
          chip: info.chip,
          photo: info.photo,
          badge: this.badgeOf(f),
          cells: cells.map((i) => obs.cells[i]),
          gearColour: gears.map((i) => obs.gearColour[i]),
          gearQuarter: gears.map((i) => obs.gearQuarter[i]),
          doubt: scan ? cells.map((i) => scan.decode.cellConfidence[i] < DOUBT) : undefined,
          gearDoubt: scan ? gears.map((i) => scan.decode.gearConfidence[i] < DOUBT) : undefined,
          changed: cells.map((i) => changedCells.has(i) && !editedCell(i)),
          gearChanged: gears.map((i) => changedGears.has(i) && !editedGear(i)),
          edited: cells.map(editedCell),
          gearEdited: gears.map(editedGear),
          problem: cells.map((i) => problemCells.has(i)),
          gearProblem: gears.map((i) => problemGears.has(i)),
          differs: truth ? cells.map((i) => truth.cells[i] !== obs.cells[i]) : undefined,
          gearDiffers: truth ? gears.map((i) => truth.gearColour[i] !== obs.gearColour[i] || truth.gearQuarter[i] !== obs.gearQuarter[i]) : undefined,
          readCells: scan ? cells.map((i) => scan.reading.cells[i]) : undefined,
          readCellProb: scan ? cells.map((i) => scan.reading.cellProb[i]) : undefined,
          readGearColour: scan ? gears.map((i) => scan.reading.gearColour[i]) : undefined,
          readGearQuarter: scan ? gears.map((i) => scan.reading.gearQuarter[i]) : undefined,
          readGearProb: scan ? gears.map((i) => Math.min(scan.reading.gearColourProb[i], scan.reading.gearQuarterProb[i])) : undefined,
        },
        {
          onCell: (c, colour) => this.edit(() => (obs.cells[16 * f + c] = colour)),
          onGear: (p, colour, q) =>
            this.edit(() => {
              obs.gearColour[4 * f + p] = colour;
              obs.gearQuarter[4 * f + p] = q;
            }),
          onShow3D: () => {
            this.viewSeg?.set(null);
            this.ses.cube.setView(`face-${FACES[f]}`);
            this.ses.cube.mount(document.getElementById('check-cube')!);
          },
        },
      );
    });
    this.cardsEl.replaceChildren(...cards);
  }

  private legend(): HTMLElement {
    const l = h('div', 'legend');
    l.innerHTML = `
      <span><i class="lg lg--ring"></i>decoded colour</span>
      <span><i class="lg lg--doubt"></i>scanner unsure: check it</span>
      <span><i class="lg lg--changed"></i>fixed by the decoder</span>
      <span><i class="lg lg--edited"></i>changed by you</span>
      <span><i class="lg lg--problem"></i>impossible</span>
      <span><i class="lg lg--arrow"></i>gear: long teeth point here</span>`;
    return l;
  }

  private renderStatus(): void {
    const d = this.ses.decode;
    const c = this.statusEl;
    c.replaceChildren(h('p', 'eyebrow', 'Is it a real cube?'));
    if (!d) {
      c.append(h('p', 'status', 'Checking…'));
      return;
    }
    if (d.legal) {
      c.append(h('p', 'status is-good', 'Yes: every piece is there once, and moves can reach this arrangement.'));
    } else {
      c.append(h('p', 'status is-alert', 'Not yet. The rings in red are part of these problems:'));
      const ul = h('ul', 'problems');
      for (const p of d.problems.slice(0, 8)) ul.append(h('li', '', escapeHtml(p.message)));
      if (d.problems.length > 8) ul.append(h('li', '', `…and ${d.problems.length - 8} more.`));
      c.append(ul);
    }
    const scan = this.ses.source === 'scan' ? this.ses.scan : null;
    if (scan) {
      const doubt = scan.decode.cellConfidence.filter((p) => p < DOUBT).length + scan.decode.gearConfidence.filter((p) => p < DOUBT).length;
      const facts = [
        `Read in ${scan.seconds.toFixed(1)} s`,
        `${scan.decode.corrections.length} place${scan.decode.corrections.length === 1 ? '' : 's'} fixed by the decoder`,
        `${doubt} to check`,
      ];
      c.append(h('p', 'hint hint--tight', facts.join(' · ')));
      if (scan.arrangement.doubtful) {
        c.append(
          h(
            'p',
            'status is-alert',
            'The photos could not be fitted together with confidence: maybe one face was photographed twice or a photo is very unclear. Check which face each card shows.',
          ),
        );
      }
      const colours = coloursNote(scan.colours, scan.alternatives, scan.arrangement.doubtful, (paint) => this.reread(this.app, paint), (err) =>
        this.app.toast(err instanceof ApiError ? err.message : String(err), 'error'),
      );
      if (colours) c.append(colours);
    }
    const edits = this.ses.original && this.ses.obs ? countDifferences(this.ses.original, this.ses.obs) : 0;
    if (edits) c.append(h('p', 'hint hint--tight', `You changed ${edits} place${edits === 1 ? '' : 's'}.`));
    if (this.truthHeld && this.ses.obs) {
      const diff = countDifferences(this.truthHeld, this.ses.obs);
      const fromScan = this.ses.original ? countDifferences(this.truthHeld, this.ses.original) : diff;
      const p = h(
        'p',
        `status ${diff ? 'is-alert' : 'is-good'}`,
        diff
          ? `Compared with your scramble, ${diff} place${diff === 1 ? '' : 's'} differ (marked ✕). The scan itself had ${fromScan} wrong.`
          : `This matches your scramble exactly${fromScan ? ` (the scan had ${fromScan} wrong; you fixed ${fromScan === 1 ? 'it' : 'them'})` : ': the scan was right'}.`,
      );
      c.append(p);
      if (diff) {
        c.append(
          button('Use the scramble’s state', 'btn btn--ghost btn--sm', () => {
            const t = this.truthHeld!;
            this.edit(() => {
              this.ses.obs!.cells.splice(0, 96, ...t.cells);
              this.ses.obs!.gearColour.splice(0, 24, ...t.gearColour);
              this.ses.obs!.gearQuarter.splice(0, 24, ...t.gearQuarter);
            });
          }),
        );
      }
    }
  }

  private modelCard(): HTMLElement {
    const c = card('3D model');
    const box = h('div', 'cube-box');
    box.id = 'check-cube';
    c.append(box);
    // The 3D simulator site's three views (in the frame shown: photo 1 is the front, its top edge up),
    // then one button per photo.
    const scan = this.ses.source === 'scan';
    const name = (face: Face) => (scan ? `photo ${this.badgeOf(FACES.indexOf(face))}` : `the ${PALETTE.nameOf(FACES.indexOf(face))} face`);
    this.viewSeg = viewButtons(
      [
        { view: 'held-corner', label: 'Corner', title: `The top-front-right corner: ${name('U')} on top, ${name('F')} front-left, ${name('R')} front-right` },
        { view: 'held-face', label: 'Face-on', title: `Straight at ${name('F')}, ${name('U')} on top` },
        { view: 'held-edge', label: 'Edge', title: `The edge between ${name('U')} (top) and ${name('F')} (front) towards you` },
      ],
      (view) => this.ses.cube.setView(view),
    );
    c.append(this.viewSeg.el);
    const views = h('div', 'view-row');
    for (const f of this.faceOrder()) {
      views.append(
        button(scan ? `Photo ${this.badgeOf(f)}` : PALETTE.nameOf(f), 'chip-btn', () => {
          this.viewSeg?.set(null);
          this.ses.cube.setView(`face-${FACES[f]}`);
        }),
      );
    }
    c.append(views);
    c.append(
      h(
        'p',
        'hint hint--tight',
        this.ses.source === 'scan'
          ? 'Drag to turn the model. The numbers on it are the photos.'
          : 'Drag to turn the model. The letters on it are the faces’ solved colours.',
      ),
    );
    return c;
  }

  private show3D(): void {
    const cube = this.ses.cube;
    cube.mount(document.getElementById('check-cube')!);
    for (const [k, v] of Object.entries(HELD_VIEWS)) cube.defineView(`held-${k}`, v);
    for (const f of FACES) cube.defineView(`face-${f}`, faceView(f));
    const badges: Partial<Record<Face, string>> = {};
    for (const f of this.faceOrder()) badges[FACES[f]] = this.badgeOf(f);
    cube.setBadges(badges);
    this.viewSeg?.set('held-corner');
    void cube.ready.then(() => {
      this.update3D();
      cube.setView('held-corner', false);
    });
  }

  private update3D(): void {
    const st = stateOf(this.ses.obs!);
    const box = document.getElementById('check-cube');
    box?.classList.toggle('is-stale', !st);
    if (st) this.ses.cube.show(st);
  }

  private renderNext(): void {
    const app = this.app;
    const c = this.nextEl;
    c.replaceChildren(h('p', 'eyebrow', 'Next'));
    const legal = !!this.ses.decode?.legal && !!this.ses.decode.state144;
    const row = h('div', 'actions');
    const solve = button('Solve this cube', 'btn btn--primary', () => app.go('solve'));
    solve.disabled = !legal;
    row.append(solve);
    c.append(row);
    const row2 = h('div', 'actions');
    const copy = button('Copy the 144 array', 'btn btn--ghost btn--sm', () => {
      const a = this.ses.decode?.state144;
      if (!a) return;
      void navigator.clipboard.writeText(a.join(' ')).then(
        () => app.toast('The 144 array is on the clipboard.', 'good'),
        () => app.toast('Copying was blocked by the browser.', 'error'),
      );
    });
    copy.disabled = !legal;
    const save = button('Save the scan (.json)', 'btn btn--ghost btn--sm', () => this.download());
    row2.append(copy, save);
    if (this.ses.original && this.ses.obs && countDifferences(this.ses.original, this.ses.obs)) {
      row2.append(
        button('Undo my changes', 'btn btn--ghost btn--sm', () =>
          this.edit(() => {
            const o = this.ses.original!;
            this.ses.obs!.cells.splice(0, 96, ...o.cells);
            this.ses.obs!.gearColour.splice(0, 24, ...o.gearColour);
            this.ses.obs!.gearQuarter.splice(0, 24, ...o.gearQuarter);
          }),
        ),
      );
    }
    c.append(row2);
    if (legal && this.ses.decode?.state144) {
      const arr = h('details', 'array-view');
      arr.innerHTML = `<summary>Show the 144 array</summary><code>${this.ses.decode.state144.join(' ')}</code>`;
      c.append(arr);
    }

    // Keeping the photos as labelled real data (to measure and fine-tune the model on real pictures).
    if (this.ses.source === 'scan' && this.ses.scan) {
      const keep = h('div', 'keep');
      if (this.ses.kept) {
        keep.append(h('p', 'hint hint--tight', 'These photos are kept as labelled real data.'));
      } else {
        const practice = !!this.truthHeld;
        keep.append(
          h(
            'p',
            'hint hint--tight',
            practice
              ? 'Keep these photos with the scramble’s exact state, to measure the scanner and train it on real pictures.'
              : 'Once every face matches your cube, keep these photos with the checked state, to train the scanner on real pictures.',
          ),
        );
        const b = button(practice ? 'Keep as labelled data' : 'Keep as labelled data (checked)', 'btn btn--ghost btn--sm', () => void this.keep());
        b.disabled = practice ? false : !legal;
        keep.append(b);
      }
      c.append(keep);
    }
  }

  // ------------------------------------------------------------------ actions

  private edit(change: () => void): void {
    change();
    this.ses.solution = null;
    this.ses.kept = false;
    // Until the server has checked the new state, nothing is known about it (and it cannot be solved).
    this.ses.decode = null;
    this.app.renderSteps();
    this.renderCards();
    this.renderStatus();
    this.renderNext();
    this.update3D();
    clearTimeout(this.decodeTimer);
    const seq = ++this.decodeSeq;
    this.decodeTimer = window.setTimeout(async () => {
      try {
        const d = await api.decode(this.ses.obs!);
        if (seq !== this.decodeSeq) return;
        this.ses.decode = d;
        this.renderCards();
        this.renderStatus();
        this.renderNext();
        this.app.renderSteps();
      } catch (err) {
        this.app.toast(err instanceof ApiError ? err.message : String(err), 'error');
      }
    }, 180);
  }

  private async keep(): Promise<void> {
    const app = this.app;
    if (!this.ses.scan) return;
    const practice = !!this.truthHeld;
    try {
      const r = await api.confirm(this.ses.scan.id, practice ? this.truthHeld! : this.ses.obs!, practice ? 'known-scramble' : 'checked', PALETTE.paint);
      this.ses.kept = true;
      const errs = r.readErrors ? r.readErrors.cells + r.readErrors.gearColour + r.readErrors.gearQuarter : null;
      app.toast(
        `Kept. ${r.scans} labelled real scan${r.scans === 1 ? '' : 's'} so far.${errs !== null ? ` The camera misread ${errs} place${errs === 1 ? '' : 's'} on this one${r.decodedRight ? ', and the decoder fixed all of them' : ''}.` : ''}`,
        'good',
        8000,
      );
      this.renderNext();
      void app.refreshHealth();
    } catch (err) {
      app.toast(err instanceof ApiError ? err.message : String(err), 'error');
    }
  }

  private download(): void {
    const data = {
      puzzle: 'maltese-gear-cube',
      savedAt: new Date().toISOString(),
      source: this.ses.source,
      scanId: this.ses.scan?.id ?? null,
      photos: this.ses.scan?.faces.map((f) => ({ shot: f.shot, name: f.name, face: f.face, solvedColour: f.solvedColour, turn: f.turn })) ?? null,
      frame: this.ses.source === 'scan' ? 'as photographed: photo 1 is F, its top edge up' : "the solver's frame",
      observation: this.ses.obs,
      legal: this.ses.decode?.legal ?? false,
      state144: this.ses.decode?.state144 ?? null,
      practiceScramble: this.ses.practice?.moves.map((m) => m.token).join(' ') ?? null,
    };
    const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `maltese-scan-${this.ses.scan?.id ?? 'array'}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
