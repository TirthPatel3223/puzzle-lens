import { FACE_IDS } from '../../vendor/tangram/pieces';
import { stateFromOrient } from '../../tangram/model';
import { type App, type Screen, button, card, download, escapeHtml, h, viewButtons } from '../app';
import { coloursNote } from '../coloursNote';
import { ApiError, api } from './api';
import { FACE_HALF, type MarkModel, drawnMarks, tangramFaceCard } from './faceCard';
import { EMPTY, FACE_END, LOOKALIKES, PALETTE, countDifferences, faceName, observeState } from './model';
import type { TangramSession } from './ui';

/** Under this probability a value the scanner settled on is marked "to check". */
const DOUBT = 0.85;

/** Step 2: the six faces beside the 3D model; fix anything that is wrong, then solve. */
export class TangramCheck implements Screen {
  /**
   * @param reread read the scan's photos again with the cube taken to have colour layout `paint`
   *               (the person chose another one); it shows the result.
   */
  constructor(
    private readonly ses: TangramSession,
    private readonly reread: (app: App, paint: readonly number[]) => Promise<void> = async () => {},
  ) {}

  private app!: App;
  private cardsEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private nextEl!: HTMLElement;
  private decodeTimer = 0;
  private decodeSeq = 0;
  private truth: number[] | null = null;
  private viewSeg: ReturnType<typeof viewButtons> | null = null;

  enter(app: App): void {
    this.app = app;
    const s = this.ses;
    this.truth = s.practice && s.source === 'scan' ? observeState(s.practice.truth) : null;

    const wrap = h('div', 'check');
    const top = h('div', 'check__top');
    top.append(
      h(
        'p',
        'check__lead',
        s.source === 'scan'
          ? 'Each marker shows what the scanner decided for that place: a circle for a sticker flat on the face, a diamond for one turned out of it, a small dot for no sticker. Compare the faces (and the 3D model) with your puzzle, and select any marker that is wrong to fix it.'
          : 'The faces as the moves leave them. Select any marker to change it.',
      ),
    );
    top.append(this.legend());
    wrap.append(top);
    this.cardsEl = h('div', 'face-grid-cards');
    wrap.append(this.cardsEl);
    app.work.append(wrap);

    app.panel.append(app.header());
    this.statusEl = card('Is it a real puzzle?');
    app.panel.append(this.statusEl, this.modelCard());
    this.nextEl = card('Next');
    app.panel.append(this.nextEl, app.serverLine());

    this.renderAll();
    this.show3D();
  }

  leave(): void {
    clearTimeout(this.decodeTimer);
  }

  private renderAll(): void {
    this.renderCards();
    this.renderStatus();
    this.renderNext();
  }

  /** Faces in the order photographed (or FACE_IDS order without photos). */
  private faceOrder(): number[] {
    const scan = this.ses.scan;
    if (this.ses.source === 'scan' && scan) return scan.photos.map((p) => p.faceIndex);
    return [0, 1, 2, 3, 4, 5];
  }

  private badgeOf(face: number): string {
    const scan = this.ses.scan;
    if (this.ses.source === 'scan' && scan) return String(scan.photos.find((p) => p.faceIndex === face)!.shot + 1);
    return PALETTE.nameOf(face)[0].toUpperCase();
  }

  private legend(): HTMLElement {
    const l = h('div', 'legend');
    l.innerHTML = `
      <span><i class="lg lg--ring"></i>sticker, flat</span>
      <span><i class="lg lg--diamond"></i>sticker, turned out</span>
      <span><i class="lg lg--none"></i>no sticker</span>
      <span><i class="lg lg--doubt"></i>scanner unsure: check it</span>
      <span><i class="lg lg--changed"></i>fixed by the decoder</span>
      <span><i class="lg lg--edited"></i>changed by you</span>
      <span><i class="lg lg--problem"></i>impossible</span>
      <span><i class="lg lg--corner"></i>turning corner</span>`;
    return l;
  }

  private renderCards(): void {
    const s = this.ses;
    const obs = s.obs!;
    const orig = s.original!;
    const scan = s.source === 'scan' ? s.scan : null;
    const problem = new Set<number>();
    for (const p of s.decode?.problems ?? []) p.positions.forEach((x) => problem.add(x));
    const changed = new Set((scan?.decode.corrections ?? []).map((c) => c.position));
    const truth = this.truth;
    const cards = this.faceOrder().map((face) => {
      const photo = scan?.photos.find((p) => p.faceIndex === face);
      const at = photo
        ? photo.positions.map((p) => ({ position: p.position, x: p.x * 2 * FACE_HALF - FACE_HALF, y: FACE_HALF - p.y * 2 * FACE_HALF, flat: p.flat, read: p.read, readProb: p.prob }))
        : drawnMarks(face);
      const marks: MarkModel[] = at.map((a) => {
        const p = a.position;
        const edited = obs[p] !== orig[p];
        return {
          ...a,
          value: obs[p],
          doubt: scan ? scan.decode.confidence[p] < DOUBT : false,
          changed: changed.has(p) && !edited,
          edited,
          problem: problem.has(p),
          differs: truth ? truth[p] !== obs[p] : false,
        };
      });
      const end = FACE_END[face];
      return tangramFaceCard(
        {
          title: photo ? `Photo ${photo.shot + 1}` : capital(faceName(face)),
          fileName: photo?.name,
          photo: photo?.flat,
          chip: { text: `${faceName(face)} (${FACE_IDS[face]}) · ${end} corner`, colour: face },
          badge: this.badgeOf(face),
          face,
          observation: obs,
          marks,
        },
        {
          onSet: (position, value) =>
            this.edit(() => {
              obs[position] = value;
              // A sticker here means none on the positions that share its centre (turned 60° from it).
              if (value !== EMPTY) for (const q of LOOKALIKES[position]) obs[q] = EMPTY;
            }),
          onShow3D: () => {
            this.viewSeg?.set(null);
            s.view.setView(`face-${face}`);
          },
        },
      );
    });
    this.cardsEl.replaceChildren(...cards);
  }

  private renderStatus(): void {
    const s = this.ses;
    const d = s.decode;
    const c = this.statusEl;
    c.replaceChildren(h('p', 'eyebrow', 'Is it a real puzzle?'));
    if (!d) {
      c.append(h('p', 'status', 'Checking…'));
      return;
    }
    if (d.legal) {
      c.append(h('p', 'status is-good', 'Yes: every piece is there once, and moves can reach this state.'));
    } else {
      c.append(h('p', 'status is-alert', 'Not yet. The markers in red are part of these problems:'));
      const ul = h('ul', 'problems');
      for (const p of d.problems.slice(0, 8)) ul.append(h('li', '', escapeHtml(p.message)));
      if (d.problems.length > 8) ul.append(h('li', '', `…and ${d.problems.length - 8} more.`));
      c.append(ul);
    }
    const scan = s.source === 'scan' ? s.scan : null;
    if (scan) {
      const doubt = scan.decode.confidence.filter((p) => p < DOUBT).length;
      const facts = [`Read in ${scan.seconds.toFixed(1)} s`, `${scan.decode.corrections.length} place${scan.decode.corrections.length === 1 ? '' : 's'} fixed by the decoder`, `${doubt} to check`];
      c.append(h('p', 'hint hint--tight', facts.join(' · ')));
      if (scan.decode.repairs.length) c.append(h('p', 'hint hint--tight', `To make it reachable the decoder assumed ${escapeHtml(scan.decode.repairs.join(' and '))}.`));
      const turned = scan.photos.filter((p) => p.turned);
      if (turned.length) c.append(h('p', 'hint hint--tight', `Photo${turned.length > 1 ? 's' : ''} ${turned.map((p) => p.shot + 1).join(', ')}: the turning corner was not where the scanner first looked; check those faces.`));
      if (scan.doubtful) {
        c.append(h('p', 'status is-alert', 'The photos could not be fitted together with confidence: maybe a face was photographed twice, corner-on, or very unclearly. Check which face each card shows.'));
      }
      const colours = coloursNote(scan.colours, scan.alternatives, scan.doubtful, (paint) => this.reread(this.app, paint), (err) =>
        this.app.toast(err instanceof ApiError ? err.message : String(err), 'error'),
      );
      if (colours) c.append(colours);
    }
    const edits = s.original && s.obs ? countDifferences(s.original, s.obs) : 0;
    if (edits) c.append(h('p', 'hint hint--tight', `You changed ${edits} place${edits === 1 ? '' : 's'}.`));
    if (this.truth && s.obs) {
      const diff = countDifferences(this.truth, s.obs);
      const fromScan = s.original ? countDifferences(this.truth, s.original) : diff;
      c.append(
        h(
          'p',
          `status ${diff ? 'is-alert' : 'is-good'}`,
          diff
            ? `Compared with your scramble, ${diff} place${diff === 1 ? '' : 's'} differ (marked ✕). The scan itself had ${fromScan} wrong.`
            : `This matches your scramble exactly${fromScan ? ` (the scan had ${fromScan} wrong; you fixed ${fromScan === 1 ? 'it' : 'them'})` : ': the scan was right'}.`,
        ),
      );
      if (diff) {
        c.append(
          button('Use the scramble’s state', 'btn btn--ghost btn--sm', () =>
            this.edit(() => {
              s.obs!.splice(0, s.obs!.length, ...this.truth!);
            }),
          ),
        );
      }
    }
  }

  private modelCard(): HTMLElement {
    const s = this.ses;
    const c = card('3D model');
    const box = h('div', 'cube-box');
    box.id = 'check-cube';
    c.append(box);
    this.viewSeg = viewButtons(
      [
        { view: 'corner', label: 'Corner', title: PALETTE.say('The orange-yellow-green turning corner towards you, green on top (as on the 3D simulator site)') },
        { view: 'face', label: 'Face-on', title: PALETTE.say('The orange face straight on, green on top') },
        { view: 'edge', label: 'Edge', title: 'The L flip edge towards you' },
      ],
      (view) => s.view.setView(view),
    );
    c.append(this.viewSeg.el);
    const views = h('div', 'view-row');
    for (const f of this.faceOrder()) {
      views.append(
        button(s.source === 'scan' ? `Photo ${this.badgeOf(f)}` : PALETTE.nameOf(f), 'chip-btn', () => {
          this.viewSeg?.set(null);
          s.view.setView(`face-${f}`);
        }),
      );
    }
    c.append(views);
    c.append(h('p', 'hint hint--tight', `Drag to turn the model. ${s.source === 'scan' ? 'The numbers on it are the photos' : 'The letters on it are the faces’ solved colours'}; U D F R L mark the parts the moves turn.`));
    return c;
  }

  private show3D(): void {
    const s = this.ses;
    const view = s.view;
    view.mount(document.getElementById('check-cube')!);
    view.showLabels(true);
    view.setBadges([0, 1, 2, 3, 4, 5].map((f) => this.badgeOf(f)));
    this.viewSeg?.set('corner');
    this.update3D();
    view.setView('corner', false);
  }

  /** The model shows the decoder's state (the closest arrangement of real pieces); dimmed when it is not legal. */
  private update3D(): void {
    const d = this.ses.decode;
    const box = document.getElementById('check-cube');
    box?.classList.toggle('is-stale', !d?.legal);
    if (d?.orient) this.ses.view.show(stateFromOrient(d.orient));
  }

  private renderNext(): void {
    const app = this.app;
    const s = this.ses;
    const c = this.nextEl;
    c.replaceChildren(h('p', 'eyebrow', 'Next'));
    const legal = !!s.decode?.legal && !!s.decode.orient;
    const row = h('div', 'actions');
    const solve = button('Solve this puzzle', 'btn btn--primary', () => app.go('solve'));
    solve.disabled = !legal;
    row.append(solve);
    c.append(row);
    const row2 = h('div', 'actions');
    const copy = button('Copy the state', 'btn btn--ghost btn--sm', () => {
      if (!s.obs) return;
      void navigator.clipboard.writeText(s.obs.join(' ')).then(
        () => app.toast('The state is on the clipboard: 120 numbers, one per sticker position (colour 0-5, 6 for none).', 'good'),
        () => app.toast('Copying was blocked by the browser.', 'error'),
      );
    });
    copy.disabled = !legal;
    row2.append(copy, button('Save the scan (.json)', 'btn btn--ghost btn--sm', () => this.save()));
    if (s.original && s.obs && countDifferences(s.original, s.obs)) {
      row2.append(button('Undo my changes', 'btn btn--ghost btn--sm', () => this.edit(() => s.obs!.splice(0, s.obs!.length, ...s.original!))));
    }
    c.append(row2);
    if (legal && s.decode?.orient) {
      const arr = h('details', 'array-view');
      arr.innerHTML = `<summary>Show the state</summary><p class="hint hint--tight">Per sticker position, in the model's colours (0-5: orange red yellow white green blue; 6 none)${PALETTE.usual ? '' : `; your cube shows them as ${PALETTE.paint.map((k) => PALETTE.names[k]).join(' ')}`}:</p><code>${s.obs!.join(' ')}</code><p class="hint hint--tight">Per piece, its orientation:</p><code>${s.decode.orient.join(' ')}</code>`;
      c.append(arr);
    }
    if (s.source === 'scan' && s.scan) {
      const keep = h('div', 'keep');
      if (s.kept) {
        keep.append(h('p', 'hint hint--tight', 'These photos are kept as labelled real data.'));
      } else {
        const practice = !!this.truth;
        keep.append(
          h(
            'p',
            'hint hint--tight',
            practice
              ? 'Keep these photos with the scramble’s exact state, to measure the scanner and train it on real pictures.'
              : 'Once every face matches your puzzle, keep these photos with the checked state, to train the scanner on real pictures.',
          ),
        );
        const b = button(practice ? 'Keep as labelled data' : 'Keep as labelled data (checked)', 'btn btn--ghost btn--sm', () => void this.keep());
        b.disabled = practice ? false : !legal;
        keep.append(b);
      }
      c.append(keep);
    }
  }

  private edit(change: () => void): void {
    const s = this.ses;
    change();
    s.solution = null;
    s.kept = false;
    s.decode = null;
    this.app.renderSteps();
    this.renderAll();
    document.getElementById('check-cube')?.classList.add('is-stale');
    clearTimeout(this.decodeTimer);
    const seq = ++this.decodeSeq;
    this.decodeTimer = window.setTimeout(async () => {
      try {
        const d = await api.decode(s.obs!);
        if (seq !== this.decodeSeq) return;
        s.decode = d;
        this.renderAll();
        this.update3D();
        this.app.renderSteps();
      } catch (err) {
        this.app.toast(err instanceof ApiError ? err.message : String(err), 'error');
      }
    }, 180);
  }

  private async keep(): Promise<void> {
    const s = this.ses;
    if (!s.scan) return;
    const practice = !!this.truth;
    try {
      const r = await api.confirm(s.scan.id, practice ? this.truth! : s.obs!, practice ? 'known-scramble' : 'checked', PALETTE.paint);
      s.kept = true;
      const errs = typeof r.readErrors === 'number' ? r.readErrors : null;
      this.app.toast(
        `Kept. ${r.scans} labelled real scan${r.scans === 1 ? '' : 's'} so far.${errs !== null ? ` The camera misread ${errs} place${errs === 1 ? '' : 's'} on this one${r.decodedRight ? ', and the decoder fixed all of them' : ''}.` : ''}`,
        'good',
        8000,
      );
      this.renderNext();
      void this.app.refreshHealth();
    } catch (err) {
      this.app.toast(err instanceof ApiError ? err.message : String(err), 'error');
    }
  }

  private save(): void {
    const s = this.ses;
    const data = {
      puzzle: 'tangram-cube-extreme',
      savedAt: new Date().toISOString(),
      source: s.source,
      scanId: s.scan?.id ?? null,
      photos: s.scan?.photos.map((p) => ({ shot: p.shot, name: p.name, face: p.face, colour: p.colour, end: p.end })) ?? null,
      frame: "the core's (the simulator's and the solver's)",
      observation: s.obs,
      // The model's colour c is shown on this cube as colours[c] (orange red yellow white green blue).
      colours: PALETTE.paint,
      orient: s.decode?.orient ?? null,
      legal: s.decode?.legal ?? false,
      practiceScramble: s.practice?.moves.map((m) => (m.axis === 'F' || m.axis === 'R' || m.axis === 'L' ? m.axis : `${m.axis}${Math.abs(m.amount) === 1 ? '' : Math.abs(m.amount)}${m.amount < 0 ? "'" : ''}`)).join(' ') ?? null,
    };
    download(`tangram-scan-${s.scan?.id ?? 'moves'}.json`, JSON.stringify(data, null, 1), 'application/json');
  }
}

const capital = (t: string): string => t[0].toUpperCase() + t.slice(1);
