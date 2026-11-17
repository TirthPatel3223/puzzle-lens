import { COLOUR_NAMES, PALETTE } from './model';

/**
 * One face of the cube as a card: the flattened photo (or a drawing of the face when there is no
 * photo) with a ring on every sticker and gear in the colour it was decoded as. Click a ring to fix it.
 *
 * Face coordinates: x right, y up, the face square is [-1, 1]² (the flattened photos show
 * [-1.15, 1.15]², the same as scanner/vision/geometry.py `rectify`). Cells follow
 * web/src/maltese/observation.ts: 4×4 grid, rows down, columns right.
 */

const HALF = 1.15;
const SVG_NS = 'http://www.w3.org/2000/svg';

/** Where each cell's ring is drawn (face coordinates). Stars and gears sit on the real stickers; the
 * two cells of an edge and the four of the centre are spread a little so each can be clicked. */
const CELL_AT: readonly (readonly [number, number])[] = (() => {
  const at: [number, number][] = [];
  const S = 0.43; // stars
  const E = 0.93; // edge stickers (on the rim)
  const P = 0.085; // half the spacing of two edge cells
  const C = 0.07; // centre cells
  const grid: Record<number, [number, number]> = {
    0: [-S, S], 3: [S, S], 12: [-S, -S], 15: [S, -S],
    1: [-P, E], 2: [P, E], 13: [-P, -E], 14: [P, -E],
    4: [-E, P], 8: [-E, -P], 7: [E, P], 11: [E, -P],
    5: [-C, C], 6: [C, C], 9: [-C, -C], 10: [C, -C],
  };
  for (let i = 0; i < 16; i++) at.push(grid[i]);
  return at;
})();
const STAR_CELLS = new Set([0, 3, 12, 15]);
const CENTRE_CELLS = new Set([5, 6, 9, 10]);
/** Gear hubs: top, right, bottom, left. */
const GEAR_AT: readonly (readonly [number, number])[] = [
  [0, 0.42],
  [0.42, 0],
  [0, -0.42],
  [-0.42, 0],
];
/** Direction from a gear's hub to its long teeth at quarter 0: towards the cube edge. */
const GEAR_OUT: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
];

export function cellKind(c: number): 'star' | 'centre' | 'edge' {
  return STAR_CELLS.has(c) ? 'star' : CENTRE_CELLS.has(c) ? 'centre' : 'edge';
}

/** Long-teeth direction of a gear at `place` turned `quarter` quarter turns counterclockwise. */
function gearDir(place: number, quarter: number): [number, number] {
  let [x, y] = GEAR_OUT[place];
  for (let k = 0; k < (quarter & 3); k++) [x, y] = [-y, x];
  return [x, y];
}

export interface FaceCardModel {
  /** Short title ("Photo 1") and details. */
  readonly title: string;
  readonly fileName?: string;
  /** e.g. "blue face · on top while solving", with that colour's index for the swatch. */
  readonly chip?: { readonly text: string; readonly colour: number };
  /** Flattened photo of the face (data URL), turned the way the cells are laid out. */
  readonly photo?: string;
  /** Badge shown on the 3D model for this face. */
  readonly badge: string;
  readonly cells: readonly number[];
  readonly gearColour: readonly number[];
  readonly gearQuarter: readonly number[];
  /** The scanner was unsure of the value shown. */
  readonly doubt?: readonly boolean[];
  readonly gearDoubt?: readonly boolean[];
  /** The decoder overruled what the camera read. */
  readonly changed?: readonly boolean[];
  readonly gearChanged?: readonly boolean[];
  /** You changed it by hand. */
  readonly edited?: readonly boolean[];
  readonly gearEdited?: readonly boolean[];
  /** Part of a problem that makes the cube impossible. */
  readonly problem?: readonly boolean[];
  readonly gearProblem?: readonly boolean[];
  /** Differs from the known scramble (practice mode). */
  readonly differs?: readonly boolean[];
  readonly gearDiffers?: readonly boolean[];
  /** What the camera read at each place and how sure it was (shown in the editor). */
  readonly readCells?: readonly number[];
  readonly readCellProb?: readonly number[];
  readonly readGearColour?: readonly number[];
  readonly readGearQuarter?: readonly number[];
  readonly readGearProb?: readonly number[];
}

export interface FaceCardHandlers {
  onCell(cell: number, colour: number): void;
  onGear(place: number, colour: number, quarter: number): void;
  onShow3D(): void;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

const svg = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] => {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};

/** A drawing of the face (when there is no photo): black plastic, four stars, four gears, the edges and the centre. */
function drawSchematic(g: SVGGElement, m: FaceCardModel): void {
  g.append(svg('rect', { x: -1.02, y: -1.02, width: 2.04, height: 2.04, rx: 0.12, fill: '#141416' }));
  const star = (cx: number, cy: number) => {
    const pts: string[] = [];
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      const r = i % 2 === 0 ? 0.36 : 0.12;
      pts.push(`${cx + r * Math.cos(a)},${-(cy + r * Math.sin(a))}`);
    }
    return pts.join(' ');
  };
  for (const c of STAR_CELLS) {
    const [x, y] = CELL_AT[c];
    g.append(svg('polygon', { points: star(x, y), fill: PALETTE.hexOf(m.cells[c]), stroke: '#000', 'stroke-width': 0.012 }));
  }
  for (let p = 0; p < 4; p++) {
    // Four petals in an X; which way the long teeth point is shown by the ring's arrow.
    const [x, y] = GEAR_AT[p];
    const petals = svg('g', { transform: `translate(${x} ${-y})` });
    for (let k = 0; k < 4; k++) {
      petals.append(svg('ellipse', { cx: 0.09, cy: 0, rx: 0.12, ry: 0.06, fill: PALETTE.hexOf(m.gearColour[p]), transform: `rotate(${45 + 90 * k})` }));
    }
    g.append(petals);
  }
  for (const c of CENTRE_CELLS) {
    const [x, y] = CELL_AT[c];
    g.append(svg('rect', { x: x - 0.06, y: -y - 0.06, width: 0.12, height: 0.12, fill: PALETTE.hexOf(m.cells[c]) }));
  }
  for (let c = 0; c < 16; c++) {
    if (STAR_CELLS.has(c) || CENTRE_CELLS.has(c)) continue;
    const [x, y] = CELL_AT[c];
    g.append(svg('rect', { x: x - 0.06, y: -y - 0.06, width: 0.12, height: 0.12, rx: 0.02, fill: PALETTE.hexOf(m.cells[c]) }));
  }
}

/** Build the card. Returns the element; re-render by building a new one. */
export function faceCard(m: FaceCardModel, h: FaceCardHandlers): HTMLElement {
  const card = el('article', 'face-card');
  const head = el('header', 'face-card__head');
  const title = el('div', 'face-card__title');
  title.append(el('span', 'face-card__badge', m.badge), el('span', 'face-card__name', m.title));
  if (m.fileName) title.append(el('span', 'face-card__file', m.fileName));
  head.append(title);
  if (m.chip) {
    const chip = el('span', 'face-chip');
    const sw = el('i', 'face-chip__swatch');
    sw.style.background = PALETTE.hexOf(m.chip.colour);
    chip.append(sw, document.createTextNode(m.chip.text));
    head.append(chip);
  }
  card.append(head);

  const pic = el('div', 'face-card__pic');
  const root = svg('svg', { viewBox: `${-HALF} ${-HALF} ${2 * HALF} ${2 * HALF}`, class: 'face-svg', role: 'img' });
  root.setAttribute('aria-label', `${m.title}: the face as read. Select a ring to change it.`);
  if (m.photo) {
    root.append(svg('image', { href: m.photo, x: -HALF, y: -HALF, width: 2 * HALF, height: 2 * HALF, preserveAspectRatio: 'none' }));
  } else {
    root.append(svg('rect', { x: -HALF, y: -HALF, width: 2 * HALF, height: 2 * HALF, class: 'face-svg__bg' }));
    const g = svg('g', {});
    drawSchematic(g, m);
    root.append(g);
  }

  const marks = svg('g', { class: 'face-marks' });
  const flags = (i: number, gear: boolean) => {
    const pick = (a?: readonly boolean[]) => !!a?.[i];
    return gear
      ? { doubt: pick(m.gearDoubt), changed: pick(m.gearChanged), edited: pick(m.gearEdited), problem: pick(m.gearProblem), differs: pick(m.gearDiffers) }
      : { doubt: pick(m.doubt), changed: pick(m.changed), edited: pick(m.edited), problem: pick(m.problem), differs: pick(m.differs) };
  };
  const markClass = (f: ReturnType<typeof flags>) =>
    ['mark', f.problem && 'is-problem', f.differs && 'is-differs', !f.edited && f.doubt && 'is-doubt', f.changed && 'is-changed', f.edited && 'is-edited']
      .filter(Boolean)
      .join(' ');
  const addFlagDot = (g: SVGGElement, x: number, y: number, r: number, f: ReturnType<typeof flags>) => {
    if (f.edited) g.append(svg('circle', { cx: x + r * 0.85, cy: -y - r * 0.85, r: r * 0.38, class: 'mark__dot mark__dot--edited' }));
    else if (f.changed) g.append(svg('circle', { cx: x + r * 0.85, cy: -y - r * 0.85, r: r * 0.38, class: 'mark__dot mark__dot--changed' }));
    if (f.differs) {
      const s = r * 0.7;
      g.append(svg('path', { d: `M${x - s} ${-y - s}L${x + s} ${-y + s}M${x + s} ${-y - s}L${x - s} ${-y + s}`, class: 'mark__x' }));
    }
  };

  for (let c = 0; c < 16; c++) {
    const [x, y] = CELL_AT[c];
    const kind = cellKind(c);
    const r = kind === 'star' ? 0.11 : kind === 'centre' ? 0.05 : 0.055;
    const f = flags(c, false);
    const g = svg('g', { class: markClass(f), tabindex: 0, role: 'button' });
    g.setAttribute('aria-label', `${kind === 'star' ? 'Corner star' : kind === 'centre' ? 'Centre' : 'Edge'}: ${PALETTE.nameOf(m.cells[c])}`);
    g.append(svg('circle', { cx: x, cy: -y, r: r * 1.6, class: 'mark__hit' }));
    g.append(svg('circle', { cx: x, cy: -y, r, class: 'mark__halo' }));
    g.append(svg('circle', { cx: x, cy: -y, r, class: 'mark__ring', stroke: PALETTE.hexOf(m.cells[c]) }));
    addFlagDot(g, x, y, r, f);
    const open = () => openEditor({ kind: 'cell', index: c, x, y });
    g.addEventListener('click', open);
    g.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), open()));
    marks.append(g);
  }
  for (let p = 0; p < 4; p++) {
    const [x, y] = GEAR_AT[p];
    const [dx, dy] = gearDir(p, m.gearQuarter[p]);
    const r = 0.1;
    const f = flags(p, true);
    const g = svg('g', { class: `${markClass(f)} mark--gear`, tabindex: 0, role: 'button' });
    g.setAttribute('aria-label', `Gear: ${PALETTE.nameOf(m.gearColour[p])}, long teeth turned ${m.gearQuarter[p]} quarter turns`);
    g.append(svg('circle', { cx: x, cy: -y, r: r * 1.6, class: 'mark__hit' }));
    g.append(svg('circle', { cx: x, cy: -y, r, class: 'mark__halo' }));
    g.append(svg('circle', { cx: x, cy: -y, r, class: 'mark__ring', stroke: PALETTE.hexOf(m.gearColour[p]) }));
    g.append(svg('path', { d: `M${x + dx * 0.03} ${-(y + dy * 0.03)}L${x + dx * 0.2} ${-(y + dy * 0.2)}`, class: 'mark__arrow' }));
    addFlagDot(g, x, y, r, f);
    const open = () => openEditor({ kind: 'gear', index: p, x, y });
    g.addEventListener('click', open);
    g.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), open()));
    marks.append(g);
  }
  root.append(marks);
  pic.append(root);
  card.append(pic);

  // Foot: what needs looking at, and the button that turns the 3D model to this face.
  const foot = el('footer', 'face-card__foot');
  const count = (a?: readonly boolean[], b?: readonly boolean[]) => (a?.filter(Boolean).length ?? 0) + (b?.filter(Boolean).length ?? 0);
  const notes: string[] = [];
  const nProblem = count(m.problem, m.gearProblem);
  const nDiff = count(m.differs, m.gearDiffers);
  const nDoubt = m.doubt ? m.doubt.filter((d, i) => d && !m.edited?.[i]).length + (m.gearDoubt?.filter((d, i) => d && !m.gearEdited?.[i]).length ?? 0) : 0;
  const nChanged = count(m.changed, m.gearChanged);
  if (nProblem) notes.push(`<b class="t-problem">${nProblem} impossible</b>`);
  if (nDiff) notes.push(`<b class="t-problem">${nDiff} differ from the scramble</b>`);
  if (nDoubt) notes.push(`<b class="t-doubt">${nDoubt} to check</b>`);
  if (nChanged) notes.push(`${nChanged} fixed by the decoder`);
  const note = el('p', 'face-card__note');
  note.innerHTML = notes.length ? notes.join(' · ') : 'Looks sure';
  const show = el('button', 'btn btn--ghost btn--sm', 'Show in 3D');
  show.type = 'button';
  show.addEventListener('click', h.onShow3D);
  foot.append(note, show);
  card.append(foot);

  // ---------------------------------------------------------------- the editor

  let popover: HTMLElement | null = null;
  const close = () => {
    popover?.remove();
    popover = null;
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', onKey, true);
  };
  const outside = (e: Event) => {
    if (popover && !popover.contains(e.target as Node)) close();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') close();
    const n = parseInt(e.key, 10);
    if (popover && n >= 1 && n <= 6) (popover.querySelectorAll<HTMLButtonElement>('.swatch')[n - 1])?.click();
  };

  function openEditor(t: { kind: 'cell' | 'gear'; index: number; x: number; y: number }) {
    close();
    const pop = el('div', 'face-pop');
    pop.setAttribute('role', 'dialog');
    const isGear = t.kind === 'gear';
    const what = isGear ? 'Gear' : cellKind(t.index) === 'star' ? 'Corner star' : cellKind(t.index) === 'centre' ? 'Centre' : 'Edge sticker';
    pop.append(el('p', 'face-pop__title', what));
    if (m.photo) {
      const zoom = el('div', 'face-pop__zoom');
      const Z = 3.2;
      const S = 120;
      const W = S * Z * (2 * HALF) / 2; // the photo is drawn S*Z/2 px per face unit
      const px = ((t.x + HALF) / (2 * HALF)) * W;
      const py = ((HALF - t.y) / (2 * HALF)) * W;
      zoom.style.backgroundImage = `url("${m.photo}")`;
      zoom.style.backgroundSize = `${W}px ${W}px`;
      zoom.style.backgroundPosition = `${-(px - S / 2)}px ${-(py - S / 2)}px`;
      pop.append(zoom);
    }
    const current = isGear ? m.gearColour[t.index] : m.cells[t.index];
    const readC = isGear ? m.readGearColour?.[t.index] : m.readCells?.[t.index];
    const readP = isGear ? m.readGearProb?.[t.index] : m.readCellProb?.[t.index];
    if (readC !== undefined && readP !== undefined) {
      const info = el('p', 'face-pop__info');
      info.textContent = `Camera read: ${PALETTE.nameOf(readC)} (${Math.round(readP * 100)}% sure)`;
      if (readC !== current) info.textContent += ` · decoded as ${PALETTE.nameOf(current)}`;
      pop.append(info);
    }
    const sw = el('div', 'swatches');
    COLOUR_NAMES.forEach((_, colour) => {
      const name = PALETTE.nameOf(colour);                  // the swatches are the cube's own colours
      const b = el('button', `swatch${colour === current ? ' is-on' : ''}`);
      b.type = 'button';
      b.title = `${name} (${colour + 1})`;
      b.setAttribute('aria-label', name);
      b.style.background = PALETTE.hexOf(colour);
      b.addEventListener('click', () => {
        close();
        if (isGear) h.onGear(t.index, colour, m.gearQuarter[t.index]);
        else h.onCell(t.index, colour);
      });
      sw.append(b);
    });
    pop.append(sw);
    if (isGear) {
      pop.append(el('p', 'face-pop__label', 'Long teeth point'));
      const qs = el('div', 'quarters');
      for (let q = 0; q < 4; q++) {
        const [dx, dy] = gearDir(t.index, q);
        const b = el('button', `quarter${q === m.gearQuarter[t.index] ? ' is-on' : ''}`);
        b.type = 'button';
        b.title = `${q} quarter turn${q === 1 ? '' : 's'} from flush`;
        const angle = (Math.atan2(-dy, dx) * 180) / Math.PI;
        b.innerHTML = `<svg viewBox="-1 -1 2 2" width="22" height="22" aria-hidden="true"><path d="M-0.6 0H0.6M0.25 -0.35L0.6 0L0.25 0.35" fill="none" stroke="currentColor" stroke-width="0.22" stroke-linecap="round" stroke-linejoin="round" transform="rotate(${angle})"/></svg><span>${q}</span>`;
        b.addEventListener('click', () => {
          close();
          h.onGear(t.index, m.gearColour[t.index], q);
        });
        qs.append(b);
      }
      pop.append(qs);
    }
    // Place it below (or above) the ring, kept inside the card horizontally.
    const width = pic.clientWidth;
    const popWidth = 228;
    const x = ((t.x + HALF) / (2 * HALF)) * width;
    const top = ((HALF - t.y) / (2 * HALF)) * 100;
    pop.style.left = `${Math.max(Math.min(x - popWidth / 2, width - popWidth), Math.min(0, width - popWidth) / 2)}px`;
    pop.style.top = `${top}%`;
    pop.classList.add(top > 50 ? 'is-above' : 'is-below');
    pic.append(pop);
    popover = pop;
    setTimeout(() => {
      document.addEventListener('pointerdown', outside, true);
      document.addEventListener('keydown', onKey, true);
    });
    (pop.querySelector('.swatch.is-on') as HTMLElement | null)?.focus();
  }

  return card;
}
