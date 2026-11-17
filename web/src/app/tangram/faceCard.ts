import { COLOUR_NAMES, EMPTY, FACE_POSITIONS, PALETTE, colourWord } from './model';

/**
 * One face of the Tangram Cube Extreme as a card: the flattened photo (its turning corner at the
 * bottom-right), or a drawing of the face when there is no photo, with a marker on each of the 29
 * positions the face's photo shows. A circle is a sticker flat on the face; a diamond is one that a 60°
 * turn has tilted out of it (it stands out, at 48°). A ring in a colour is a sticker; a small grey dot
 * is no sticker. Select a marker to fix it.
 *
 * Face units: x right, y up; the face is [-1, 1]², the picture shows [-HALF, HALF]² (the same as the
 * server's flattened photos, scanner/puzzles/tangram/api.py FLAT_HALF).
 */

const HALF = 1.5;
const SVG_NS = 'http://www.w3.org/2000/svg';

export interface MarkModel {
  readonly position: number;
  /** Where the marker goes, face units (x right, y up). */
  readonly x: number;
  readonly y: number;
  readonly flat: boolean;
  /** Its current value: a colour 0-5 or EMPTY. */
  readonly value: number;
  readonly doubt?: boolean;
  readonly changed?: boolean;
  readonly edited?: boolean;
  readonly problem?: boolean;
  readonly differs?: boolean;
  /** What the camera read (0-5, 6 none, 7 hidden) and how sure it was. */
  readonly read?: number;
  readonly readProb?: number;
}

export interface TangramFaceModel {
  readonly title: string;
  readonly fileName?: string;
  readonly chip?: { readonly text: string; readonly colour: number };
  readonly photo?: string;
  readonly badge: string;
  /** FACE_IDS index. */
  readonly face: number;
  /** The whole observation (for the drawing when there is no photo). */
  readonly observation: readonly number[];
  readonly marks: readonly MarkModel[];
}

export interface TangramFaceHandlers {
  onSet(position: number, value: number): void;
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

const readWord = (r: number): string => (r === 7 ? 'hidden (something in front)' : colourWord(r === 6 ? EMPTY : r));

/** A straight-on drawing of the face: every sticker the face's photo would show, back to front. */
function drawFace(g: SVGGElement, face: number, observation: readonly number[]): void {
  g.append(svg('rect', { x: -1.02, y: -1.02, width: 2.04, height: 2.04, rx: 0.06, fill: '#141416' }));
  for (const p of FACE_POSITIONS[face]) {
    const c = observation[p.position];
    if (c === EMPTY) continue;
    const pts = p.outline.map(([x, y]) => `${x},${-y}`).join(' ');
    g.append(
      svg('polygon', {
        points: pts,
        fill: PALETTE.hexOf(c),
        'fill-opacity': p.flat ? 1 : 0.78,
        stroke: '#000',
        'stroke-width': p.flat ? 0.03 : 0.022,
        'stroke-dasharray': p.flat ? 'none' : '0.05 0.03',
        'stroke-linejoin': 'round',
      }),
    );
  }
}

/** The marker's shape: a circle for a flat position, a diamond for a turned-out one. */
function shape(m: MarkModel, r: number, cls: string, extra: Record<string, string | number> = {}): SVGElement {
  if (m.flat) return svg('circle', { cx: m.x, cy: -m.y, r, class: cls, ...extra });
  const d = r * 1.3;
  return svg('polygon', { points: `${m.x},${-m.y - d} ${m.x + d},${-m.y} ${m.x},${-m.y + d} ${m.x - d},${-m.y}`, class: cls, ...extra });
}

export function tangramFaceCard(m: TangramFaceModel, h: TangramFaceHandlers): HTMLElement {
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
  root.setAttribute('aria-label', `${m.title}: the face as read. Select a marker to change it.`);
  if (m.photo) {
    root.append(svg('image', { href: m.photo, x: -HALF, y: -HALF, width: 2 * HALF, height: 2 * HALF, preserveAspectRatio: 'none' }));
  } else {
    root.append(svg('rect', { x: -HALF, y: -HALF, width: 2 * HALF, height: 2 * HALF, class: 'face-svg__bg' }));
    const g = svg('g', {});
    drawFace(g, m.face, m.observation);
    root.append(g);
  }
  // The turning corner is at the bottom-right of every face.
  root.append(svg('circle', { cx: 1, cy: 1, r: 0.05, class: 'turning-corner' }));

  const marks = svg('g', { class: 'face-marks' });
  const order = [...m.marks].sort((a, b) => Number(a.value !== EMPTY) - Number(b.value !== EMPTY));
  for (const mk of order) {
    const empty = mk.value === EMPTY;
    const r = empty ? 0.035 : 0.075;
    const cls = ['mark', empty && 'mark--empty', !mk.flat && 'mark--out', mk.problem && 'is-problem', mk.differs && 'is-differs', !mk.edited && mk.doubt && 'is-doubt', mk.changed && 'is-changed', mk.edited && 'is-edited']
      .filter(Boolean)
      .join(' ');
    const g = svg('g', { class: cls, tabindex: 0, role: 'button' });
    g.setAttribute('aria-label', `${mk.flat ? 'Flat sticker' : 'Turned-out sticker'}: ${colourWord(mk.value)}`);
    g.append(shape(mk, Math.max(r, 0.06) * 1.5, 'mark__hit'));
    if (empty) {
      g.append(shape(mk, r, 'mark__none'));
    } else {
      g.append(shape(mk, r, 'mark__halo'));
      g.append(shape(mk, r, 'mark__ring', { stroke: PALETTE.hexOf(mk.value) }));
    }
    const dot = (c: string) => g.append(svg('circle', { cx: mk.x + r * 0.85, cy: -mk.y - r * 0.85, r: 0.028, class: `mark__dot ${c}` }));
    if (mk.edited) dot('mark__dot--edited');
    else if (mk.changed) dot('mark__dot--changed');
    if (mk.differs) {
      const s = r * 0.7;
      g.append(svg('path', { d: `M${mk.x - s} ${-mk.y - s}L${mk.x + s} ${-mk.y + s}M${mk.x + s} ${-mk.y - s}L${mk.x - s} ${-mk.y + s}`, class: 'mark__x' }));
    }
    const open = () => openEditor(mk);
    g.addEventListener('click', open);
    g.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), open()));
    marks.append(g);
  }
  root.append(marks);
  pic.append(root);
  card.append(pic);

  const foot = el('footer', 'face-card__foot');
  const notes: string[] = [];
  const n = (f: (x: MarkModel) => boolean | undefined) => m.marks.filter(f).length;
  const nProblem = n((x) => x.problem);
  const nDiff = n((x) => x.differs);
  const nDoubt = n((x) => x.doubt && !x.edited);
  const nChanged = n((x) => x.changed);
  const nOut = n((x) => !x.flat && x.value !== EMPTY);
  if (nProblem) notes.push(`<b class="t-problem">${nProblem} impossible</b>`);
  if (nDiff) notes.push(`<b class="t-problem">${nDiff} differ from the scramble</b>`);
  if (nDoubt) notes.push(`<b class="t-doubt">${nDoubt} to check</b>`);
  if (nChanged) notes.push(`${nChanged} fixed by the decoder`);
  if (nOut) notes.push(`${nOut} turned out`);
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
    if (!popover) return;
    const k = parseInt(e.key, 10);
    if (k >= 1 && k <= 6) popover.querySelectorAll<HTMLButtonElement>('.swatch')[k - 1]?.click();
    if (e.key === '0' || e.key.toLowerCase() === 'n') popover.querySelector<HTMLButtonElement>('.swatch-none')?.click();
  };

  function openEditor(mk: MarkModel) {
    close();
    const pop = el('div', 'face-pop');
    pop.setAttribute('role', 'dialog');
    pop.append(el('p', 'face-pop__title', mk.flat ? 'Sticker flat on this face' : 'Sticker turned out of this face'));
    if (m.photo) {
      const zoom = el('div', 'face-pop__zoom');
      const Z = 2.6;
      const S = 120;
      const W = (S * Z * (2 * HALF)) / 2;
      const px = ((mk.x + HALF) / (2 * HALF)) * W;
      const py = ((HALF - mk.y) / (2 * HALF)) * W;
      zoom.style.backgroundImage = `url("${m.photo}")`;
      zoom.style.backgroundSize = `${W}px ${W}px`;
      zoom.style.backgroundPosition = `${-(px - S / 2)}px ${-(py - S / 2)}px`;
      pop.append(zoom);
    }
    if (mk.read !== undefined && mk.readProb !== undefined) {
      const info = el('p', 'face-pop__info');
      info.textContent = `Camera read: ${readWord(mk.read)} (${Math.round(mk.readProb * 100)}% sure)`;
      if ((mk.read === 6 ? EMPTY : mk.read) !== mk.value && mk.read !== 7) info.textContent += ` · decoded as ${colourWord(mk.value)}`;
      pop.append(info);
    }
    if (!mk.flat) pop.append(el('p', 'face-pop__label', 'It is also in the photo of the face it leans towards.'));
    const sw = el('div', 'swatches swatches--7');
    COLOUR_NAMES.forEach((_, colour) => {
      const name = PALETTE.nameOf(colour);                  // the swatches are the cube's own colours
      const b = el('button', `swatch${colour === mk.value ? ' is-on' : ''}`);
      b.type = 'button';
      b.title = `${name} (${colour + 1})`;
      b.setAttribute('aria-label', name);
      b.style.background = PALETTE.hexOf(colour);
      b.addEventListener('click', () => {
        close();
        h.onSet(mk.position, colour);
      });
      sw.append(b);
    });
    const none = el('button', `swatch swatch-none${mk.value === EMPTY ? ' is-on' : ''}`, '∅');
    none.type = 'button';
    none.title = 'No sticker here (0)';
    none.setAttribute('aria-label', 'no sticker here');
    none.addEventListener('click', () => {
      close();
      h.onSet(mk.position, EMPTY);
    });
    sw.append(none);
    pop.append(sw);
    const width = pic.clientWidth;
    const popWidth = 248;
    const x = ((mk.x + HALF) / (2 * HALF)) * width;
    const top = ((HALF - mk.y) / (2 * HALF)) * 100;
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

/** Markers for a face with no photo: where a straight-on drawing shows each position. */
export function drawnMarks(face: number): { position: number; x: number; y: number; flat: boolean }[] {
  return FACE_POSITIONS[face].map((p) => ({ position: p.position, x: p.x, y: p.y, flat: p.flat }));
}

export const FACE_HALF = HALF;
