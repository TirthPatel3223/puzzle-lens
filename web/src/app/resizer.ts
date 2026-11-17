/**
 * The side panel's width, set by dragging the bar between the panel and the work area (or with the
 * arrow keys on it; double-click puts it back). The width is the `--panel-w` grid column of #app,
 * remembered in this browser. On narrow screens the panel is below the work area and the bar is hidden.
 */

const KEY = 'scanner-panel-width';
export const PANEL_DEFAULT = 410;
const PANEL_MIN = 320;
const PANEL_MAX = 900;
/** The work area keeps at least this much room. */
const WORK_MIN = 420;
const KEY_STEP = 16;

export function setupPanelResizer(appEl: HTMLElement, handle: HTMLElement): void {
  /** The width asked for; what is shown is this clamped to the window. */
  let wanted = PANEL_DEFAULT;
  try {
    const saved = Number(localStorage.getItem(KEY));
    if (Number.isFinite(saved) && saved > 0) wanted = saved;
  } catch {
    /* storage blocked: start at the default */
  }

  const limits = () => ({ min: PANEL_MIN, max: Math.max(PANEL_MIN, Math.min(PANEL_MAX, window.innerWidth - WORK_MIN)) });
  const shown = () => {
    const { min, max } = limits();
    return Math.round(Math.min(max, Math.max(min, wanted)));
  };
  const apply = () => {
    const w = shown();
    const { min, max } = limits();
    appEl.style.setProperty('--panel-w', `${w}px`);
    handle.setAttribute('aria-valuemin', String(min));
    handle.setAttribute('aria-valuemax', String(max));
    handle.setAttribute('aria-valuenow', String(w));
  };
  const save = () => {
    try {
      localStorage.setItem(KEY, String(shown()));
    } catch {
      /* not remembered */
    }
  };
  const set = (w: number) => {
    wanted = w;
    apply();
  };

  let drag: { id: number; x0: number; w0: number } | null = null;
  handle.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    drag = { id: e.pointerId, x0: e.clientX, w0: shown() };
    handle.setPointerCapture(e.pointerId);
    handle.classList.add('is-dragging');
    document.documentElement.classList.add('is-resizing');
  });
  handle.addEventListener('pointermove', (e) => {
    if (drag?.id === e.pointerId) set(drag.w0 + (e.clientX - drag.x0));
  });
  const end = (e: PointerEvent) => {
    if (drag?.id !== e.pointerId) return;
    drag = null;
    handle.classList.remove('is-dragging');
    document.documentElement.classList.remove('is-resizing');
    // Keep what is shown, so a drag past the limit does not leave a hidden "wanted" width behind.
    wanted = shown();
    save();
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
  handle.addEventListener('dblclick', () => {
    set(PANEL_DEFAULT);
    save();
  });
  handle.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 4 * KEY_STEP : KEY_STEP;
    const w = shown();
    const next =
      e.key === 'ArrowLeft' ? w - step : e.key === 'ArrowRight' ? w + step : e.key === 'Home' ? limits().min : e.key === 'End' ? limits().max : null;
    if (next === null) return;
    e.preventDefault();
    set(next);
    save();
  });
  // The window changing size changes the limits (the wanted width comes back when there is room again).
  window.addEventListener('resize', apply);
  apply();
}
