import { button, escapeHtml, h } from './app';
import { type ColoursInfo, describeLayout } from './palette';

/**
 * The Check step's line about the cube's colour layout (palette.ts): shown when a scan found the cube
 * stickered differently from the usual, or when the scan is doubtful and another layout came close.
 * `pick` reads the photos again with the cube taken to have the chosen layout.
 */
export function coloursNote(
  info: ColoursInfo | undefined,
  alternatives: readonly ColoursInfo[] | undefined,
  doubtful: boolean,
  pick: (paint: readonly number[]) => Promise<void>,
  failed: (err: unknown) => void,
): HTMLElement | null {
  const alts = alternatives ?? [];
  if (!info || (info.usual && !(doubtful && alts.length))) return null;
  const box = h('div', 'colours-note');
  const opts = h('div', 'colours-note__options');
  for (const a of alts) {
    const label = a.usual ? 'The usual colours' : capital(describeLayout(a));
    opts.append(
      button(label, 'chip-btn', () => {
        opts.querySelectorAll('button').forEach((b) => ((b as HTMLButtonElement).disabled = true));
        const wait = h('p', 'hint hint--tight', 'Reading the photos again…');
        box.append(wait);
        pick(a.paint).catch((err) => {
          wait.remove();
          opts.querySelectorAll('button').forEach((b) => ((b as HTMLButtonElement).disabled = false));
          failed(err);
        });
      }),
    );
  }
  if (!info.usual) {
    box.append(
      h('p', 'status is-note', `Your cube is stickered differently from the usual: ${escapeHtml(describeLayout(info))}. The model and the faces here are shown in your cube's colours.`),
    );
    if (alts.length) {
      opts.hidden = true;
      box.append(
        button('Not right? Change the colours', 'btn btn--ghost btn--sm', () => {
          opts.hidden = !opts.hidden;
        }),
        opts,
      );
    }
  } else {
    box.append(h('p', 'hint hint--tight', 'If your cube is stickered differently from the usual colours, the photos can be read that way:'), opts);
  }
  return box;
}

const capital = (t: string): string => t[0].toUpperCase() + t.slice(1);
