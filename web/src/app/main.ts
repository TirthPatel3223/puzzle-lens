import '@fontsource-variable/plus-jakarta-sans';
import '../vendor/styles.css';
import './app.css';
import { App } from './app';
import { PhotosScreen } from './photos';
import { PUZZLES } from './puzzles';
import { setupPanelResizer } from './resizer';

// Light / dark theme, as on the 3D simulator site (the inline script in index.html sets it before paint).
const THEME_KEY = 'theme-choice';
const themeBtn = document.getElementById('theme-toggle')!;
const themeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
function applyTheme(t: 'light' | 'dark', save: boolean) {
  document.documentElement.dataset.theme = t;
  themeBtn.setAttribute('aria-label', t === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  themeMeta?.setAttribute('content', t === 'dark' ? '#0f1419' : '#f8f9fc');
  if (save) {
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch {
      /* private mode: just don't remember it */
    }
  }
}
applyTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light', false);
themeBtn.addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true));

setupPanelResizer(document.getElementById('app')!, document.getElementById('panel-resizer')!);

const app = new App(PUZZLES(), new PhotosScreen());
app.go('photos');

// Warm the server up: the puzzle's vision networks load in the background so the first scan is quick.
void app.refreshHealth().then((h) => {
  if (!h) {
    app.toast('The scanner server is not answering. Start it in the Cube_Scanner folder with:  python -m scanner serve', 'error', 15000);
    return;
  }
  if (!app.puzzleHealth()?.vision.loaded) void app.puzzle.photos.loadVision().catch(() => {});
});
setInterval(() => void app.refreshHealth(), 15000);

// Handy for debugging in the browser console.
Object.assign(window as unknown as Record<string, unknown>, { scanner: app });
