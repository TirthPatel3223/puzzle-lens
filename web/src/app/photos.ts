import { ApiError, type Checked } from './api';
import { type App, type PhotoProtocol, type Screen, type Shot, button, card, escapeHtml, h, newShotId } from './app';
import { type FrameRect, HoldTimer, LIVE, type Verdict, judge } from './liveCheck';

const BROWSER_IMAGE = /^image\/(jpeg|png|webp|gif|bmp|avif)$/;
/** Live checks: the frame sent is shrunk to this longer side, at most this often. */
const CHECK_SIDE = 1280;
const CHECK_EVERY_MS = 240;
/** A live check older than this says nothing about the picture now. */
const CHECK_FRESH_MS = 800;
/** After a photo is taken, the live check pauses this long (the outline says the photo was taken). */
const TAKEN_PAUSE_MS = 1300;
const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Step 1, shared by every puzzle: the photos (upload or webcam, with the live check and auto-capture).
 * The puzzle's PhotoProtocol says how many, what to tell the person, and how to read them; its
 * `photoCards` add other ways in (a practice scramble, a typed state).
 */
export class PhotosScreen implements Screen {
  private app!: App;
  private mode: 'upload' | 'webcam' = 'upload';
  private slotsEl!: HTMLElement;
  private readBtn!: HTMLButtonElement;
  private countEl!: HTMLElement;
  private fileInput!: HTMLInputElement;
  private stream: MediaStream | null = null;
  /** Mirror the live picture? null = decided by the camera (a camera facing you is shown mirrored). */
  private mirrorChoice: boolean | null = null;
  private mirrorBtn: HTMLButtonElement | null = null;
  private focusTimer = 0;
  /** Bumped whenever the webcam is opened or closed: work for an older webcam session stops. */
  private camGen = 0;
  /** Take the photo by itself after the outline has been green for LIVE.holdMs. */
  private auto = true;
  private readonly hold = new HoldTimer();
  private lastCheck: { at: number; checked: Checked; verdict: Verdict } | null = null;
  /** Shown instead of the live verdict (the server cannot check, six photos taken, …). */
  private liveNote: string | null = null;
  private takenUntil = 0;
  private takenNumber = 0;
  private busy = false;

  private get protocol(): PhotoProtocol {
    return this.app.puzzle.photos;
  }

  enter(app: App): void {
    this.app = app;
    this.buildWork();
    this.buildPanel();
    this.renderSlots();
    void app.refreshHealth();
  }

  leave(): void {
    this.stopWebcam();
  }

  // ------------------------------------------------------------------ work area

  private buildWork(): void {
    const { work } = this.app;
    const wrap = h('div', 'photos');
    const top = h('div', 'photos__top');
    const seg = h('div', 'seg seg--sm');
    seg.setAttribute('role', 'tablist');
    const tab = (id: 'upload' | 'webcam', label: string) => {
      const b = button(label, id === this.mode ? 'is-on' : '', () => {
        if (this.mode === id) return;
        this.mode = id;
        seg.querySelectorAll('button').forEach((x) => x.classList.toggle('is-on', x === b));
        this.showMode();
      });
      b.setAttribute('role', 'tab');
      return b;
    };
    seg.append(tab('upload', 'Upload photos'), tab('webcam', 'Use the webcam'));
    top.append(seg, h('p', 'photos__hint', this.protocol.hint));
    wrap.append(top);

    const cam = h('div', 'webcam');
    cam.id = 'webcam';
    cam.hidden = true;
    wrap.append(cam);

    this.slotsEl = h('div', 'slots');
    wrap.append(this.slotsEl);

    this.fileInput = document.createElement('input');
    this.fileInput.type = 'file';
    this.fileInput.multiple = true;
    this.fileInput.accept = 'image/*,.heic,.heif';
    this.fileInput.hidden = true;
    this.fileInput.addEventListener('change', () => {
      this.addFiles(Array.from(this.fileInput.files ?? []));
      this.fileInput.value = '';
    });
    wrap.append(this.fileInput);

    // Drop anywhere on the work area.
    wrap.addEventListener('dragover', (e) => {
      e.preventDefault();
      wrap.classList.add('is-drop');
    });
    wrap.addEventListener('dragleave', (e) => {
      if (e.target === wrap) wrap.classList.remove('is-drop');
    });
    wrap.addEventListener('drop', (e) => {
      e.preventDefault();
      wrap.classList.remove('is-drop');
      this.addFiles(Array.from(e.dataTransfer?.files ?? []));
    });
    work.append(wrap);
    this.showMode();
  }

  private showMode(): void {
    const cam = document.getElementById('webcam')!;
    if (this.mode === 'webcam') {
      cam.hidden = false;
      void this.startWebcam(cam);
    } else {
      cam.hidden = true;
      this.stopWebcam();
    }
  }

  private renderSlots(): void {
    const shots = this.app.shots;
    this.slotsEl.replaceChildren();
    const markCorner = this.protocol.markCorner;
    for (let i = 0; i < this.protocol.count; i++) {
      const s = shots[i];
      const slot = h('figure', `slot${s ? ` slot--${s.status}` : ' slot--empty'}`);
      if (!s) {
        const add = button(
          `<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg><span>Photo ${i + 1}</span>`,
          'slot__add',
          () => (this.mode === 'webcam' ? this.capture() : this.fileInput.click()),
          this.mode === 'webcam' ? 'Capture from the webcam' : 'Choose photos',
        );
        slot.append(add);
      } else {
        // A square tile; the photo is letterboxed in it (object-fit: contain) and the face outline is
        // drawn in the photo's own pixels with the same letterboxing (preserveAspectRatio meet).
        const pic = h('div', 'slot__pic');
        const src = s.preview ?? s.localUrl;
        if (src) {
          const img = document.createElement('img');
          img.src = src;
          img.alt = s.name;
          pic.append(img);
        } else {
          pic.append(h('span', 'slot__placeholder', 'HEIC photo<br>preview coming…'));
        }
        if (s.quad && s.width && s.height) {
          const ns = 'http://www.w3.org/2000/svg';
          const svg = document.createElementNS(ns, 'svg');
          svg.setAttribute('viewBox', `0 0 ${s.width} ${s.height}`);
          svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
          svg.classList.add('slot__quad');
          const poly = document.createElementNS(ns, 'polygon');
          poly.setAttribute('points', s.quad.map(([x, y]) => `${x * s.width!},${y * s.height!}`).join(' '));
          poly.setAttribute('vector-effect', 'non-scaling-stroke');
          svg.append(poly);
          // One corner marked (the puzzle says which: the top-left as the photo shows it, or the turning corner).
          const dot = document.createElementNS(ns, 'circle');
          dot.setAttribute('cx', String(s.quad[markCorner][0] * s.width));
          dot.setAttribute('cy', String(s.quad[markCorner][1] * s.height));
          dot.setAttribute('r', String(Math.max(s.width, s.height) * 0.014));
          const tip = document.createElementNS(ns, 'title');
          tip.textContent = this.protocol.markTitle;
          dot.append(tip);
          svg.append(dot);
          pic.append(svg);
        }
        if (s.status === 'locating') pic.append(h('span', 'slot__spinner', ''));
        slot.append(pic);
        const cap = h('figcaption', 'slot__cap');
        const status =
          s.status === 'locating'
            ? 'finding the face…'
            : s.status === 'error'
              ? `<span class="t-problem">${s.error ?? 'could not be read'}</span>`
              : (s.confidence ?? 0) < 0.45
                ? '<span class="t-doubt">face found, but unsure</span>'
                : 'face found';
        cap.innerHTML = `<span class="slot__n">${i + 1}</span><span class="slot__name" title="${escapeHtml(s.name)}">${escapeHtml(s.name)}</span><span class="slot__status">${status}</span>`;
        slot.append(cap);
        slot.append(
          button(
            '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
            'slot__remove',
            () => this.removeShot(s.id),
            'Remove this photo',
          ),
        );
      }
      this.slotsEl.append(slot);
    }
    this.updateReadButton();
  }

  private updateReadButton(): void {
    if (!this.readBtn) return;
    const n = this.app.shots.length;
    const want = this.protocol.count;
    const ready = n === want && this.app.shots.every((s) => s.status === 'ready');
    this.readBtn.disabled = !ready || this.busy;
    this.countEl.textContent = `${n} of ${want} photos`;
  }

  private addFiles(files: File[]): void {
    const room = this.protocol.count - this.app.shots.length;
    if (!files.length) return;
    if (room <= 0) {
      this.app.toast(`There are ${this.protocol.count} photos already. Remove one to add another.`, 'info');
      return;
    }
    if (files.length > room) this.app.toast(`Only ${room} more photo${room === 1 ? '' : 's'} fit; the rest were left out.`, 'info');
    for (const f of files.slice(0, room)) this.addShot(f, f.name);
  }

  private addShot(blob: Blob, name: string, signature?: unknown): void {
    const shot: Shot = {
      id: newShotId(),
      blob,
      name,
      localUrl: BROWSER_IMAGE.test(blob.type) ? URL.createObjectURL(blob) : null,
      status: 'locating',
      signature,
    };
    const puzzle = this.app.puzzle;
    this.app.shots.push(shot);
    this.renderSlots();
    this.protocol
      .locate(blob, name)
      .then((r) => {
        Object.assign(shot, { status: 'ready', preview: r.preview, width: r.width, height: r.height, quad: r.quad, confidence: r.confidence });
      })
      .catch((err: unknown) => {
        shot.status = 'error';
        shot.error = err instanceof ApiError ? err.message : 'could not be read';
        if (err instanceof ApiError && err.status === 0) this.app.toast(err.message, 'error', 9000);
      })
      .finally(() => {
        if (this.app.shots.includes(shot) && this.app.step === 'photos' && this.app.puzzle === puzzle) this.renderSlots();
      });
  }

  private removeShot(id: number): void {
    const s = this.app.shots.find((x) => x.id === id);
    if (s?.localUrl) URL.revokeObjectURL(s.localUrl);
    this.app.shots = this.app.shots.filter((x) => x.id !== id);
    this.renderSlots();
  }

  // ------------------------------------------------------------------ webcam

  private async startWebcam(box: HTMLElement): Promise<void> {
    if (this.stream) return;
    const gen = ++this.camGen;
    box.replaceChildren();
    const view = h('div', 'webcam__view');
    const video = document.createElement('video');
    video.playsInline = true;
    video.muted = true;
    video.autoplay = true;
    // The face the networks found, drawn over the picture (flipped with it when it is mirrored).
    const face = document.createElementNS(SVG_NS, 'svg');
    face.classList.add('webcam__face');
    face.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    face.setAttribute('aria-hidden', 'true');
    const poly = document.createElementNS(SVG_NS, 'polygon');
    poly.setAttribute('vector-effect', 'non-scaling-stroke');
    face.append(poly);
    // The guide square: white until the first check, then red or green; the bar fills while held green.
    const guide = h('div', 'webcam__guide');
    guide.append(h('span', 'webcam__guide-text', 'Fill the square with one face'), h('span', 'webcam__hold'));
    guide.setAttribute('role', 'status');
    guide.setAttribute('aria-live', 'polite');
    const focus = h('div', 'webcam__focus', 'Focus');
    view.append(video, face, guide, focus);
    const bar = h('div', 'webcam__bar');
    const select = document.createElement('select');
    select.className = 'webcam__select';
    select.title = 'Camera';
    const shoot = button('Capture <kbd>Space</kbd>', 'btn btn--pill', () => this.capture());
    this.mirrorBtn = button(
      '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v18M8 7l-5 5 5 5V7zM16 7l5 5-5 5V7z"/></svg> Mirror',
      'btn btn--ghost webcam__toggle',
      () => {
        this.mirrorChoice = !video.classList.contains('is-mirrored');
        this.applyMirror(video);
      },
      'Flip the live picture left to right (the photos themselves are never flipped)',
    );
    const autoBtn = button(
      '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M9 2h6"/></svg> Auto-capture',
      'btn btn--ghost webcam__toggle',
      () => {
        this.auto = !this.auto;
        this.hold.reset();
        showAuto();
      },
      `Take the photo by itself once the outline has been green for ${LIVE.holdMs / 1000} seconds`,
    );
    const showAuto = () => {
      autoBtn.classList.toggle('is-on', this.auto);
      autoBtn.setAttribute('aria-pressed', String(this.auto));
    };
    showAuto();
    bar.append(select, this.mirrorBtn, autoBtn, shoot);
    box.append(view, bar);
    try {
      await this.openCamera(video, undefined);
      if (gen !== this.camGen) {
        // Closed while the camera was opening.
        (this.stream as MediaStream | null)?.getTracks().forEach((t) => t.stop()); // set by openCamera
        this.stream = null;
        return;
      }
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
      select.hidden = cams.length < 2;
      const opened = this.stream as MediaStream | null; // set by openCamera
      const active = opened?.getVideoTracks()[0]?.getSettings().deviceId;
      for (const c of cams) {
        const o = document.createElement('option');
        o.value = c.deviceId;
        o.textContent = c.label || `Camera ${select.length + 1}`;
        o.selected = c.deviceId === active;
        select.append(o);
      }
      select.addEventListener('change', () => void this.openCamera(video, select.value));
    } catch (err) {
      box.replaceChildren(h('p', 'webcam__error', `The webcam could not be opened (${err instanceof Error ? err.message : String(err)}). Allow camera access for this page, or upload photos instead.`));
      return;
    }
    document.addEventListener('keydown', this.onKey);
    // A rough sharpness meter: the energy of a Laplacian over the middle of the picture.
    const small = document.createElement('canvas');
    small.width = small.height = 128;
    const sctx = small.getContext('2d', { willReadFrequently: true })!;
    this.focusTimer = window.setInterval(() => {
      if (!video.videoWidth) return;
      const s = Math.min(video.videoWidth, video.videoHeight) * 0.5;
      sctx.drawImage(video, (video.videoWidth - s) / 2, (video.videoHeight - s) / 2, s, s, 0, 0, 128, 128);
      const d = sctx.getImageData(0, 0, 128, 128).data;
      const g = new Float32Array(128 * 128);
      for (let i = 0; i < g.length; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
      let sum = 0;
      let sq = 0;
      let n = 0;
      for (let y = 1; y < 127; y++) {
        for (let x = 1; x < 127; x++) {
          const i = y * 128 + x;
          const l = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - 128] - g[i + 128];
          sum += l;
          sq += l * l;
          n++;
        }
      }
      const v = sq / n - (sum / n) ** 2;
      const sharp = v > 140;
      focus.textContent = sharp ? 'Sharp' : 'Blurry: hold still, more light';
      focus.classList.toggle('is-good', sharp);
    }, 350);
    this.hold.reset();
    this.lastCheck = null;
    this.liveNote = null;
    void this.liveLoop(video, guide, gen);
    const tick = () => {
      if (gen !== this.camGen) return;
      this.renderLive(video, face, poly, guide);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ live check and auto-capture

  /** About four times a second: a small copy of the frame goes to the networks; the verdict colours the guide. */
  private async liveLoop(video: HTMLVideoElement, guide: HTMLElement, gen: number): Promise<void> {
    const canvas = document.createElement('canvas');
    let previous: number[][] | null = null;
    /** Checks failed in a row: one is retried quietly, more are shown on the guide. */
    let failures = 0;
    while (gen === this.camGen) {
      const t0 = performance.now();
      const full = this.app.shots.length >= this.protocol.count;
      if (document.hidden || !video.videoWidth || full || t0 < this.takenUntil) {
        this.liveNote = full ? `All ${this.protocol.count} photos taken: now ${this.protocol.readLabel}` : null;
        if (full || document.hidden) this.lastCheck = null;
        this.hold.reset();
        previous = null;
      } else {
        try {
          const checked = await this.protocol.check(await frameBlob(video, canvas, CHECK_SIDE));
          if (gen !== this.camGen) return;
          failures = 0;
          const now = performance.now();
          // A photo taken while this frame was being checked: the verdict is about the old picture.
          if (now >= this.takenUntil) {
            const verdict = judge(
              checked,
              this.guideRect(video, guide),
              previous,
              this.app.shots.map((s) => s.signature),
              { sameAs: this.protocol.sameAs, maxUnsure: this.protocol.maxUnsure },
            );
            previous = checked.reading ? checked.quad : null;
            this.liveNote = null;
            this.lastCheck = { at: now, checked, verdict };
            this.hold.update(verdict.state === 'good', now);
          }
        } catch (err) {
          if (gen !== this.camGen) return;
          failures++;
          const status = err instanceof ApiError ? err.status : -1;
          const outdated = status === 404 || status === 405;
          if (failures >= 2 || outdated) {
            this.hold.reset();
            this.lastCheck = null;
            previous = null;
            this.liveNote = outdated
              ? 'The live check needs the updated server: restart python -m scanner serve'
              : status === 0 || status === 502 || status === 504
                ? 'Cannot reach the scanner server: retrying…'
                : `The live check failed: ${err instanceof Error ? err.message : String(err)}`;
          }
          await sleep(failures >= 2 || outdated ? 2500 : 200);
        }
      }
      await sleep(Math.max(0, CHECK_EVERY_MS - (performance.now() - t0)));
    }
  }

  /** The guide square in fractions of the camera's picture (the video is letterboxed into its box, maybe mirrored). */
  private guideRect(video: HTMLVideoElement, guide: HTMLElement): FrameRect {
    const v = video.getBoundingClientRect();
    const g = guide.getBoundingClientRect();
    const scale = Math.min(v.width / video.videoWidth, v.height / video.videoHeight);
    const cw = video.videoWidth * scale;
    const ch = video.videoHeight * scale;
    const ox = v.left + (v.width - cw) / 2;
    const oy = v.top + (v.height - ch) / 2;
    let x0 = (g.left - ox) / cw;
    let x1 = (g.right - ox) / cw;
    if (video.classList.contains('is-mirrored')) [x0, x1] = [1 - x1, 1 - x0];
    const clamp = (x: number) => Math.min(1, Math.max(0, x));
    return { x0: clamp(x0), y0: clamp((g.top - oy) / ch), x1: clamp(x1), y1: clamp((g.bottom - oy) / ch) };
  }

  /** Every animation frame: the guide's colour and words, the hold bar, the face outline; and the auto-capture. */
  private renderLive(video: HTMLVideoElement, face: SVGSVGElement, poly: SVGPolygonElement, guide: HTMLElement): void {
    const now = performance.now();
    const check = this.lastCheck && now - this.lastCheck.at < CHECK_FRESH_MS ? this.lastCheck : null;
    let text: string;
    let state: '' | 'good' | 'bad' = '';
    let progress = 0;
    if (now < this.takenUntil) {
      text = `Photo ${this.takenNumber} taken`;
      state = 'good';
    } else if (this.liveNote) {
      text = this.liveNote;
    } else if (!check) {
      text = 'Fill the square with one face';
    } else if (this.hold.green) {
      state = 'good';
      if (this.auto) {
        progress = this.hold.progress(now);
        text = `Hold still… ${Math.max(1, Math.ceil(((1 - progress) * LIVE.holdMs) / 1000))}`;
        // Taken only on a good check (a forgiven bad one waits for the next).
        if (progress >= 1 && check.verdict.state === 'good') {
          this.capture();
          return;
        }
      } else {
        text = 'Readable: press Space to capture';
      }
    } else {
      state = 'bad';
      text = check.verdict.message;
    }
    const label = guide.querySelector('.webcam__guide-text')!;
    if (label.textContent !== text) label.textContent = text;
    guide.classList.toggle('is-good', state === 'good');
    guide.classList.toggle('is-bad', state === 'bad');
    guide.classList.toggle('is-holding', progress > 0);
    guide.style.setProperty('--hold', progress.toFixed(3));
    // The face as the networks found it.
    const c = check?.checked;
    const showFace = !!c?.reading && c.confidence >= LIVE.minConfidence && now >= this.takenUntil;
    face.style.display = showFace ? '' : 'none';
    if (showFace && video.videoWidth) {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      face.setAttribute('viewBox', `0 0 ${vw} ${vh}`);
      poly.setAttribute('points', c!.quad.map(([x, y]) => `${(x * vw).toFixed(1)},${(y * vh).toFixed(1)}`).join(' '));
      face.classList.toggle('is-good', state === 'good');
      face.classList.toggle('is-bad', state === 'bad');
    }
  }

  private async openCamera(video: HTMLVideoElement, deviceId: string | undefined): Promise<void> {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        : { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = this.stream;
    this.applyMirror(video);
    await video.play().catch(() => {});
  }

  /**
   * Show the live picture like a mirror, so moving the cube left moves it left on screen. That is
   * right for a camera facing you (a laptop webcam); a phone's back camera already moves with you.
   * Only the preview is flipped: a captured photo is the camera's own picture, because a mirror image
   * of a face is not a face of any real cube.
   */
  private applyMirror(video: HTMLVideoElement): void {
    const facing = this.stream?.getVideoTracks()[0]?.getSettings().facingMode;
    const on = this.mirrorChoice ?? facing !== 'environment';
    video.classList.toggle('is-mirrored', on);
    video.parentElement?.querySelector('.webcam__face')?.classList.toggle('is-mirrored', on);
    if (this.mirrorBtn) {
      this.mirrorBtn.setAttribute('aria-pressed', String(on));
      this.mirrorBtn.classList.toggle('is-on', on);
    }
  }

  private stopWebcam(): void {
    this.camGen++;
    this.hold.reset();
    this.lastCheck = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.mirrorBtn = null;
    clearInterval(this.focusTimer);
    document.removeEventListener('keydown', this.onKey);
  }

  private onKey = (e: KeyboardEvent) => {
    if (e.code === 'Space' && this.mode === 'webcam' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
      e.preventDefault();
      this.capture();
    }
  };

  /** Take a photo now (Space, the button, or the auto-capture). The camera's own picture: never mirrored. */
  private capture(): void {
    // Whatever happens, the hold starts over (so a failed auto-capture is not retried every frame).
    this.hold.reset();
    const video = document.querySelector<HTMLVideoElement>('#webcam video');
    if (!video || !video.videoWidth) {
      this.app.toast('The webcam is not ready yet.', 'info');
      return;
    }
    if (this.app.shots.length >= this.protocol.count) {
      this.app.toast(`There are ${this.protocol.count} photos already. Remove one to take another.`, 'info');
      return;
    }
    const c = document.createElement('canvas');
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext('2d')!.drawImage(video, 0, 0);
    const n = this.app.shots.length + 1;
    // What the live check just read, to recognise this face if it is shown again.
    const now = performance.now();
    const check = this.lastCheck && now - this.lastCheck.at < CHECK_FRESH_MS ? this.lastCheck : null;
    const signature = check?.checked.reading && check.verdict.state !== 'none' ? this.protocol.signature(check.checked) : undefined;
    c.toBlob((b) => b && this.addShot(b, `webcam-${n}.jpg`, signature), 'image/jpeg', 0.93);
    this.lastCheck = null;
    this.takenUntil = now + TAKEN_PAUSE_MS;
    this.takenNumber = n;
    const view = document.querySelector('.webcam__view');
    view?.classList.remove('is-flash');
    void (view as HTMLElement | null)?.offsetWidth;
    view?.classList.add('is-flash');
  }

  // ------------------------------------------------------------------ panel

  private buildPanel(): void {
    const { panel } = this.app;
    const protocol = this.protocol;
    panel.append(this.app.header());

    const take = card(`Take ${protocol.count} photos`);
    take.innerHTML += protocol.instructions;
    const row = h('div', 'actions');
    this.readBtn = button(protocol.readLabel, 'btn btn--primary', () => void this.read());
    this.countEl = h('span', 'tag', `0 of ${protocol.count} photos`);
    row.append(this.readBtn, this.countEl);
    if (this.app.shots.length) {
      row.append(
        button('Clear', 'btn btn--ghost', () => {
          this.app.shots.forEach((s) => s.localUrl && URL.revokeObjectURL(s.localUrl));
          this.app.shots = [];
          this.renderSlots();
          this.buildPanelAgain();
        }),
      );
    }
    take.append(row);
    panel.append(take);

    panel.append(...this.app.puzzle.photoCards(this.app, () => this.buildPanelAgain()));
    panel.append(this.app.serverLine());
    this.updateReadButton();
  }

  private buildPanelAgain(): void {
    if (this.app.step !== 'photos') return;
    this.app.panel.replaceChildren();
    this.buildPanel();
  }

  // ------------------------------------------------------------------ actions

  private async read(): Promise<void> {
    const app = this.app;
    if (app.shots.length !== this.protocol.count || this.busy) return;
    this.busy = true;
    this.updateReadButton();
    this.readBtn.innerHTML = `<span class="spinner"></span> Reading the ${this.protocol.count} photos…`;
    try {
      await app.puzzle.read(app, app.shots);
    } catch (err) {
      app.toast(err instanceof ApiError ? err.message : `Reading failed: ${String(err)}`, 'error', 9000);
    } finally {
      this.busy = false;
      if (app.step === 'photos') {
        this.readBtn.textContent = this.protocol.readLabel;
        this.updateReadButton();
      }
    }
  }
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** The video's current frame, shrunk so its longer side is at most `side`, as a JPEG. */
function frameBlob(video: HTMLVideoElement, canvas: HTMLCanvasElement, side: number): Promise<Blob> {
  const f = Math.min(1, side / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * f);
  canvas.height = Math.round(video.videoHeight * f);
  canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('no frame'))), 'image/jpeg', 0.85));
}
