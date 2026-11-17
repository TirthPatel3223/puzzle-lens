import wasmUrl from 'manifold-3d/manifold.wasm?url';
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { FACES, FACE_NORMAL, type Face } from '../../vendor/maltese/geometry';
import type { Move } from '../../vendor/maltese/notation';
import { MalteseAnimator } from '../../vendor/maltese/render/animator';
import { type MalteseVisual, buildCoreMesh, buildMalteseVisuals, setHighlight } from '../../vendor/maltese/render/meshes';
import { buildMalteseGeometry, defaultFillet } from '../../vendor/maltese/render/pieces';
import { MalteseState } from '../../vendor/maltese/state';
import { GestureControls } from '../../vendor/render/controls';
import { loadManifoldKernel } from '../../vendor/render/manifold';
import { type Stage, type ViewPreset, createStage } from '../../vendor/render/scene';
import { HELD_OVERVIEW, PALETTE } from './model';

const LAYER_TINT = new THREE.Color('#3b82f6');

/**
 * The interactive 3D Maltese Gear Cube used on every screen: one WebGL stage, moved between screens.
 * Drag to rotate, pinch or Ctrl+wheel to zoom. Pieces are built once (about a second of mesh booleans).
 */
export class CubeView {
  readonly el: HTMLElement;
  readonly stage: Stage;
  readonly ready: Promise<void>;
  private readonly views: Record<string, ViewPreset> = { overview: HELD_OVERVIEW };
  private animator: MalteseAnimator | null = null;
  private visuals: MalteseVisual[] = [];
  private shown: MalteseState = MalteseState.solved();
  private readonly badgeGroup = new THREE.Group();
  private readonly badges = new Map<Face, { obj: CSS2DObject; el: HTMLElement }>();
  private readonly camDir = new THREE.Vector3();
  private readonly tmp = new THREE.Vector3();
  /** Each sticker material's model colour (it is painted in the cube's colours: PALETTE). */
  private readonly stickerColour = new WeakMap<object, number>();
  /** Called after each animated move has been applied. */
  onApplied: (m: Move) => void = () => {};

  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'stage cube-view';
    this.el.setAttribute('aria-label', '3D model of the cube. Drag to turn it around.');
    this.stage = createStage(this.el, this.views, 'overview');
    new GestureControls(this.el, this.stage, { onTap: () => {}, onGestureStart: () => {} });
    this.stage.root.add(this.badgeGroup);
    for (const f of FACES) {
      const el = document.createElement('div');
      el.className = 'part-label cube-badge';
      const obj = new CSS2DObject(el);
      const [x, y, z] = FACE_NORMAL[f];
      obj.position.set(x * 1.62, y * 1.62, z * 1.62);
      obj.visible = false;
      this.badgeGroup.add(obj);
      this.badges.set(f, { obj, el });
    }
    requestAnimationFrame(this.loop);
    this.ready = this.build();
  }

  private loop = (now: number) => {
    this.animator?.update(now);
    this.camDir.copy(this.stage.camera.position).normalize();
    for (const [f, b] of this.badges) {
      if (!b.obj.visible) continue;
      const facing = this.tmp.set(...FACE_NORMAL[f]).applyQuaternion(this.stage.root.quaternion).dot(this.camDir);
      b.el.classList.toggle('is-behind', facing < -0.2);
    }
    this.stage.frame(now);
    requestAnimationFrame(this.loop);
  };

  private async build(): Promise<void> {
    const wasm = await loadManifoldKernel(wasmUrl);
    const geo = buildMalteseGeometry(wasm, defaultFillet());
    this.visuals = buildMalteseVisuals(geo);
    for (const v of this.visuals) this.stage.root.add(v.group);
    this.stage.root.add(buildCoreMesh(geo));
    this.animator = new MalteseAnimator(this.visuals, this.shown.clone());
    this.animator.onApplied = (m) => this.onApplied(m);
    this.repaint();
    PALETTE.onChange(() => this.repaint());
  }

  /** Paint the stickers in the colours of the cube in hand (the scan's colour layout). */
  private repaint(): void {
    PALETTE.repaint(
      this.visuals.flatMap((v) => v.stickerMaterials),
      this.stickerColour,
    );
  }

  /** Put the view inside `parent` (it fills it). */
  mount(parent: HTMLElement): void {
    if (this.el.parentElement !== parent) parent.appendChild(this.el);
    this.stage.resize();
  }

  defineView(name: string, preset: ViewPreset): void {
    this.views[name] = preset;
  }

  setView(name: string, animate = true): void {
    this.stage.setView(name, animate);
  }

  /** Show a state at once (any running animation is dropped). */
  show(state: MalteseState): void {
    this.shown = state.clone();
    if (this.animator) this.animator.reset(state.clone());
    this.highlight(null);
  }

  /** The state on screen (after the animations that already finished). */
  get state(): MalteseState {
    return this.animator ? this.animator.visualState : this.shown;
  }

  get busy(): boolean {
    return this.animator?.busy ?? false;
  }

  /** Animate one move. Returns false while the pieces are still being built. */
  play(move: Move, durationMs: number): boolean {
    if (!this.animator) return false;
    this.highlight(null);
    this.animator.enqueue(move, durationMs);
    return true;
  }

  /**
   * Light up the layer a turn of `face` would move (its ring fainter), or nothing. Only the black
   * plastic is tinted, so the sticker colours stay true for comparing with the real cube.
   */
  highlight(face: Face | null): void {
    if (!this.animator) return;
    for (const v of this.visuals) setHighlight(v, 'none');
    if (!face) return;
    const { layer, ring } = this.animator.visualState.movingParts(face);
    const inLayer = new Set(layer);
    const inRing = new Set(ring);
    for (const v of this.visuals) {
      const level = inLayer.has(v.pieceId) ? 0.42 : inRing.has(v.pieceId) ? 0.16 : 0;
      if (!level) continue;
      v.bodyMaterial.emissive.copy(LAYER_TINT);
      v.bodyMaterial.emissiveIntensity = level;
    }
  }

  /** Labels just outside faces (by engine face of the frame shown); faces left out get none. */
  setBadges(text: Partial<Record<Face, string>>, title: Partial<Record<Face, string>> = {}): void {
    for (const [f, b] of this.badges) {
      const t = text[f];
      b.obj.visible = !!t;
      b.el.style.display = t ? '' : 'none';
      b.el.textContent = t ?? '';
      b.el.title = title[f] ?? '';
    }
  }
}
