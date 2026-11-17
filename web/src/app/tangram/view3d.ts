import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { Axis, Move } from '../../vendor/tangram/moves';
import { Animator } from '../../vendor/tangram/render/animator';
import { PartLabels } from '../../vendor/tangram/render/labels';
import { type PieceVisual, buildPieceVisuals, setHighlighted } from '../../vendor/tangram/render/meshes';
import { PuzzleState } from '../../vendor/tangram/state';
import { GestureControls } from '../../vendor/render/controls';
import { type Stage, type ViewPreset, createStage } from '../../vendor/render/scene';
import { FACE_VIEWS, MODEL } from '../../tangram/model';
import { MODEL_VIEWS, PALETTE } from './model';

/**
 * The interactive 3D Tangram Cube Extreme used on every screen: the 3D simulator's own meshes, moved
 * between screens. Drag to rotate, pinch or Ctrl+wheel to zoom. U D F R L labels mark the turning parts
 * (as on the simulator site); badges just outside the faces can show photo numbers.
 */
export class TangramView {
  readonly el: HTMLElement;
  readonly stage: Stage;
  private readonly views: Record<string, ViewPreset> = { ...MODEL_VIEWS };
  private readonly visuals: PieceVisual[];
  private readonly animator: Animator;
  private readonly labels: PartLabels;
  private readonly badges: { obj: CSS2DObject; el: HTMLElement; dir: THREE.Vector3 }[] = [];
  private readonly camDir = new THREE.Vector3();
  private readonly tmp = new THREE.Vector3();
  /** Each sticker material's model colour (it is painted in the cube's colours: PALETTE). */
  private readonly stickerColour = new WeakMap<object, number>();
  /** Called after each animated move has been applied. */
  onApplied: (m: Move) => void = () => {};

  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'stage cube-view';
    this.el.setAttribute('aria-label', '3D model of the puzzle. Drag to turn it around.');
    this.stage = createStage(this.el, this.views, 'corner');
    new GestureControls(this.el, this.stage, { onTap: () => {}, onGestureStart: () => {} });
    this.visuals = buildPieceVisuals(MODEL);
    this.animator = new Animator(this.stage.root, this.visuals, PuzzleState.solved(MODEL));
    this.animator.onMoveApplied = (m) => this.onApplied(m);
    this.labels = new PartLabels(this.stage.root);
    this.labels.build(MODEL);
    const group = new THREE.Group();
    this.stage.root.add(group);
    for (const f of FACE_VIEWS) {
      const el = document.createElement('div');
      el.className = 'part-label cube-badge';
      const obj = new CSS2DObject(el);
      const [x, y, z] = f.normal;
      obj.position.set(x * 1.75, y * 1.75, z * 1.75);
      obj.visible = false;
      group.add(obj);
      this.badges.push({ obj, el, dir: new THREE.Vector3(x, y, z) });
    }
    this.repaint();
    PALETTE.onChange(() => this.repaint());
    requestAnimationFrame(this.loop);
  }

  /** Paint the stickers in the colours of the cube in hand (the scan's colour layout). */
  private repaint(): void {
    PALETTE.repaint(
      this.visuals.flatMap((v) => v.stickerMaterials),
      this.stickerColour,
    );
  }

  private loop = (now: number) => {
    this.animator.update(now);
    this.labels.update(this.stage.camera);
    this.camDir.copy(this.stage.camera.position).normalize();
    for (const b of this.badges) {
      if (!b.obj.visible) continue;
      const facing = this.tmp.copy(b.dir).applyQuaternion(this.stage.root.quaternion).dot(this.camDir);
      b.el.classList.toggle('is-behind', facing < -0.2);
    }
    this.stage.frame(now);
    requestAnimationFrame(this.loop);
  };

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
  show(state: PuzzleState): void {
    this.animator.reset(this.visuals, state.clone());
    this.highlight(null);
  }

  /** The state on screen (after the animations that already finished). */
  get state(): PuzzleState {
    return this.animator.visualState;
  }

  get busy(): boolean {
    return this.animator.busy;
  }

  play(move: Move, durationMs: number): void {
    this.highlight(null);
    this.animator.enqueue(move, durationMs);
  }

  /** Light up the part a move of `axis` would turn, or nothing. */
  highlight(axis: Axis | null): void {
    const moving = axis ? new Set(this.animator.visualState.movingPieces({ axis, amount: 1 })) : new Set<number>();
    for (const v of this.visuals) setHighlighted(v, moving.has(v.pieceId));
    this.labels.setActive(axis);
  }

  showLabels(on: boolean): void {
    this.labels.setVisible(on);
  }

  /** Labels just outside faces (FACE_IDS order); faces given no text get none. */
  setBadges(text: readonly (string | undefined)[], title: readonly (string | undefined)[] = []): void {
    this.badges.forEach((b, i) => {
      const t = text[i];
      b.obj.visible = !!t;
      b.el.style.display = t ? '' : 'none';
      b.el.textContent = t ?? '';
      b.el.title = title[i] ?? '';
    });
  }
}
