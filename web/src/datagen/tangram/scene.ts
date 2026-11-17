import * as THREE from 'three';
import { FACE_NORMALS } from '../../vendor/tangram/pieces';
import { type PieceVisual, buildPieceVisuals, lookDims, stickerOutline } from '../../vendor/tangram/render/meshes';
import type { PuzzleState } from '../../vendor/tangram/state';
import { mulberry32 } from '../../vendor/puzzle/rng';
import { type Vec3, add, dot, mat3Vec, scale } from '../../vendor/puzzle/vec';
import { FACE_VIEWS, MODEL, TABLES, colourOfFace, observe, placementOf } from '../../tangram/model';
import { gauss } from '../common/random';
import { StudioBase, v3 } from '../common/studio';
import type { TangramSpec } from './randomise';

/**
 * The Tangram Cube Extreme in the photo studio, and the exact labels of each picture:
 *   corners     the photographed face's four corners (top-left, top-right, bottom-right = the turning
 *               corner, bottom-left in the face's own frame), projected; the face is the middle layer's,
 *               so corners that a turned-out piece has carried away are still where the face's are
 *   content     what every position shows (colour or EMPTY)
 *   visible     the positions in the photo whose sticker the camera really sees (ray cast against every
 *               piece and the hand: at least half of the sticker's sample points unblocked)
 */

export interface TangramRendered {
  readonly corners: number[][];
  readonly content: number[];
  readonly visible: number[];
}

/** Fractions of the way from a sticker's centre towards its corners where visibility is sampled. */
const SAMPLE_REACH = 0.45;
/** Visibility is worked out for every position at least this face-on (a little wider than the reader's FACE_MIN_COS). */
const MIN_COS = 0.3;

export class TangramStudio extends StudioBase {
  private readonly visuals: PieceVisual[];
  /** Per piece: the sticker index (into PieceDef.stickers) of each sticker mesh (meshes[1 + k]). */
  private readonly stickerOf: number[][];
  private readonly occluders: THREE.Object3D[];
  private readonly raycaster = new THREE.Raycaster();
  private readonly lift: number;

  constructor(canvas: HTMLCanvasElement, size: number) {
    super(canvas, size, 4.2);
    this.visuals = buildPieceVisuals(MODEL);
    const d = lookDims(MODEL.variant);
    this.lift = d.lift;
    // The meshes skip stickers too small to draw; keep track of which is which.
    this.stickerOf = MODEL.pieces.map((p) =>
      p.stickers.flatMap((s, k) => (stickerOutline(s.polygon, FACE_NORMALS[s.face], d) ? [k] : [])),
    );
    for (const v of this.visuals) {
      v.meshes.forEach((m, i) => {
        m.castShadow = i === 0;
        m.receiveShadow = true;
      });
      this.scene.add(v.group);
    }
    this.occluders = [...this.visuals.flatMap((v) => v.meshes), ...this.fingers, this.palm];
  }

  /** `paint`: the puzzle's colour layout (paint[c] is shown where the model has colour c); labels follow it. */
  render(spec: TangramSpec, state: PuzzleState, paint: readonly number[] = [0, 1, 2, 3, 4, 5]): TangramRendered {
    const { group } = MODEL;
    const jitter = mulberry32(spec.seed ^ 0x5bd1e995);
    for (const v of this.visuals) {
      const [x, y, z, w] = group.quats[state.orient[v.pieceId]];
      v.group.quaternion.set(x, y, z, w);
      v.group.updateMatrixWorld(true);
      v.bodyMaterial.color.setRGB(spec.body.colour[0], spec.body.colour[1], spec.body.colour[2], THREE.SRGBColorSpace);
      v.bodyMaterial.roughness = spec.body.roughness;
      v.bodyMaterial.clearcoat = spec.body.clearcoat;
      const piece = MODEL.pieces[v.pieceId];
      v.stickerMaterials.forEach((m, k) => {
        const c = spec.stickers[paint[colourOfFace(piece.stickers[this.stickerOf[v.pieceId][k]].face)]];
        const f = 1 + spec.stickerLook.spread * gauss(jitter);
        m.color.setRGB(Math.min(1, c[0] * f), Math.min(1, c[1] * f), Math.min(1, c[2] * f), THREE.SRGBColorSpace);
        m.roughness = spec.stickerLook.roughness;
        m.clearcoat = spec.stickerLook.clearcoat;
      });
    }
    const view = FACE_VIEWS[spec.face];
    this.applyScene(spec, v3(view.normal));
    this.scene.updateMatrixWorld(true);
    this.draw();
    return {
      corners: this.projectWorld(view.corners),
      content: Array.from(observe(state), (x) => (x < paint.length ? paint[x] : x)),
      visible: this.visibleStickers(state, view.normal),
    };
  }

  /** Positions in the photo of the face with normal `facing` whose sticker the camera sees (at least half of its sample points). */
  private visibleStickers(state: PuzzleState, facing: Vec3): number[] {
    const out: number[] = [];
    const eye = this.camera.position;
    const dir = new THREE.Vector3();
    const target = new THREE.Vector3();
    for (const v of this.visuals) {
      const i = v.pieceId;
      const pl = placementOf(i, state.orient[i])!;
      const m = state.pieceMatrix(i);
      const piece = MODEL.pieces[i];
      this.stickerOf[i].forEach((k, meshIndex) => {
        const pos = pl.positions[k];
        if (dot(TABLES.positions[pos].normal, facing) < MIN_COS) return;
        const mesh = v.meshes[1 + meshIndex];
        const poly = piece.stickers[k].polygon.map((p) => mat3Vec(m, p));
        const n = mat3Vec(m, FACE_NORMALS[piece.stickers[k].face]);
        let c: Vec3 = [0, 0, 0];
        for (const p of poly) c = add(c, p);
        c = scale(c, 1 / poly.length);
        const samples = [c, ...poly.map((p) => add(c, scale(add(p, scale(c, -1)), SAMPLE_REACH)))].map((p) => add(p, scale(n, this.lift * 1.5)));
        let seen = 0;
        for (const s of samples) {
          target.set(s[0], s[1], s[2]);
          dir.subVectors(target, eye);
          const dist = dir.length();
          this.raycaster.set(eye, dir.normalize());
          this.raycaster.far = dist + 0.05;
          const hit = this.raycaster.intersectObjects(this.occluders, false)[0];
          if (hit && (hit.object === mesh || hit.distance > dist - 0.02)) seen++;
        }
        if (seen * 2 >= samples.length) out.push(pos);
      });
    }
    return out.sort((a, b) => a - b);
  }

  /** Where model points fall in the last picture drawn. */
  project(points: readonly Vec3[]): number[][] {
    return this.projectWorld(points);
  }
}


