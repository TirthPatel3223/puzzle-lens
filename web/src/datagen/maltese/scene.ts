import * as THREE from 'three';
import { FACES, FACE_NORMAL } from '../../vendor/maltese/geometry';
import { PIECES } from '../../vendor/maltese/model';
import { type MalteseVisual, buildCoreMesh, buildMalteseVisuals, setPieceTransform } from '../../vendor/maltese/render/meshes';
import type { MalteseGeometry } from '../../vendor/maltese/render/pieces';
import type { MalteseState } from '../../vendor/maltese/state';
import { mulberry32 } from '../../vendor/puzzle/rng';
import { FACE_RIGHT, FACE_UP, colourOf } from '../../maltese/observation';
import { StudioBase, v3 } from '../common/studio';
import { type SampleSpec, gauss } from './randomise';

/**
 * The Maltese Gear Cube in the photo studio: the exact puzzle meshes. `render(spec, state)` applies one
 * SampleSpec and draws it; nothing random happens here.
 */

export interface Rendered {
  /** The face's four corners in the picture, as fractions of its width and height (y down): top-left, top-right, bottom-right, bottom-left of the face in its own orientation. */
  readonly quad: number[][];
}

export class Studio extends StudioBase {
  private readonly visuals: MalteseVisual[];

  constructor(canvas: HTMLCanvasElement, geo: MalteseGeometry, size: number) {
    super(canvas, size);
    this.visuals = buildMalteseVisuals(geo);
    for (const v of this.visuals) {
      v.meshes.forEach((m, i) => {
        // Bodies shade each other (a proud edge darkens the face beside it); stickers only receive.
        m.castShadow = i === 0;
        m.receiveShadow = true;
      });
      this.scene.add(v.group);
    }
    const core = buildCoreMesh(geo);
    core.castShadow = true;
    core.receiveShadow = true;
    this.scene.add(core);
  }

  /** `paint`: the cube's colour layout (paint[c] is shown where the model has colour c). */
  render(spec: SampleSpec, state: MalteseState, paint: readonly number[] = [0, 1, 2, 3, 4, 5]): Rendered {
    const face = FACES[spec.face];
    const n = v3(FACE_NORMAL[face]);

    // --- the cube --------------------------------------------------------------------------------------
    const jitter = mulberry32(spec.seed ^ 0x5bd1e995);
    for (const v of this.visuals) {
      setPieceTransform(v, state.matrix(v.pieceId));
      v.bodyMaterial.color.setRGB(spec.body.colour[0], spec.body.colour[1], spec.body.colour[2], THREE.SRGBColorSpace);
      v.bodyMaterial.roughness = spec.body.roughness;
      v.bodyMaterial.clearcoat = spec.body.clearcoat;
      const piece = PIECES[v.pieceId];
      v.stickerMaterials.forEach((m, i) => {
        const c = spec.stickers[paint[colourOf(piece.colors[i])]];
        const k = 1 + spec.stickerLook.spread * gauss(jitter); // stickers of one colour are not all equally fresh
        m.color.setRGB(Math.min(1, c[0] * k), Math.min(1, c[1] * k), Math.min(1, c[2] * k), THREE.SRGBColorSpace);
        m.roughness = spec.stickerLook.roughness;
        m.clearcoat = spec.stickerLook.clearcoat;
      });
    }

    this.applyScene(spec, n);
    this.draw();
    return { quad: this.project(spec.face, [[-1, 1], [1, 1], [1, -1], [-1, -1]]) };
  }

  /** Where points of a face (face coordinates: x right, y up, the face is [-1, 1]²) fall in the last picture drawn. */
  project(faceIndex: number, points: readonly (readonly number[])[]): number[][] {
    const face = FACES[faceIndex];
    const n = v3(FACE_NORMAL[face]);
    const right = v3(FACE_RIGHT[face]);
    const up = v3(FACE_UP[face]);
    return this.projectWorld(points.map(([x, y]) => {
      const p = n.clone().addScaledVector(right, x).addScaledVector(up, y);
      return [p.x, p.y, p.z] as const;
    }));
  }
}
