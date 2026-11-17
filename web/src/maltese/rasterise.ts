import * as THREE from 'three';

/**
 * A small software renderer: orthographic, z-buffered, one flat colour per mesh. It needs no WebGL,
 * so it runs in tests and in Node tools. Adapted from Cube_Modeling's tools/favicons/render.gen.ts.
 *
 * `rows` are the view axes in model coordinates: screen right, screen up, and towards the viewer.
 */
export interface Raster {
  readonly size: number;
  /** Half-width of the picture in model units. */
  readonly extent: number;
  /** Per pixel: index into `palette` (0 = nothing there). */
  readonly index: Uint8Array;
  /** Material colours as '#rrggbb'; palette[0] is 'none'. */
  readonly palette: string[];
  /** Per pixel: how far the visible surface is towards the viewer, in model units (-Infinity where empty). */
  readonly depth: Float64Array;
}

export type ViewRows = readonly [readonly number[], readonly number[], readonly number[]];

export function rasterise(root: THREE.Object3D, rows: ViewRows, size: number, extent = 1.25): Raster {
  root.updateMatrixWorld(true);
  const depth = new Float64Array(size * size).fill(-Infinity);
  const index = new Uint8Array(size * size);
  const palette: string[] = ['none'];
  const visible = new THREE.Layers();
  const v = new THREE.Vector3();
  const project = (p: THREE.Vector3): [number, number, number] => {
    const x = rows[0][0] * p.x + rows[0][1] * p.y + rows[0][2] * p.z;
    const y = rows[1][0] * p.x + rows[1][1] * p.y + rows[1][2] * p.z;
    const z = rows[2][0] * p.x + rows[2][1] * p.y + rows[2][2] * p.z;
    return [((x + extent) / (2 * extent)) * size, ((extent - y) / (2 * extent)) * size, z];
  };
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.layers.test(visible) || !mesh.visible) return;
    const hex = '#' + (mesh.material as THREE.MeshStandardMaterial).color.getHexString();
    let ci = palette.indexOf(hex);
    if (ci < 0) ci = palette.push(hex) - 1;
    const pos = mesh.geometry.getAttribute('position');
    const idx = mesh.geometry.getIndex();
    const nTri = (idx ? idx.count : pos.count) / 3;
    const pts: [number, number, number][] = [];
    for (let i = 0; i < pos.count; i++) pts.push(project(v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld)));
    for (let t = 0; t < nTri; t++) {
      const a = pts[idx ? idx.getX(3 * t) : 3 * t];
      const b = pts[idx ? idx.getX(3 * t + 1) : 3 * t + 1];
      const c = pts[idx ? idx.getX(3 * t + 2) : 3 * t + 2];
      const area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
      if (Math.abs(area) < 1e-12) continue;
      const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0], c[0])));
      const x1 = Math.min(size - 1, Math.ceil(Math.max(a[0], b[0], c[0])));
      const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1], c[1])));
      const y1 = Math.min(size - 1, Math.ceil(Math.max(a[1], b[1], c[1])));
      for (let y = y0; y <= y1; y++) {
        const sy = y + 0.5;
        for (let x = x0; x <= x1; x++) {
          const sx = x + 0.5;
          const w0 = ((b[0] - sx) * (c[1] - sy) - (b[1] - sy) * (c[0] - sx)) / area;
          const w1 = ((c[0] - sx) * (a[1] - sy) - (c[1] - sy) * (a[0] - sx)) / area;
          const w2 = 1 - w0 - w1;
          if (w0 < 0 || w1 < 0 || w2 < 0) continue;
          const z = w0 * a[2] + w1 * b[2] + w2 * c[2];
          const k = y * size + x;
          if (z > depth[k]) {
            depth[k] = z;
            index[k] = ci;
          }
        }
      }
    }
  });
  return { size, extent, index, palette, depth };
}

/** Pixel under the point (x right, y up) of the picture plane, in model units. */
export function pixelAt(r: Raster, x: number, y: number): number {
  const px = Math.min(r.size - 1, Math.max(0, Math.floor(((x + r.extent) / (2 * r.extent)) * r.size)));
  const py = Math.min(r.size - 1, Math.max(0, Math.floor(((r.extent - y) / (2 * r.extent)) * r.size)));
  return py * r.size + px;
}

/** Colour ('#rrggbb', or 'none') seen at a point of the picture plane. */
export function colourAt(r: Raster, x: number, y: number): string {
  return r.palette[r.index[pixelAt(r, x, y)]];
}

export function depthAt(r: Raster, x: number, y: number): number {
  return r.depth[pixelAt(r, x, y)];
}
