import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { mulberry32 } from '../../vendor/puzzle/rng';
import type { Vec3 } from '../../vendor/puzzle/vec';
import { type LightSpec, type SceneSpec, gauss, uni } from './random';
import { paintTexture } from './textures';

/**
 * The photo studio every puzzle's synthetic photos are taken in: a WebGL renderer, three lights, an
 * environment to reflect, a backdrop, an optional table and an optional hand. A puzzle's studio extends
 * it with its own meshes; `applyScene(spec, toward)` sets everything else from a SceneSpec. Nothing random
 * happens here.
 */

const ENVIRONMENTS = 10;
const PLANE = 90;
const TONE = {
  none: THREE.NoToneMapping,
  linear: THREE.LinearToneMapping,
  neutral: THREE.NeutralToneMapping,
  aces: THREE.ACESFilmicToneMapping,
  agx: THREE.AgXToneMapping,
} as const;

export const v3 = (v: Vec3): THREE.Vector3 => new THREE.Vector3(v[0], v[1], v[2]);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);

/** A room of glowing panels to reflect in the plastic: every one is a different mix of windows and lamps. */
function proceduralRoom(seed: number): THREE.Scene {
  const rng = mulberry32(seed);
  const scene = new THREE.Scene();
  const wall = uni(rng, 0.02, 0.6);
  const room = new THREE.Mesh(
    new THREE.BoxGeometry(20, 20, 20),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(wall * uni(rng, 0.85, 1.1), wall, wall * uni(rng, 0.85, 1.1)), side: THREE.BackSide }),
  );
  scene.add(room);
  const panels = 2 + Math.floor(rng() * 6);
  for (let i = 0; i < panels; i++) {
    const power = uni(rng, 2, 40);
    const tint = new THREE.Color(uni(rng, 0.75, 1), uni(rng, 0.8, 1), uni(rng, 0.7, 1)).multiplyScalar(power);
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(uni(rng, 1, 9), uni(rng, 1, 9)), new THREE.MeshBasicMaterial({ color: tint, side: THREE.DoubleSide }));
    const dir = new THREE.Vector3(gauss(rng), gauss(rng), gauss(rng)).normalize();
    panel.position.copy(dir).multiplyScalar(9);
    panel.lookAt(0, 0, 0);
    scene.add(panel);
  }
  return scene;
}

export class StudioBase {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(30, 1, 0.05, 400);
  private readonly key = new THREE.DirectionalLight();
  private readonly fills = [new THREE.DirectionalLight(), new THREE.DirectionalLight()];
  private readonly hemisphere = new THREE.HemisphereLight();
  private readonly environments: THREE.Texture[] = [];
  private readonly backdropCanvas = document.createElement('canvas');
  private readonly groundCanvas = document.createElement('canvas');
  private readonly backdropTexture: THREE.CanvasTexture;
  private readonly groundTexture: THREE.CanvasTexture;
  private readonly backdropLit: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshStandardMaterial>;
  private readonly backdropFlat: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private readonly ground: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshStandardMaterial>;
  /** The hand's meshes (occluders the labels must know about). */
  readonly fingers: THREE.Mesh[] = [];
  readonly palm: THREE.Mesh;
  private readonly skin = new THREE.MeshStandardMaterial({ roughness: 0.62 });

  constructor(
    canvas: HTMLCanvasElement,
    readonly size: number,
    /** Half-width of the shadow camera: the puzzle must fit inside it. */
    shadowHalf = 3.6,
  ) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(1);
    renderer.setSize(size, size, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer = renderer;

    const pmrem = new THREE.PMREMGenerator(renderer);
    this.environments.push(pmrem.fromScene(new RoomEnvironment(), 0.04).texture);
    for (let i = 1; i < ENVIRONMENTS; i++) this.environments.push(pmrem.fromScene(proceduralRoom(7919 * i), 0.03).texture);
    pmrem.dispose();

    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    const sc = this.key.shadow.camera;
    sc.left = sc.bottom = -shadowHalf;
    sc.right = sc.top = shadowHalf;
    sc.near = 1;
    sc.far = 40;
    sc.updateProjectionMatrix();
    this.key.shadow.bias = -0.0004;
    this.key.shadow.normalBias = 0.012;
    this.scene.add(this.key, ...this.fills, this.hemisphere);

    const texture = (c: HTMLCanvasElement) => {
      c.width = c.height = 4;
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = t.wrapT = THREE.MirroredRepeatWrapping;
      return t;
    };
    this.backdropTexture = texture(this.backdropCanvas);
    this.groundTexture = texture(this.groundCanvas);
    const plane = new THREE.PlaneGeometry(PLANE, PLANE);
    this.backdropLit = new THREE.Mesh(plane, new THREE.MeshStandardMaterial({ map: this.backdropTexture, roughness: 0.8 }));
    this.backdropLit.receiveShadow = true;
    this.backdropFlat = new THREE.Mesh(plane, new THREE.MeshBasicMaterial({ map: this.backdropTexture }));
    this.ground = new THREE.Mesh(plane, new THREE.MeshStandardMaterial({ map: this.groundTexture, roughness: 0.7 }));
    this.ground.receiveShadow = true;
    this.scene.add(this.backdropLit, this.backdropFlat, this.ground);

    const capsule = new THREE.CapsuleGeometry(1, 1, 6, 20);
    for (let i = 0; i < 4; i++) {
      const f = new THREE.Mesh(capsule, this.skin);
      f.castShadow = true;
      f.receiveShadow = true;
      this.fingers.push(f);
      this.scene.add(f);
    }
    this.palm = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), this.skin);
    this.palm.receiveShadow = true;
    this.scene.add(this.palm);
  }

  get rendererName(): string {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : 'unknown';
  }

  private setLight(light: THREE.DirectionalLight, spec: LightSpec): void {
    light.position.copy(v3(spec.dir)).multiplyScalar(14);
    light.intensity = spec.intensity;
    light.color.setRGB(spec.colour[0], spec.colour[1], spec.colour[2], THREE.SRGBColorSpace);
  }

  /** Light, environment, backdrop (square to `toward`, behind the puzzle), table, hand and camera. */
  applyScene(spec: SceneSpec, toward: THREE.Vector3): void {
    const { renderer, scene, camera } = this;
    const n = toward;

    // --- light -----------------------------------------------------------------------------------------
    this.setLight(this.key, spec.key);
    this.key.shadow.intensity = spec.shadow.intensity;
    this.key.shadow.radius = spec.shadow.radius;
    spec.fills.forEach((f, i) => this.setLight(this.fills[i], f));
    this.hemisphere.color.setRGB(spec.hemisphere.sky[0], spec.hemisphere.sky[1], spec.hemisphere.sky[2], THREE.SRGBColorSpace);
    this.hemisphere.groundColor.setRGB(spec.hemisphere.ground[0], spec.hemisphere.ground[1], spec.hemisphere.ground[2], THREE.SRGBColorSpace);
    this.hemisphere.intensity = spec.hemisphere.intensity;
    this.hemisphere.position.copy(v3(spec.camera.up));
    scene.environment = this.environments[spec.environment.index % this.environments.length];
    scene.environmentIntensity = spec.environment.intensity;
    scene.environmentRotation.set(spec.environment.rotation * 0.71, spec.environment.rotation, spec.environment.rotation * 1.37);
    renderer.toneMapping = TONE[spec.toneMapping];
    renderer.toneMappingExposure = spec.exposure;

    // --- behind and under ------------------------------------------------------------------------------
    paintTexture(this.backdropCanvas, spec.backdrop.texture);
    this.backdropTexture.needsUpdate = true;
    const tiles = 4 * spec.backdrop.texture.repeat;
    this.backdropTexture.repeat.set(tiles, tiles);
    const backdrop = spec.backdrop.lit ? this.backdropLit : this.backdropFlat;
    this.backdropLit.visible = spec.backdrop.lit;
    this.backdropFlat.visible = !spec.backdrop.lit;
    backdrop.position.copy(n).multiplyScalar(-spec.backdrop.distance);
    backdrop.quaternion.setFromUnitVectors(Z, n);
    if (spec.backdrop.lit) this.backdropLit.material.color.setScalar(1);
    else this.backdropFlat.material.color.setScalar(spec.backdrop.brightness);

    this.ground.visible = spec.ground !== null;
    if (spec.ground) {
      paintTexture(this.groundCanvas, spec.ground.texture);
      this.groundTexture.needsUpdate = true;
      const g = 4 * spec.ground.texture.repeat;
      this.groundTexture.repeat.set(g, g);
      const gn = v3(spec.ground.normal);
      this.ground.position.copy(gn).multiplyScalar(-spec.ground.offset);
      this.ground.quaternion.setFromUnitVectors(Z, gn);
      this.ground.material.roughness = spec.ground.roughness;
    }

    // --- a hand ----------------------------------------------------------------------------------------
    this.skin.color.setRGB(spec.skin[0], spec.skin[1], spec.skin[2], THREE.SRGBColorSpace);
    this.fingers.forEach((mesh, i) => {
      const f = spec.fingers[i];
      mesh.visible = f !== undefined;
      if (!f) return;
      mesh.position.copy(v3(f.centre));
      mesh.quaternion.setFromUnitVectors(Y, v3(f.axis));
      mesh.scale.set(f.radius, f.length / 3, f.radius); // the capsule is 3 long and 1 in radius
    });
    this.palm.visible = spec.palm !== null;
    if (spec.palm) {
      this.palm.position.copy(v3(spec.palm.centre));
      this.palm.scale.copy(v3(spec.palm.scale));
    }

    // --- camera ----------------------------------------------------------------------------------------
    camera.fov = spec.camera.fov;
    camera.position.copy(v3(spec.camera.position));
    camera.up.copy(v3(spec.camera.up));
    camera.lookAt(v3(spec.camera.lookAt));
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  }

  draw(): void {
    this.renderer.render(this.scene, this.camera);
  }

  /** Where world points fall in the last picture drawn, as fractions of its width and height (y down). */
  projectWorld(points: readonly Vec3[]): number[][] {
    return points.map((p) => {
      const q = v3(p).project(this.camera);
      return [(q.x + 1) / 2, (1 - q.y) / 2];
    });
  }
}
