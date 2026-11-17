import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { PALETTE as MALTESE, SOLVER_FACE_WORDS } from '../src/app/maltese/model';
import { Palette, describeLayout, isLayout } from '../src/app/palette';
import { LANDMARKS, PALETTE as TANGRAM, colourWord, describeMove, faceName, parseMoves } from '../src/app/tangram/model';
import { COLOUR_HEX } from '../src/tangram/model';

/** A DaYan Tangram Cube Extreme: orange and red swapped, green and blue swapped. */
const DAYAN = [1, 0, 2, 3, 5, 4];

describe('a cube stickered another way (palette)', () => {
  it('names and draws the model colours as the cube shows them', () => {
    TANGRAM.set(DAYAN);
    try {
      expect(TANGRAM.usual).toBe(false);
      expect(TANGRAM.nameOf(0)).toBe('red');
      expect(TANGRAM.hexOf(4)).toBe(COLOUR_HEX[5]);
      expect(TANGRAM.modelOf(1)).toBe(0);              // the person picks red: the model's orange
      expect(colourWord(6)).toBe('no sticker');
      expect(faceName(0)).toBe('red face');
    } finally {
      TANGRAM.set(null);
    }
    expect(TANGRAM.usual).toBe(true);
    expect(TANGRAM.nameOf(0)).toBe('orange');
  });

  it('says instructions in the cube colours, whole words only', () => {
    TANGRAM.set(DAYAN);
    try {
      expect(LANDMARKS.U).toBe('the red-yellow-blue corner (the T corner)');
      expect(LANDMARKS.D).toBe('the orange-white-green corner (the B corner)');
      expect(describeMove(parseMoves('U2')[0])).toContain('Top corner (red-yellow-blue)');
      expect(TANGRAM.say('Orange face up; shared, ordered and redone stay.')).toBe('Red face up; shared, ordered and redone stay.');
    } finally {
      TANGRAM.set(null);
    }
    expect(LANDMARKS.U).toBe('the orange-yellow-green corner (the T corner)');
  });

  it('tells listeners, and ignores something that is not a layout', () => {
    const p = new Palette(['#000000', '#111111', '#222222', '#333333', '#444444', '#555555'], ['a', 'b', 'c', 'd', 'e', 'f']);
    let calls = 0;
    const stop = p.onChange(() => calls++);
    p.set([1, 0, 2, 3, 4, 5]);
    p.set([1, 0, 2, 3, 4, 5]);                            // no change: no call
    p.set([0, 0, 1, 2, 3, 4]);                            // not a layout: the usual colours
    stop();
    p.set([1, 0, 2, 3, 4, 5]);
    expect(calls).toBe(2);
    expect(isLayout([5, 4, 3, 2, 1, 0])).toBe(true);
    expect(isLayout([0, 1, 2, 3, 4])).toBe(false);
  });

  it('repaints three.js sticker materials and remembers their model colour', () => {
    const mats = COLOUR_HEX.map((hex) => new THREE.MeshPhysicalMaterial({ color: hex }));
    const memo = new WeakMap<object, number>();
    TANGRAM.set(DAYAN);
    try {
      TANGRAM.repaint(mats, memo);
      expect(mats.map((m) => `#${m.color.getHexString()}`)).toEqual(DAYAN.map((k) => COLOUR_HEX[k].toLowerCase()));
      TANGRAM.set(null);
      TANGRAM.repaint(mats, memo);                       // back: the memo knows each one's model colour
      expect(mats.map((m) => `#${m.color.getHexString()}`)).toEqual(COLOUR_HEX.map((x) => x.toLowerCase()));
    } finally {
      TANGRAM.set(null);
    }
  });

  it('describes a layout as the server reports it', () => {
    const changed = [
      { model: 'orange', cube: 'red' },
      { model: 'red', cube: 'orange' },
      { model: 'green', cube: 'blue' },
      { model: 'blue', cube: 'green' },
    ];
    expect(describeLayout({ paint: DAYAN, kind: 'rearranged', usual: false, changed })).toBe('orange and red swapped; green and blue swapped');
    expect(describeLayout({ paint: [0, 1, 2, 3, 4, 5], kind: 'default', usual: true, changed: [] })).toBe('the usual colours');
  });

  it('the Maltese Gear Cube says its solving faces in the cube colours too', () => {
    MALTESE.set([0, 1, 2, 3, 5, 4]);                     // a mirror image: red and purple swapped
    try {
      expect(SOLVER_FACE_WORDS.L).toBe('left (red) face');
      expect(MALTESE.say('Hold the cube blue on top, yellow in front, purple on the left')).toBe('Hold the cube blue on top, yellow in front, red on the left');
    } finally {
      MALTESE.set(null);
    }
  });
});
