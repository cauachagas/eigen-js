import { describe, it, expect, beforeAll } from 'vitest';
import eig from '../dist/index.mjs';

describe('Matrix operations', () => {
  beforeAll(async () => {
    await eig.ready;
  });

  it('creates matrices and checks dimensions', () => {
    const M = new eig.Matrix([
      [1, 2, 3],
      [4, 5, 6]
    ]);
    expect(M.rows()).toBe(2);
    expect(M.cols()).toBe(3);
    expect(M.get(0, 0)).toBe(1);
    expect(M.get(0, 1)).toBe(2);
    expect(M.get(1, 2)).toBe(6);
    eig.GC.flush();
  });

  it('computes matrix multiplication correctly', () => {
    const A = new eig.Matrix([
      [1, 2],
      [3, 4]
    ]);
    const B = new eig.Matrix([
      [2, 0],
      [1, 2]
    ]);
    const C = A.matMul(B);
    // [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
    // [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
    expect(C.get(0, 0)).toBeCloseTo(4);
    expect(C.get(0, 1)).toBeCloseTo(4);
    expect(C.get(1, 0)).toBeCloseTo(10);
    expect(C.get(1, 1)).toBeCloseTo(8);
    eig.GC.flush();
  });

  it('inverts an invertible matrix', () => {
    const A = new eig.Matrix([
      [4, 7],
      [2, 6]
    ]);
    // det(A) = 24 - 14 = 10
    // A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
    expect(A.det()).toBeCloseTo(10);
    const Ainv = A.inverse();
    expect(Ainv.get(0, 0)).toBeCloseTo(0.6);
    expect(Ainv.get(0, 1)).toBeCloseTo(-0.7);
    expect(Ainv.get(1, 0)).toBeCloseTo(-0.2);
    expect(Ainv.get(1, 1)).toBeCloseTo(0.4);

    const I = A.matMul(Ainv);
    expect(I.get(0, 0)).toBeCloseTo(1);
    expect(I.get(0, 1)).toBeCloseTo(0);
    expect(I.get(1, 0)).toBeCloseTo(0);
    expect(I.get(1, 1)).toBeCloseTo(1);
    eig.GC.flush();
  });

  it('supports special matrix initializers', () => {
    const I = eig.Matrix.identity(3, 3);
    expect(I.get(0, 0)).toBe(1);
    expect(I.get(1, 1)).toBe(1);
    expect(I.get(0, 1)).toBe(0);

    const ones = eig.Matrix.ones(2, 2);
    expect(ones.get(0, 0)).toBe(1);
    expect(ones.get(1, 1)).toBe(1);
    expect(ones.sum()).toBeCloseTo(4);

    eig.GC.flush();
  });
});
