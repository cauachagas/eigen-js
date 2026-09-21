import { describe, it, expect, beforeAll } from 'vitest';
import eig from '../dist/index.mjs';

describe('Matrix Decompositions', () => {
  beforeAll(async () => {
    await eig.ready;
  });

  it('performs SVD (Singular Value Decomposition)', () => {
    const A = new eig.Matrix([
      [3, 2, 2],
      [2, 3, -2]
    ]);
    const svd = eig.Decompositions.svd(A, true);
    expect(svd.sv.rows()).toBe(2);
    // Singular values of [[3,2,2],[2,3,-2]] are 5 and 3
    expect(svd.sv.get(0, 0)).toBeCloseTo(5);
    expect(svd.sv.get(1, 0)).toBeCloseTo(3);
    eig.GC.flush();
  });

  it('computes LU decomposition', () => {
    const A = new eig.Matrix([
      [1, 2],
      [3, 4]
    ]);
    const lu = eig.Decompositions.lu(A);
    expect(lu.L.rows()).toBe(2);
    expect(lu.U.rows()).toBe(2);
    eig.GC.flush();
  });

  it('computes QR decomposition', () => {
    const A = new eig.Matrix([
      [12, -51],
      [6, 167],
      [-4, 24]
    ]);
    const qr = eig.Decompositions.qr(A);
    expect(qr.Q.rows()).toBe(3);
    expect(qr.R.rows()).toBe(3);
    eig.GC.flush();
  });

  it('computes Cholesky decomposition', () => {
    const A = new eig.Matrix([
      [4, 12],
      [12, 45]
    ]);
    const chol = eig.Decompositions.cholesky(A);
    // L should be [[2, 0], [6, 3]]
    expect(chol.L.get(0, 0)).toBeCloseTo(2);
    expect(chol.L.get(1, 0)).toBeCloseTo(6);
    expect(chol.L.get(1, 1)).toBeCloseTo(3);
    eig.GC.flush();
  });
});
