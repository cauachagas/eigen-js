import { describe, it, expect, beforeAll } from 'vitest';
import eig from '../dist/index.mjs';

describe('Sparse Matrix & SimplicialCholesky', () => {
  beforeAll(async () => {
    await eig.ready;
  });

  it('creates SparseMatrix from TripletVector', () => {
    const triplets = new eig.TripletVector(3);
    triplets.add(0, 0, 10);
    triplets.add(1, 1, 20);
    triplets.add(0, 1, 5);

    const sm = new eig.SparseMatrix(2, 2, triplets);
    expect(sm.rows()).toBe(2);
    expect(sm.cols()).toBe(2);
    expect(sm.nonZeros()).toBe(3);
    expect(sm.get(0, 0)).toBe(10);
    expect(sm.get(1, 1)).toBe(20);
    expect(sm.get(0, 1)).toBe(5);
    expect(sm.get(1, 0)).toBe(0);

    const dense = sm.toDense();
    expect(dense.get(0, 0)).toBe(10);
    expect(dense.get(1, 1)).toBe(20);
    eig.GC.flush();
  });

  it('solves linear systems with SimplicialCholesky', () => {
    // Solve A * x = b for positive definite A
    // A = [[4, 1], [1, 3]], b = [1, 2]
    const triplets = new eig.TripletVector(4);
    triplets.add(0, 0, 4);
    triplets.add(0, 1, 1);
    triplets.add(1, 0, 1);
    triplets.add(1, 1, 3);

    const A = new eig.SparseMatrix(2, 2, triplets);
    const b = new eig.Matrix([1, 2]);

    const chol = new eig.SimplicialCholesky(A);
    const x = chol.solve(b);

    // Exact solution: x_0 = 1/11 ≈ 0.090909, x_1 = 7/11 ≈ 0.636363
    expect(x.get(0, 0)).toBeCloseTo(1 / 11, 4);
    expect(x.get(1, 0)).toBeCloseTo(7 / 11, 4);
    eig.GC.flush();
  });

  it('solves linear systems with SparseLU', () => {
    // Solve non-symmetric system A * x = b
    // A = [[1, 2], [3, 4]], b = [5, 11] -> x = [1, 2]
    const triplets = new eig.TripletVector(4);
    triplets.add(0, 0, 1);
    triplets.add(0, 1, 2);
    triplets.add(1, 0, 3);
    triplets.add(1, 1, 4);

    const A = new eig.SparseMatrix(2, 2, triplets);
    const b = new eig.Matrix([5, 11]);

    const lu = new eig.SparseLU(A);
    const x = lu.solve(b);

    expect(x.get(0, 0)).toBeCloseTo(1, 4);
    expect(x.get(1, 0)).toBeCloseTo(2, 4);
    eig.GC.flush();
  });
});
