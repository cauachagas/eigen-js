import { describe, it, expect, beforeAll } from 'vitest';
import eig from '../dist/index.mjs';

describe('Solvers and QuadProgSolver', () => {
  beforeAll(async () => {
    await eig.ready;
  });

  it('computes eigenvalues and eigenvectors with Solvers.eigenSolve', () => {
    const A = new eig.Matrix([
      [2, 1],
      [1, 2]
    ]);
    const res = eig.Solvers.eigenSolve(A);
    expect(res.info).toBe(eig.ComputationInfo.Success);
    // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    const ev1 = res.eigenvalues.get(0, 0);
    const ev2 = res.eigenvalues.get(1, 0);
    const sorted = [ev1.real(), ev2.real()].sort();
    expect(sorted[0]).toBeCloseTo(1);
    expect(sorted[1]).toBeCloseTo(3);
    eig.GC.flush();
  });

  it('solves quadratic program with QuadProgSolver', () => {
    let tripletsP = new eig.TripletVector(3);
    tripletsP.add(0, 0, 4);
    tripletsP.add(0, 1, 1);
    tripletsP.add(1, 1, 2);
    const P = new eig.SparseMatrix(2, 2, tripletsP);

    let tripletsA = new eig.TripletVector(4);
    tripletsA.add(0, 0, 1);
    tripletsA.add(0, 1, 1);
    tripletsA.add(1, 0, 1);
    tripletsA.add(2, 1, 1);
    const A = new eig.SparseMatrix(3, 2, tripletsA);

    const q = new eig.Matrix([1, 1]);
    const l = new eig.Matrix([1, 0, 0]);
    const u = new eig.Matrix([1, 0.7, 0.7]);

    const x = eig.QuadProgSolver.solve(P, q, A, l, u);
    expect(x.rows()).toBe(2);
    expect(x.cols()).toBe(1);
    // Optimal solution x satisfies bounds
    expect(x.get(0, 0)).toBeGreaterThanOrEqual(0);
    expect(x.get(1, 0)).toBeGreaterThanOrEqual(0);
    eig.GC.flush();
  });
});
