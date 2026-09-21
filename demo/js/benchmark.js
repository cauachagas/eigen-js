/**
 * Pseudo-random generator for reproducible benchmark data
 */
class Random {
  constructor(seed = 1928473) {
    this.seed = seed;
  }

  get() {
    const x = Math.sin(this.seed++) * 10000;
    return x - Math.floor(x);
  }
}

export function getRandomArray2D(size, seed = 1928473) {
  const r = new Random(seed);
  const arr = new Array(size);
  for (let i = 0; i < size; i++) {
    const row = new Array(size);
    for (let j = 0; j < size; j++) {
      row[j] = r.get();
    }
    arr[i] = row;
  }
  return arr;
}

/**
 * Matrix factory across libraries
 */
export function createMatrix(type, size, array2D = null) {
  const array = array2D || getRandomArray2D(size);
  switch (type) {
    case 'eig': {
      if (!window.eig?.Matrix) throw new Error('Eigen.js is not loaded or ready.');
      return new window.eig.Matrix(array);
    }
    case 'mlmatrix': {
      const Matrix = window.Matrix || window.mlMatrix?.Matrix || window.mlMatrix?.default || window.MLMatrix?.Matrix || window.MLMatrix;
      if (!Matrix) throw new Error('ML Matrix library not loaded.');
      return new Matrix(array);
    }
    case 'mathjs': {
      if (!window.math?.matrix) throw new Error('Math.js library not loaded.');
      return window.math.matrix(array);
    }
    case 'linalg': {
      const linalg = typeof window.linearAlgebra === 'function' ? window.linearAlgebra() : (window.linalg || null);
      if (!linalg?.Matrix) throw new Error('Linear-algebra library not loaded.');
      return new linalg.Matrix(array);
    }
    case 'lalolib': {
      if (!window.lalolib?.array2mat) throw new Error('LaloLib library not loaded.');
      return window.lalolib.array2mat(array);
    }
    default:
      throw new Error(`Unknown matrix type: ${type}`);
  }
}

/**
 * Available libraries metadata
 */
export const libraries = {
  eig: { name: 'Eigen JS', key: 'eig' },
  openblas: { name: 'OpenBLAS (WASM)', key: 'openblas' },
  suitesparse: { name: 'SuiteSparse (WASM)', key: 'suitesparse' },
  mlmatrix: { name: 'Ml Matrix', key: 'mlmatrix' },
  mathjs: { name: 'Math JS', key: 'mathjs' },
  linalg: { name: 'Linalg', key: 'linalg' },
  lalolib: { name: 'LaloLib', key: 'lalolib' },
};

/**
 * Generate 1D Poisson tridiagonal symmetric positive-definite sparse matrix
 * A = [ 4 -1  0 ... ]
 *     [-1  4 -1 ... ]
 *     [ 0 -1  4 ... ]
 * in Compressed Sparse Column (CSC) format and rhs b = [1, 1, ..., 1]^T
 */
export function generateSparsePoissonProblem(n) {
  const ap = new Int32Array(n + 1);
  const aiList = [];
  const axList = [];

  let count = 0;
  for (let col = 0; col < n; col++) {
    ap[col] = count;
    if (col > 0) {
      aiList.push(col - 1);
      axList.push(-1.0);
      count++;
    }
    aiList.push(col);
    axList.push(4.0);
    count++;
    if (col < n - 1) {
      aiList.push(col + 1);
      axList.push(-1.0);
      count++;
    }
  }
  ap[n] = count;

  const ai = Int32Array.from(aiList);
  const ax = Float64Array.from(axList);
  const az = new Float64Array(count);
  const bx = new Float64Array(n).fill(1.0);
  const bz = new Float64Array(n);

  return { nRow: n, nCol: n, nnz: count, ap, ai, ax, az, bx, bz };
}

/**
 * Benchmark Definitions
 */
export const benchmarkDefinitions = [
  {
    id: 'mat_mul',
    name: 'Matrix multiplication',
    description: 'Matrix multiplication performance test',
    params: { size: 100, iterations: 100 },
    supportedLibs: ['eig', 'openblas', 'mlmatrix', 'mathjs', 'linalg', 'lalolib'],
    codes: {
      eig: `const A = createMatrix('eig', size);
const B = createMatrix('eig', size);
for (let k = 0; k < iterations; k++) {
  A.matMul(B);
}`,
      openblas: `// OpenBLAS CBLAS Level 3 DGEMM in WebAssembly
for (let k = 0; k < iterations; k++) {
  openBlas.callTest('dgemm');
}`,
      mlmatrix: `const A = createMatrix('mlmatrix', size);
const B = createMatrix('mlmatrix', size);
for (let k = 0; k < iterations; k++) {
  A.mmul(B);
}`,
      mathjs: `const A = createMatrix('mathjs', size);
const B = createMatrix('mathjs', size);
for (let k = 0; k < iterations; k++) {
  math.multiply(A, B);
}`,
      linalg: `const A = createMatrix('linalg', size);
const B = createMatrix('linalg', size);
for (let k = 0; k < iterations; k++) {
  A.dot(B);
}`,
      lalolib: `const A = createMatrix('lalolib', size);
const B = createMatrix('lalolib', size);
for (let k = 0; k < iterations; k++) {
  lalolib.mul(A, B);
}`
    },
    run: (type, size, iterations) => {
      if (type === 'openblas') {
        if (!window.openBlas?.callTest) throw new Error('OpenBLAS WebAssembly is not loaded or ready.');
        const start = performance.now();
        for (let k = 0; k < iterations; k++) {
          window.openBlas.callTest('dgemm');
        }
        return performance.now() - start;
      }

      const A = createMatrix(type, size);
      const B = createMatrix(type, size);
      const start = performance.now();

      if (type === 'eig') {
        for (let k = 0; k < iterations; k++) {
          A.matMul(B);
        }
        const elapsed = performance.now() - start;
        window.eig?.GC?.flush();
        return elapsed;
      } else if (type === 'mlmatrix') {
        for (let k = 0; k < iterations; k++) {
          A.mmul(B);
        }
        return performance.now() - start;
      } else if (type === 'mathjs') {
        for (let k = 0; k < iterations; k++) {
          window.math.multiply(A, B);
        }
        return performance.now() - start;
      } else if (type === 'linalg') {
        for (let k = 0; k < iterations; k++) {
          A.dot(B);
        }
        return performance.now() - start;
      } else if (type === 'lalolib') {
        for (let k = 0; k < iterations; k++) {
          window.lalolib.mul(A, B);
        }
        return performance.now() - start;
      }
    }
  },
  {
    id: 'mat_inv',
    name: 'Matrix inversion',
    description: 'Matrix inversion performance test',
    params: { size: 100, iterations: 20 },
    supportedLibs: ['eig', 'mlmatrix', 'mathjs', 'lalolib'],
    codes: {
      eig: `const A = createMatrix('eig', size);
for (let k = 0; k < iterations; k++) {
  A.inverse();
}`,
      mlmatrix: `const A = createMatrix('mlmatrix', size);
const inv = window.mlMatrix?.inverse || window.MLMatrix?.inverse;
for (let k = 0; k < iterations; k++) {
  inv(A);
}`,
      mathjs: `const A = createMatrix('mathjs', size);
for (let k = 0; k < iterations; k++) {
  math.inv(A);
}`,
      lalolib: `const A = createMatrix('lalolib', size);
for (let k = 0; k < iterations; k++) {
  lalolib.inv(A);
}`
    },
    run: (type, size, iterations) => {
      const A = createMatrix(type, size);
      const start = performance.now();

      if (type === 'eig') {
        for (let k = 0; k < iterations; k++) {
          A.inverse();
        }
        const elapsed = performance.now() - start;
        window.eig?.GC?.flush();
        return elapsed;
      } else if (type === 'mlmatrix') {
        const inv = window.inverse || window.Matrix?.inverse || window.mlMatrix?.inverse || window.MLMatrix?.inverse;
        for (let k = 0; k < iterations; k++) {
          inv ? inv(A) : A.inverse();
        }
        return performance.now() - start;
      } else if (type === 'mathjs') {
        for (let k = 0; k < iterations; k++) {
          window.math.inv(A);
        }
        return performance.now() - start;
      } else if (type === 'lalolib') {
        for (let k = 0; k < iterations; k++) {
          window.lalolib.inv(A);
        }
        return performance.now() - start;
      }
    }
  },
  {
    id: 'mat_svd',
    name: 'Singular value decomposition',
    description: 'Singular value decomposition performance test',
    params: { size: 100, iterations: 20 },
    supportedLibs: ['eig', 'lalolib'],
    codes: {
      eig: `const A = createMatrix('eig', size);
for (let k = 0; k < iterations; k++) {
  eig.Decompositions.svd(A, true);
}`,
      lalolib: `const A = createMatrix('lalolib', size);
for (let k = 0; k < iterations; k++) {
  lalolib.svd(A, "thin");
}`
    },
    run: (type, size, iterations) => {
      const A = createMatrix(type, size);
      const start = performance.now();

      if (type === 'eig') {
        for (let k = 0; k < iterations; k++) {
          window.eig.Decompositions.svd(A, true);
        }
        const elapsed = performance.now() - start;
        window.eig?.GC?.flush();
        return elapsed;
      } else if (type === 'lalolib') {
        for (let k = 0; k < iterations; k++) {
          window.lalolib.svd(A, "thin");
        }
        return performance.now() - start;
      }
    }
  },
  {
    id: 'sparse_solve',
    name: 'Sparse linear system (A · x = b)',
    description: 'Solves sparse linear system A · x = b using WebAssembly sparse direct LU factorization (Eigen SparseLU vs SuiteSparse UMFPACK)',
    params: { size: 100, iterations: 50 },
    supportedLibs: ['eig', 'suitesparse'],
    codes: {
      eig: `// Eigen.js (WASM) - SparseMatrix + SparseLU Solver
const A = createSparsePoissonMatrix(size);
const b = createVector(size);
for (let k = 0; k < iterations; k++) {
  const lu = new eig.SparseLU(A);
  const x = lu.solve(b);
}`,
      suitesparse: `// SuiteSparse (WASM) - UMFPACK Sparse LU Solver
const problem = createCscSparseProblem(size);
for (let k = 0; k < iterations; k++) {
  const res = umfpack.solveComplexSystem(problem);
}`
    },
    run: (type, size, iterations) => {
      const problem = generateSparsePoissonProblem(size);
      const start = performance.now();

      if (type === 'eig') {
        if (!window.eig?.SparseMatrix) throw new Error('Eigen.js is not loaded or ready.');
        const triplets = new window.eig.TripletVector(problem.nnz);
        for (let col = 0; col < problem.nCol; col++) {
          for (let p = problem.ap[col]; p < problem.ap[col + 1]; p++) {
            triplets.add(problem.ai[p], col, problem.ax[p]);
          }
        }
        const A = new window.eig.SparseMatrix(problem.nRow, problem.nCol, triplets);
        const b = new window.eig.Matrix(Array.from(problem.bx).map(v => [v]));

        for (let k = 0; k < iterations; k++) {
          const lu = new window.eig.SparseLU(A);
          const x = lu.solve(b);
        }
        const elapsed = performance.now() - start;
        window.eig?.GC?.flush();
        return elapsed;
      } else if (type === 'suitesparse') {
        if (!window.umfpack?.solveComplexSystem) throw new Error('SuiteSparse (UMFPACK) is not loaded or ready.');
        for (let k = 0; k < iterations; k++) {
          window.umfpack.solveComplexSystem(problem);
        }
        return performance.now() - start;
      }
    }
  }
];
