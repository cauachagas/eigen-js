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
  mlmatrix: { name: 'Ml Matrix', key: 'mlmatrix' },
  mathjs: { name: 'Math JS', key: 'mathjs' },
  linalg: { name: 'Linalg', key: 'linalg' },
  lalolib: { name: 'LaloLib', key: 'lalolib' },
};

/**
 * Benchmark Definitions
 */
export const benchmarkDefinitions = [
  {
    id: 'mat_mul',
    name: 'Matrix multiplication',
    description: 'Matrix multiplication performance test',
    params: { size: 100, iterations: 100 },
    supportedLibs: ['eig', 'mlmatrix', 'mathjs', 'linalg', 'lalolib'],
    codes: {
      eig: `const A = createMatrix('eig', size);
const B = createMatrix('eig', size);
for (let k = 0; k < iterations; k++) {
  A.matMul(B);
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
  }
];
