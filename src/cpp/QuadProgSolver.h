#ifndef QUADPROGSOLVER
#define QUADPROGSOLVER

#include <Eigen/Sparse>
#include "SparseMatrix.h"
#include "DenseMatrix.h"
#include "osqp.h"
#include <vector>

#define PRINT_MAT(name, M, size)     \
  printf("%s: [", name);             \
  for (int k = 0; k < (size); k++)   \
    printf("%.2f,", (double)(M)[k]); \
  printf("]\n");

class QuadProgSolver
{
  using T = Eigen::Triplet<double>;
  using DMD = DenseMatrix<double>;
  using SMD = SparseMatrix<double>;
  using QInt = OSQPInt;
  using QFloat = OSQPFloat;

public:
  // Minimize 0.5 xT.P.x + qT.x
  // Suject to l <= Ax <= u
  static DMD solve(SMD &P, DMD &q, SMD &A, DMD &l, DMD &u) {
    assert(P.rows() == P.cols() && "P must be a square matrix");
    assert(q.rows() == P.rows() && "q and P must have the same number of rows");
    assert(A.cols() == P.rows() && "A.cols must equal P.rows");
    assert(l.rows() == A.rows() && "l and A must have the same number of rows");
    assert(u.rows() == A.rows() && "u and A must have the same number of rows");


    QInt n = static_cast<QInt>(P.rows());
    QInt m = static_cast<QInt>(A.rows());
    OSQPCscMatrix* P_ = sparseToCSC(P);
    OSQPCscMatrix* A_ = sparseToCSC(A);
    QFloat* q_ = denseToArray(q);
    QFloat* l_ = denseToArray(l);
    QFloat* u_ = denseToArray(u);

    // Workspace settings
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    // settings.max_iter = 10;
    settings.verbose = 0;

    OSQPSolver* solver = nullptr;
    int exitflag = osqp_setup(&solver, P_, q_, A_, l_, u_, m, n, &settings);
    DMD x(n, 1);
    if (!exitflag && solver)
    {
      osqp_solve(solver);
      QFloat* xArr = solver->solution->x;
      for (QInt k = 0; k < n; k++)
        x.set(k, 0, xArr[k]);
    }
    osqp_cleanup(solver);
    // Free matrices
    OSQPCscMatrix_free(P_);
    OSQPCscMatrix_free(A_);
    free(q_);
    free(l_);
    free(u_);
    return x;
  }


  static void solveSparse()
  {
    TripletVector<double> tripletsP(3);
    tripletsP.add(0, 0, 4);
    tripletsP.add(0, 1, 1);
    tripletsP.add(1, 1, 2);
    SMD P(2, 2, &tripletsP);
    // P.data.setFromTriplets(triplets.begin(), triplets.end());

    TripletVector<double> tripletsA(4);
    tripletsA.add(0, 0, 1);
    tripletsA.add(0, 1, 1);
    tripletsA.add(1, 0, 1);
    tripletsA.add(2, 1, 1);
    SMD A(3, 2, &tripletsA);
    // A.data.setFromTriplets(triplets.begin(), triplets.end());

    // Get lists
    // const int nnz = mat.nonZeros();
    // PRINT_MAT("Values", mat.valuePtr(), nnz);
    // PRINT_MAT("Inner indices", mat.innerIndexPtr(), nnz);
    // const int nCols = mat.cols();
    // PRINT_MAT("Outer index ptr", mat.outerIndexPtr(), nCols + 1);

    // Build csc from matrix
    OSQPCscMatrix* P_csc = sparseToCSC(P);
    OSQPCscMatrix* A_csc = sparseToCSC(A);
    QFloat q[2] = {1.0, 1.0};
    QFloat l[3] = {1.0, 0.0, 0.0};
    QFloat u[3] = {1.0, 0.7, 0.7};
    QInt n = 2;
    QInt m = 3;

    // Workspace settings
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    // settings.max_iter = 10;
    settings.verbose = 0;

    OSQPSolver* solver = nullptr;
    int exitflag = osqp_setup(&solver, P_csc, q, A_csc, l, u, m, n, &settings);
    if (!exitflag)
    {
      osqp_solve(solver);
    }
    osqp_cleanup(solver);
    // Free matrices
    OSQPCscMatrix_free(P_csc);
    OSQPCscMatrix_free(A_csc);
  }

  static QFloat* denseToArray(DMD mat) {
    QFloat *arr = (QFloat *) malloc(mat.rows() * sizeof(QFloat));
    for (int k = 0; k < mat.rows(); k++) {
      arr[k] = mat.get(k, 0);
    }
    return arr;
  }

  static OSQPCscMatrix* sparseToCSC(SMD &mat)
  {
    return OSQPCscMatrix_new(
      static_cast<QInt>(mat.rows()),
      static_cast<QInt>(mat.cols()),
      static_cast<QInt>(mat.nonZeros()),
      reinterpret_cast<QFloat*>(mat.valuePtr()),
      reinterpret_cast<QInt*>(mat.innerIndexPtr()),
      reinterpret_cast<QInt*>(mat.outerIndexPtr())
    );
  }

  static void solveBasic()
  {
    // Load problem data
    // P = [[4, 1]
    //      [0, 2]]
    QFloat P_x[3] = {4.0, 1.0, 2.0};
    QInt P_nnz = 3;
    QInt P_i[3] = {0, 0, 1};
    QInt P_p[3] = {0, 1, 3};
    QFloat q[2] = {1.0, 1.0};
    // A = [[1 1
    //       1 0
    //       0 1]]
    QFloat A_x[4] = {1.0, 1.0, 1.0, 1.0};
    QInt A_nnz = 4;
    QInt A_i[4] = {0, 1, 0, 2};
    QInt A_p[3] = {0, 2, 4};
    QFloat l[3] = {1.0, 0.0, 0.0};
    QFloat u[3] = {1.0, 0.7, 0.7};
    QInt n = 2;
    QInt m = 3;

    // Exitflag
    int exitflag = 0;

    // Workspace structures
    OSQPSettings settings;
    OSQPCscMatrix* P_mat = OSQPCscMatrix_new(n, n, P_nnz, P_x, P_i, P_p);
    OSQPCscMatrix* A_mat = OSQPCscMatrix_new(m, n, A_nnz, A_x, A_i, A_p);

    // Define solver settings as default
    osqp_set_default_settings(&settings);
    settings.alpha = 1.0; // Change alpha parameter
    // settings.max_iter = 10;
    settings.verbose = 0;

    // Setup workspace
    OSQPSolver* solver = nullptr;
    exitflag = osqp_setup(&solver, P_mat, q, A_mat, l, u, m, n, &settings);

    // Solve Problem
    osqp_solve(solver);

    // Cleanup
    osqp_cleanup(solver);
    OSQPCscMatrix_free(P_mat);
    OSQPCscMatrix_free(A_mat);
    // return exitflag;
  }

  // private:
};

#endif // QUADPROGSOLVER