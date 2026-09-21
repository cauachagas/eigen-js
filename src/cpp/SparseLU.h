#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include "DenseMatrix.h"

template <typename MatrixWrapType, typename MatrixType>
class SparseLU
{
public:
  SparseLU(const MatrixWrapType &m): m_(&m.data) {
    data = new Eigen::SparseLU<MatrixType>();
    data->compute(*m_);
  }

  SparseLU(const SparseLU<MatrixWrapType, MatrixType> &s) {
    m_ = s.m_;
    data = new Eigen::SparseLU<MatrixType>();
    data->compute(*m_);
  }

  ~SparseLU() {
    delete data;
  }

  SparseLU& operator=(const SparseLU& s) {
    if (this != &s) {
      if (data) {
        delete data;
      }
      m_ = s.m_;
      data = new Eigen::SparseLU<MatrixType>();
      data->compute(*m_);
    }
    return *this;
  }

  DenseMatrix<double> solve(DenseMatrix<double> &y) {
    return DenseMatrix<double>(data->solve(y.data));
  }

protected:
  Eigen::SparseLU<MatrixType> *data;
  const MatrixType *m_;
};
