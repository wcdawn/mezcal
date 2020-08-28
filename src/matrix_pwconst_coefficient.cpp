#include "matrix_pwconst_coefficient.hpp"

using namespace mfem;

void MatrixPWConstCoefficient::Eval(DenseMatrix& K, ElementTransformation& T,
                                    const IntegrationPoint& ip)
{
  const int att{T.Attribute};
  K = coefficients[att-1];
}
