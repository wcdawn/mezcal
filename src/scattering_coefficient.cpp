#include "scattering_coefficient.hpp"

using namespace mfem;

double ScatteringCoefficient::Eval(ElementTransformation& T,
                                   const IntegrationPoint& ip)
{
  const int attribute{T.Attribute - 1};

  const XSRecord& record{mat_lib.mat[attribute].rec};
  const SparseMatrix& scat{record.sigma_s};

  Array<int> col_idx;
  Vector data;
  scat.GetRow(g, col_idx, data);

  double sum{0.0};
  for (int i{0}; i < col_idx.Size(); ++i)
  {
    const int gprime{col_idx[i]};
    if (gprime != g) sum += data[i] * phi(gprime).GetValue(T, ip);
  }

  return sum;
}
