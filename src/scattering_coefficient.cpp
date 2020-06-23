#include "scattering_coefficient.hpp"

// TODO remove this
#include <iostream>
using std::cout;
using std::endl;

using namespace mfem;

double ScatteringCoefficient::Eval(ElementTransformation& T,
                                   const IntegrationPoint& ip)
{
  const int attribute{T.Attribute - 1};
  const int element_id{T.ElementNo};

  const XSRecord& record{xs.rec[attribute]};
  const SparseMatrix& scat{record.sigma_s};

  constexpr bool dense_scattering{true};
  constexpr bool transpose_scattering{true};
  double sum{0.0};
  if (dense_scattering)
  {
    DenseMatrix scat_mat;
    scat.ToDenseMatrix(scat_mat);
    if (transpose_scattering) scat_mat.Transpose();
    for (int gprime{0}; gprime < xs.ngroup(); ++gprime)
    {
      if (gprime == g) continue;
      sum += scat_mat(gprime,g) * phi[gprime].GetValue(element_id, ip);
    }
  }
  else
  {
    Array<int> col_idx;
    Vector data;
    scat.GetRow(g, col_idx, data);

    for (int i{0}; i < col_idx.Size(); ++i)
    {
      const int gprime{col_idx[i]};
      if (gprime == g) continue;
      sum += data[i] * phi[gprime].GetValue(element_id, ip);
    }
  }

  return sum;
}
