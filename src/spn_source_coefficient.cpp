#include "spn_source_coefficient.hpp"

using namespace mfem;

constexpr int xs_temp_idx = 0;

void SPNSourceCoefficient::Eval(Vector& V, ElementTransformation& T, const IntegrationPoint& ip)
{

  const int attribute{T.Attribute - 1};
  const XSRecord& record{mat_lib.mat[attribute].rec[xs_temp_idx]};

  V.SetSize(dim);
  V = 0.0;

  if (n > 0)
  {
    phi(g, n-1).GetGradient(T, V);
    V *= -xmult_prev / record.sigma_r(g,2*n-1);
  }

  if ((n+1) < max_ord)
  {
    Vector tmp(dim);
    phi(g, n+1).GetGradient(T, tmp);
    V.Add(-xmult_next / record.sigma_r(g,2*n+1), tmp);
  }

}
