#include "fission_coefficient.hpp"

using namespace mfem;

constexpr int xs_temp_idx = 0;

double FissionCoefficient::Eval(ElementTransformation& T,
                                const IntegrationPoint& ip)
{

  const int attribute{T.Attribute - 1};

  const XSRecord& record{mat_lib.mat[attribute].rec[xs_temp_idx]};
  const std::vector<double>& nusf{record.nusf};

  if (!record.is_fiss) return 0.0;

  double sum{0.0};
  for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    sum += nusf[gprime] * phi(gprime).GetValue(T, ip);

  sum /= keff;

  const double ans{get_chi(T, ip) * sum};

  return ans;
}

double FissionCoefficient::get_chi(const ElementTransformation& T,
                                   const IntegrationPoint& ip) const
{

  if (mat_lib.rx_wide_chi) return mat_lib.chi[g];

  const int attribute{T.Attribute - 1};
  const int element_id{T.ElementNo};

  // use precalculated value
  if (flux_free_chi) return mat_lib.mat[attribute].rec[xs_temp_idx].chi[g];

  // collect flux spectrum
  std::vector<double> spectrum(mat_lib.ngroup());
  for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    spectrum[gprime] = phi(gprime).GetValue(element_id, ip);

  return mat_lib.mat[attribute].calculate_chi(spectrum, g, xs_temp_idx);
}
