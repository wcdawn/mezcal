#include "fission_coefficient.hpp"

using namespace mfem;

using std::vector;

double FissionCoefficient::Eval(ElementTransformation& T,
                                const IntegrationPoint& ip)
{
  const int attribute{T.Attribute - 1};

  const XSRecord& record{mat_lib.mat[attribute].rec};
  const vector<double>& nusf{record.nusf};

  if (!record.is_fiss) return 0.0;

  double sum{0.0};
  for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    sum += nusf[gprime] * phi(gprime).GetValue(T, ip);

  sum /= keff;

  return get_chi(T, ip) * sum;
}

double FissionCoefficient::get_chi(const ElementTransformation& T,
                                   const IntegrationPoint& ip) const
{
  const int attribute{T.Attribute - 1};
  const int element_id{T.ElementNo};

  // use precalculated value
  if (flux_free_chi) return mat_lib.mat[attribute].rec.chi[g];

  // collect flux spectrum
  vector<double> spectrum(mat_lib.ngroup());
  for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    spectrum[gprime] = phi(gprime).GetValue(element_id, ip);

  return mat_lib.mat[attribute].calculate_chi(spectrum, g);
}
