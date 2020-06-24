#include "fission_coefficient.hpp"

using namespace mfem;

using std::vector;

double FissionCoefficient::Eval(ElementTransformation& T,
                                const IntegrationPoint& ip)
{
  const int attribute{T.Attribute - 1};
  const int element_id{T.ElementNo};

  const XSRecord& record{xs.rec[attribute]};
  const vector<double>& nusf{record.nusf};

  double sum{0.0};

  for (int gprime{0}; gprime < xs.ngroup(); ++gprime)
    sum += nusf[gprime] * phi[gprime].GetValue(element_id, ip);

  sum /= keff;

  return get_chi(T, ip) * sum;
}

// TODO do not currently support multiple isotopes at a single position
double FissionCoefficient::get_chi(const ElementTransformation& T,
                                   const IntegrationPoint& ip) const
{
  const int attribute{T.Attribute - 1};
  const int element_id{T.ElementNo};
  const XSRecord& record{xs.rec[attribute]};
  return record.chi[g];
}
