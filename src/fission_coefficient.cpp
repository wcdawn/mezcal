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

  // TODO this is a hack of chi for two group problems
  return (g == 0) ? sum : 0.0;
}
