#include "thermal_conductivity_coefficient.hpp"

using namespace mfem;

double ThermalConductivityCoefficient::Eval(ElementTransformation& T, 
                                            const IntegrationPoint& ip)
{
  const int mat_num{T.Attribute-1};
  const double temperature{temp->GetValue(T, ip)};
  const double k{mat_lib.mat[mat_num].thermal_conductivity(temperature)*1e-2};
  return k; // return [W/cm/K]
}
