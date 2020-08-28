#include "vector_pwconst_coefficient.hpp"

using namespace mfem;

void VectorPWConstCoefficient::Eval(Vector& V, ElementTransformation& T,
                                    const IntegrationPoint& ip)
{
  const int att{T.Attribute};
  V = coefficients[att-1];
}
