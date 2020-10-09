#include "incoming_flux_coefficient.hpp"

using namespace mfem;

IncomingFluxCoefficient::IncomingFluxCoefficient(int vd, int G, int N,
                                                 const Result& af,
                                                 const Quadrature* const quad)
  : VectorCoefficient{vd}, g{G}, n{N}, angular_flux{af}, q{quad}
{
}

void IncomingFluxCoefficient::Eval(Vector& V, ElementTransformation& T, const IntegrationPoint& ip)
{
  V = 0.0;
}
