#include "incoming_flux_coefficient.hpp"

using namespace mfem;

IncomingFluxCoefficient::IncomingFluxCoefficient(int G, int N, const Result& af,
                                                 const Quadrature* const quad)
  : g{G}, n{N}, angular_flux{af}, q{quad}
{
}

double IncomingFluxCoefficient::Eval(ElementTransformation& T, const IntegrationPoint& ip)
{

  Vector nhat(3);
  nhat = 0.0;
  CalcOrtho(T.Jacobian(), nhat);
  nhat /= nhat.Norml2();

  auto weight{q->interpolate_reflection(n, nhat)};
  double incoming_flux{0.0};

  for (const std::pair<const int,const double> w : weight)
    incoming_flux += angular_flux(g, w.first).GetValue(T, ip) * w.second;

  return incoming_flux;
}
