#include "vector_incoming_flux_coefficient.hpp"

using namespace mfem;

void VectorIncomingFluxCoefficient::Eval(Vector &v,
                                         ElementTransformation &T,
                                         const IntegrationPoint &ip)
{

  T.SetIntPoint(&ip);

  Vector nhat(T.GetSpaceDim());
  nhat = 0.0;
  CalcOrtho(T.Jacobian(), nhat);
  nhat /= nhat.Norml2();

  v.SetSize(quad.nd());
  v = 0.0;

  Vector af(quad.nd());
  angular_flux.GetVectorValue(T, ip, af);

  for (int n{0}; n < quad.nd(); ++n)
  {
    auto weight{quad.interpolate_reflection(n, nhat)};
    for (const std::pair<const int, const double> w : weight)
      v(n) += af(w.first) * w.second;
  }

  /*
  // Alternatively, just use the single "nearest" angle.
  for (int n{0}; n < quad.nd(); ++n)
  {
    auto weight{quad.interpolate_reflection(n, nhat)};
    int max_idx{-1};
    double max_weight{0.0};
    for (const std::pair<const int, const double> w : weight)
    {
      if (w.second > max_weight)
      {
        max_weight = w.second;
        max_idx = w.first;
      }
    }
    v(n) = af(max_idx);
  }
  */
  
  return;
}
