#ifndef INCOMING_FLUX_COEFFICIENT
#define INCOMING_FLUX_COEFFICIENT

#include "mfem.hpp"

#include "result.hpp"
#include "quadrature.hpp"

class IncomingFluxCoefficient final : public mfem::Coefficient
{

public:

  IncomingFluxCoefficient(int G, int N, const Result& af, const Quadrature* const quad)
    : g{G}, n{N}, angular_flux{af}, q{quad}
  {}

  virtual double Eval(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip) override;

private:

  const int g, n;
  const Result& angular_flux;
  const Quadrature* const q;

};

#endif
