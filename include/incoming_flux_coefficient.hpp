#ifndef INCOMING_FLUX_COEFFICIENT
#define INCOMING_FLUX_COEFFICIENT

#include "mfem.hpp"

#include "result.hpp"
#include "quadrature.hpp"

class IncomingFluxCoefficient final : public mfem::VectorCoefficient
{

public:

  IncomingFluxCoefficient(int vd, int G, int N, const Result& af,
                          const Quadrature* const quad);

  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

private:

  const int g, n;
  const Result& angular_flux;
  const Quadrature* const q;

  // TODO we will need to implement caching to not do the angle calculations
  // every time
  bool cached{false};

};

#endif
