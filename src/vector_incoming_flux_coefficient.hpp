#ifndef VECTOR_INCOMING_FLUX_COEFFICIENT_HPP
#define VECTOR_INCOMING_FLUX_COEFFICIENT_HPP

#include "mfem.hpp"
#include "quadrature.hpp"

class VectorIncomingFluxCoefficient final : public mfem::VectorCoefficient
{

public:

  VectorIncomingFluxCoefficient(const mfem::ParGridFunction &af, const Quadrature &q)
    : mfem::VectorCoefficient(q.nd()), angular_flux{af}, quad{q}
  {}

  virtual void Eval(mfem::Vector &v,
                    mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override;

private:

  const mfem::ParGridFunction &angular_flux;
  const Quadrature &quad;

};

#endif
