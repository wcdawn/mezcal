#ifndef NDA_BOUNDARY_FACTOR_COEFFICIENT_HPP
#define NDA_BOUNDARY_FACTOR_COEFFICIENT_HPP

#include "mfem.hpp"

#include "quadrature.hpp"
#include "material_library.hpp"

class NDABoundaryFactorCoefficient final : public mfem::VectorCoefficient
{

public:

  NDABoundaryFactorCoefficient(const mfem::ParGridFunction &scalar_flux_in, 
                               const mfem::ParGridFunction &current_in,
                               const mfem::ParGridFunction &angular_flux_in,
                               const Quadrature * const quad_in,
                               const Material_library &mat_lib_in)
    : VectorCoefficient{current_in.VectorDim()}, scalar_flux{scalar_flux_in},
    current{current_in}, angular_flux{angular_flux_in}, quad{quad_in},
    mat_lib{mat_lib_in}
  {}

  void Eval(mfem::Vector &v,
            mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;

private:

  const mfem::ParGridFunction &scalar_flux;
  const mfem::ParGridFunction &current;
  const mfem::ParGridFunction &angular_flux;
  const Quadrature * const quad;
  const Material_library &mat_lib;

};

#endif
