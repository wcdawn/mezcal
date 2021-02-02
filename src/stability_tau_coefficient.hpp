#ifndef STABILITY_TAU_COEFFICIENT
#define STABILITY_TAU_COEFFICIENT

#include "material_library.hpp"

#include "mfem.hpp"

class StabilityTauCoefficient final : public mfem::Coefficient
{

public:

  StabilityTauCoefficient(mfem::Mesh* const m, int group, const Material_library& mat_lib_lib)
    : mesh{m}, g{group}, mat_lib{mat_lib_lib}
  {}

  virtual double Eval(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip) override;

private:

  mfem::Mesh* const mesh;

  int g;

  const Material_library& mat_lib;

};

#endif
