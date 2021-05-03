#ifndef DOWNSCATTERING_COEFFICIENT
#define DOWNSCATTERING_COEFFICIENT

#include "mfem.hpp"

#include "material_library.hpp"
#include "result.hpp"

class DownscatteringCoefficient : public mfem::Coefficient
{

public:

  DownscatteringCoefficient(int group,
                            const Material_library& mat_lib_lib,
                            const Result& scalar_flux_flux)
    : g{group}, mat_lib{mat_lib_lib}, scalar_flux{scalar_flux_flux}
  {
  }

  virtual double Eval(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip) override;

private:

  const int g;

  const Material_library& mat_lib;

  const Result& scalar_flux;

};

#endif
