#ifndef UPSCATTERING_COEFFICIENT
#define UPSCATTERING_COEFFICIENT

#include "mfem.hpp"

#include "material_library.hpp"
#include "result.hpp"

#include <vector>

class UpscatteringCoefficient : public mfem::Coefficient
{

public:

  UpscatteringCoefficient(int group,
                          const Material_library& mat_lib_lib,
                          const Result& scalar_flux_flux)
    : g{group}, mat_lib{mat_lib_lib}, scalar_flux{scalar_flux_flux}
  {
    is_upscatter = check_if_upscatter_present();
  }

  virtual double Eval(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip) override;

private:

  const int g;

  const Material_library& mat_lib;

  const Result& scalar_flux;

  std::vector<bool> is_upscatter;

  std::vector<bool> check_if_upscatter_present();

};

#endif
