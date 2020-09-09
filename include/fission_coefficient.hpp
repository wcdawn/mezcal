#ifndef FISSION_COEFFICIENT
#define FISSION_COEFFICIENT

#include "material_library.hpp"
#include "mfem.hpp"
#include "xslibrary.hpp"
#include "result.hpp"

#include <vector>

class FissionCoefficient : public mfem::Coefficient
{
public:
  FissionCoefficient(int group, const double& init_keff,
                     const Material_library& mat_lib_lib,
                     const Result& flux,
                     bool flux_free_chi_choice = false)
    : g{group}, keff{init_keff}, mat_lib{mat_lib_lib}, phi{flux},
      flux_free_chi{flux_free_chi_choice}
  {
  }

  virtual double Eval(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip) override;

private:
  // Current energy group.
  const int g;

  // Store a reference to the current value of keff.
  const double& keff;

  // Store a reference to the cross section library.
  // We count on the fact that it does not go out of scope while we need it.
  const Material_library& mat_lib;

  // Store a reference to the flux vector.
  const Result& phi;

  // option for whether to recalculate chi using flux
  // true = do not recalculate chi
  // false = recalculate chi using the local fluxes
  // default to false
  bool flux_free_chi;

  // Routine to calculate chi on-the-fly using th flux profile.
  double get_chi(const mfem::ElementTransformation& T,
                 const mfem::IntegrationPoint& ip) const;
};

#endif
