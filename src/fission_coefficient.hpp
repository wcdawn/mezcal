#ifndef FISSION_COEFFICIENT
#define FISSION_COEFFICIENT

#include "mfem.hpp"
#include "xslibrary.hpp"
#include "material_library.hpp"
#include <vector>

class FissionCoefficient : public mfem::Coefficient
{
  public:
    FissionCoefficient(int group, const double& init_keff, 
                       const Material_library& mat_lib_lib,
                       const std::vector<mfem::ParGridFunction>& flux)
      : g{group}, keff{init_keff}, mat_lib{mat_lib_lib}, phi{flux} {}

    // void SetGroup(int gnew) { g = gnew; }

    virtual double Eval(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip) override;

  private:

    // Current working group.
    int g;

    // Store a reference to the current value of keff.
    const double& keff;

    // Store a reference to the cross section library.
    // We count on the fact that it does not go out of scope while we need it.
    const Material_library& mat_lib;

    // Store a reference to the flux vector.
    const std::vector<mfem::ParGridFunction>& phi;

    // Routine to calculate chi on-the-fly using th flux profile.
    double get_chi(const mfem::ElementTransformation& T, 
                   const mfem::IntegrationPoint& ip) const;
};

#endif
