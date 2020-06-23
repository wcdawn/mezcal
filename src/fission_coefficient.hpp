#ifndef FISSION_COEFFICIENT
#define FISSION_COEFFICIENT

#include "mfem.hpp"
#include "xslibrary.hpp"
#include <vector>

class FissionCoefficient : public mfem::Coefficient
{
  public:
    FissionCoefficient(int group, const double& init_keff, 
                       const XSLibrary& xslib, 
                       const std::vector<mfem::ParGridFunction>& flux)
      : g{group}, keff{init_keff}, xs{xslib}, phi{flux} {}

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
    const XSLibrary& xs;

    // Store a reference to the flux vector.
    const std::vector<mfem::ParGridFunction>& phi;
};

#endif
