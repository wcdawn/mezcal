#ifndef SCATTERING_COEFFICIENT
#define SCATTERING_COEFFICIENT

#include "mfem.hpp"
#include "xslibrary.hpp"
#include <vector>

class ScatteringCoefficient : public mfem::Coefficient
{
  public:
    ScatteringCoefficient(int group, const XSLibrary& xslib, 
                          const std::vector<mfem::ParGridFunction>& flux) 
      : g{group}, xs{xslib}, phi{flux} {}

    // void SetGroup(int gnew) { g = gnew; }

    virtual double Eval(mfem::ElementTransformation& T, 
                        const mfem::IntegrationPoint& ip) override;

  private:
  
    // Current working group.
    int g;

    // Store a reference to the cross section library.
    // We count on the fact that it does not go out of scope while we need it.
    const XSLibrary& xs; 

    const std::vector<mfem::ParGridFunction>& phi;
};

#endif
