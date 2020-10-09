#ifndef SCATTERING_COEFFICIENT
#define SCATTERING_COEFFICIENT

#include "material_library.hpp"
#include "mfem.hpp"
#include "xslibrary.hpp"
#include "result.hpp"

#include <vector>

class ScatteringCoefficient : public mfem::Coefficient
{
public:
  ScatteringCoefficient(int group, const Material_library& mat_lib_lib,
                        const Result& flux, bool include_self_scatter=false)
    : g{group}, mat_lib{mat_lib_lib}, phi{flux}, include_self{include_self_scatter}
  {
  }

  virtual double Eval(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip) override;

private:
  // Current working group.
  const int g;

  // Store a reference to the cross section library.
  // We count on the fact that it does not go out of scope while we need it.
  const Material_library& mat_lib;

  const Result& phi;

  const bool include_self;
};

#endif
