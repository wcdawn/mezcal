#ifndef THERMAL_CONDUCTIVITY_COEFFICIENT_HPP
#define THERMAL_CONDUCTIVITY_COEFFICIENT_HPP

#include "material_library.hpp"

#include "mfem.hpp"

class ThermalConductivityCoefficient final : public mfem::Coefficient
{
public:
  ThermalConductivityCoefficient(const Material_library& mat_lib_lib,
                                 const mfem::GridFunction* const temp_in)
    : mat_lib{mat_lib_lib}, temp{temp_in}
  {}
  double Eval(mfem::ElementTransformation& T, 
              const mfem::IntegrationPoint& ip) override;
private:
  const Material_library& mat_lib;
  const mfem::GridFunction* const temp;
};

#endif
