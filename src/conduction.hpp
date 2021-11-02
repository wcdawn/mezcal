#ifndef CONDUCTION_HPP
#define CONDUCTION_HPP

#include "material_library.hpp"
#include "input.hpp"

#include "mfem.hpp"

double linear_source(const mfem::Vector& v);
double linear_solution(const mfem::Vector& v);

double radial_source(const mfem::Vector& v);
double radial_solution(const mfem::Vector& v);

double var_cond_source(const mfem::Vector& v);
double var_cond_solution(const mfem::Vector& v);

class Var_cond_conductivity final : public mfem::Coefficient
{
public:
  Var_cond_conductivity(const mfem::GridFunction &temp_gf) : temp{temp_gf} {}
  double Eval (mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip)
  {
    return (constant) ? kbar : (k0 - alpha*temp.GetValue(T, ip));
  }
private:
  const mfem::GridFunction &temp;
  static constexpr bool constant{false};
  static constexpr double k0{10.0};
  static constexpr double alpha{1e-2};
  static constexpr double kbar{5.100359646753139};
};

void temperature_heat_conduction(const Input& inp,
                                 const Material_library& mat_lib, 
                                 const mfem::ParGridFunction& power,
                                 mfem::ParGridFunction& temperature);

void conduction_error_analysis(mfem::Coefficient& coeff, 
                               mfem::GridFunction& sol);

#endif
