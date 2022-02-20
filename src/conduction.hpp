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

void temperature_heat_conduction(const Input &inp,
                                 const Material_library &mat_lib, 
                                 mfem::QuadratureFunction &power,
                                 mfem::ParGridFunction &temperature);

void conduction_error_analysis(mfem::Coefficient &coeff, 
                               mfem::ParGridFunction &sol);

void calc_thermal_conductivity(const Input &inp,
                               const Material_library &mat_lib,
                               const mfem::ParGridFunction &temperature,
                               mfem::QuadratureFunction &thermal_conductivity_quad);

#endif
