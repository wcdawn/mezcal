#ifndef MULTIPHYSICS_HPP
#define MULTIPHYSICS_HPP

#include "mfem.hpp"
#include "solver_saaf_angle_power_iteration.hpp"
#include "conduction.hpp"

void multiphysics(const int state,
                  const Quadrature * const quad,
                  Timer &time,
                  Input &inp,
                  const Material_library &mat_lib,
                  std::vector<std::unique_ptr<mfem::ParGridFunction>> &angular_flux,
                  std::vector<std::unique_ptr<mfem::ParGridFunction>> &scalar_flux,
                  double &keff,
                  mfem::ParGridFunction &power,
                  mfem::ParGridFunction &temperature);


#endif
