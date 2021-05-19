#ifndef SOLVER_CHOICE
#define SOLVER_CHOICE

#include <string>

enum class Solver_choice
{
  invalid = -1,
  power_iteration,
  amgx_power_iteration,
  lobpcg,
  slepc_vector,
  amgx_vector,
  slepc_manual,
  epdo_power_iteration,
  sfent_power_iteration,
  saaf_power_iteration,
  saaf_angle_power_iteration,
  saaf_slepc_vector,
  saaf_true_slepc_vector,
  saaf_true_amgx_vector,
  saaf_split_slepc_vector,
  saaf_split_power_iteration,
  saaf_true_slepc_diffusion_vector,
  saaf_true_slepc_transport_vector,
  conduction_linear, // TODO move this into "analytic"
  conduction_radial // TODO move this into "analytic"
};

std::string solver_choice_enum_to_str(const Solver_choice& solver);
Solver_choice solver_choice_str_to_enum(const std::string& solver);

#endif
