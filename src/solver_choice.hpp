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
  saaf_efficient_power_iteration,
  saaf_angle_jacobi_power_iteration,
  saaf_angle_nda_slepc,
  saaf_slepc_vector,
  saaf_true_slepc_vector,
  saaf_true_amgx_vector,
  saaf_split_slepc_vector,
  saaf_split_power_iteration,
  saaf_self_power_iteration,
  saaf_true_slepc_diffusion_vector,
  saaf_true_slepc_transport_vector,
  dg_spn_power_iteration,
  dg_transport_power_iteration,
  spn_partial_assembly_power_iteration,
  conduction_linear, // TODO move this into "analytic"
  conduction_radial, // TODO move this into "analytic"
  conduction_var_cond,
  none // do absolutely nothing
};

std::string solver_choice_enum_to_str(const Solver_choice& solver);
Solver_choice solver_choice_str_to_enum(const std::string& solver);

#endif
