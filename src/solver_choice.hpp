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
  saaf_true_slepc_vector
};

std::string solver_choice_enum_to_str(const Solver_choice& solver);
Solver_choice solver_choice_str_to_enum(const std::string& solver);

#endif
