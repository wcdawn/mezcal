#include "solver_choice.hpp"

using std::string;

string solver_choice_enum_to_str(const Solver_choice& solver)
{
  switch (solver)
  {
    case (Solver_choice::power_iteration): return "power_iteration";
    case (Solver_choice::lobpcg):          return "lobpcg";
    case (Solver_choice::slepc_vector):    return "slepc_vector";
    case (Solver_choice::slepc_manual):    return "slepc_manual";
    case (Solver_choice::epdo_power_iteration): return "epdo_power_iteration";
    case (Solver_choice::sfent_power_iteration): return "sfent_power_iteration";
    case (Solver_choice::saaf_power_iteration): return "saaf_power_iteration";
    case (Solver_choice::saaf_angle_power_iteration): return "saaf_angle_power_iteration";
    case (Solver_choice::saaf_slepc_vector): return "saaf_slepc_vector";
    case (Solver_choice::saaf_true_slepc_vector): return "saaf_true_slepc_vector";
    default: return "invalid";
  }
}

Solver_choice solver_choice_str_to_enum(const string& solver)
{
  if (solver == "power_iteration") return Solver_choice::power_iteration;
  if (solver == "lobpcg")          return Solver_choice::lobpcg;
  if (solver == "slepc_vector")    return Solver_choice::slepc_vector;
  if (solver == "slepc_manual")    return Solver_choice::slepc_manual;
  if (solver == "epdo_power_iteration") return Solver_choice::epdo_power_iteration;
  if (solver == "sfent_power_iteration") return Solver_choice::sfent_power_iteration;
  if (solver == "saaf_power_iteration") return Solver_choice::saaf_power_iteration;
  if (solver == "saaf_angle_power_iteration") return Solver_choice::saaf_angle_power_iteration;
  if (solver == "saaf_slepc_vector") return Solver_choice::saaf_slepc_vector;
  if (solver == "saaf_true_slepc_vector") return Solver_choice::saaf_true_slepc_vector;
  return Solver_choice::invalid;
}

