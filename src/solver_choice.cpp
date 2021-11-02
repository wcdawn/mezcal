#include "solver_choice.hpp"

using std::string;

string solver_choice_enum_to_str(const Solver_choice& solver)
{
  switch (solver)
  {
    case (Solver_choice::power_iteration): return "power_iteration";
    case (Solver_choice::amgx_power_iteration): return "amgx_power_iteration";
    case (Solver_choice::lobpcg):          return "lobpcg";
    case (Solver_choice::slepc_vector):    return "slepc_vector";
    case (Solver_choice::amgx_vector):    return "amgx_vector";
    case (Solver_choice::slepc_manual):    return "slepc_manual";
    case (Solver_choice::epdo_power_iteration): return "epdo_power_iteration";
    case (Solver_choice::sfent_power_iteration): return "sfent_power_iteration";
    case (Solver_choice::saaf_power_iteration): return "saaf_power_iteration";
    case (Solver_choice::saaf_angle_power_iteration): return "saaf_angle_power_iteration";
    case (Solver_choice::saaf_angle_nda_slepc): return "saaf_angle_nda_slepc";
    case (Solver_choice::saaf_angle_jacobi_power_iteration): return "saaf_angle_jacobi_power_iteration";
    case (Solver_choice::saaf_slepc_vector): return "saaf_slepc_vector";
    case (Solver_choice::saaf_true_slepc_vector): return "saaf_true_slepc_vector";
    case (Solver_choice::saaf_true_slepc_diffusion_vector): return "saaf_true_slepc_diffusion_vector";
    case (Solver_choice::saaf_true_slepc_transport_vector): return "saaf_true_slepc_transport_vector";
    case (Solver_choice::saaf_true_amgx_vector): return "saaf_true_amgx_vector";
    case (Solver_choice::saaf_split_slepc_vector): return "saaf_split_slepc_vector";
    case (Solver_choice::saaf_split_power_iteration): return "saaf_split_power_iteration";
    case (Solver_choice::saaf_self_power_iteration): return "saaf_self_power_iteration";
    case (Solver_choice::dg_spn_power_iteration): return "dg_spn_power_iteration";
    case (Solver_choice::dg_transport_power_iteration): return "dg_transport_power_iteration";
    case (Solver_choice::spn_partial_assembly_power_iteration): return "spn_partial_assembly_power_iteration";
    case (Solver_choice::conduction_linear): return "conduction_linear";
    case (Solver_choice::conduction_radial): return "conduction_radial";
    case (Solver_choice::conduction_var_cond): return "conduction_var_cond";
    case (Solver_choice::none): return "none";
    default: return "invalid";
  }
}

Solver_choice solver_choice_str_to_enum(const string& solver)
{
  if (solver == "power_iteration") return Solver_choice::power_iteration;
  if (solver == "amgx_power_iteration") return Solver_choice::amgx_power_iteration;
  if (solver == "lobpcg")          return Solver_choice::lobpcg;
  if (solver == "slepc_vector")    return Solver_choice::slepc_vector;
  if (solver == "amgx_vector")    return Solver_choice::amgx_vector;
  if (solver == "slepc_manual")    return Solver_choice::slepc_manual;
  if (solver == "epdo_power_iteration") return Solver_choice::epdo_power_iteration;
  if (solver == "sfent_power_iteration") return Solver_choice::sfent_power_iteration;
  if (solver == "saaf_power_iteration") return Solver_choice::saaf_power_iteration;
  if (solver == "saaf_angle_power_iteration") return Solver_choice::saaf_angle_power_iteration;
  if (solver == "saaf_angle_nda_slepc") return Solver_choice::saaf_angle_nda_slepc;
  if (solver == "saaf_angle_jacobi_power_iteration") return Solver_choice::saaf_angle_jacobi_power_iteration;
  if (solver == "saaf_slepc_vector") return Solver_choice::saaf_slepc_vector;
  if (solver == "saaf_true_slepc_vector") return Solver_choice::saaf_true_slepc_vector;
  if (solver == "saaf_true_slepc_diffusion_vector") return Solver_choice::saaf_true_slepc_diffusion_vector;
  if (solver == "saaf_true_slepc_transport_vector") return Solver_choice::saaf_true_slepc_transport_vector;
  if (solver == "saaf_true_amgx_vector") return Solver_choice::saaf_true_amgx_vector;
  if (solver == "saaf_split_slepc_vector") return Solver_choice::saaf_split_slepc_vector;
  if (solver == "saaf_split_power_iteration") return Solver_choice::saaf_split_power_iteration;
  if (solver == "saaf_self_power_iteration") return Solver_choice::saaf_self_power_iteration;
  if (solver == "dg_spn_power_iteration") return Solver_choice::dg_spn_power_iteration;
  if (solver == "dg_transport_power_iteration") return Solver_choice::dg_transport_power_iteration;
  if (solver == "spn_partial_assembly_power_iteration") return Solver_choice::spn_partial_assembly_power_iteration;
  if (solver == "conduction_linear") return Solver_choice::conduction_linear;
  if (solver == "conduction_radial") return Solver_choice::conduction_radial;
  if (solver == "conduction_var_cond") return Solver_choice::conduction_var_cond;
  if (solver == "none") return Solver_choice::none;
  return Solver_choice::invalid;
}

