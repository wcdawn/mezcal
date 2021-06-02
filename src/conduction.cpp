#include "conduction.hpp"

#include "thermal_conductivity_coefficient.hpp"

#include <iostream>
#include <unordered_set>

#include <cmath>

using namespace mfem;

double linear_source(const Vector& v)
{
  const double x{v(0)-0.2};
  constexpr double L{1.0};
  constexpr double q0{100.};
  return q0 * std::sin(M_PI * x / L);
}

double radial_source(const Vector& v)
{
  const double x{v(0)};
  const double y{v(1)};
  const double r{std::sqrt(std::pow(x,2) + std::pow(y,2))};

  constexpr double R_HP{0.5};
  constexpr double R_B{2.0};

  constexpr double q0{100.};
  return q0 * (R_B - r) / (R_B - R_HP);
}

double linear_solution(const Vector& v)
{
  const double x{v(0)-0.2};

  constexpr double L{1.0};

  constexpr double T_HP{400.};
  constexpr double T_B{300.};

  constexpr double k{0.1};
  constexpr double q0{100.};

  constexpr double c1{(T_B - T_HP) / L};
  constexpr double c2{T_HP};

  if (x < 0.0) return T_HP;
  return q0/k * std::pow(L/M_PI,2) * std::sin(M_PI * x / L) + c1*x + c2;
}

double radial_solution(const Vector& v)
{
  const double x{v(0)};
  const double y{v(1)};
  const double r{std::sqrt(std::pow(x,2) + std::pow(y,2))};

  constexpr double R_HP{0.5};
  constexpr double R_B{2.0};

  constexpr double T_HP{400.};
  constexpr double T_B{300.};

  constexpr double k{0.1};
  constexpr double q0{100.};

  constexpr double c1{1.0/std::log(R_HP/R_B) * (
    (T_HP-T_B)*(R_B-R_HP)*k/q0 - std::pow(R_HP,3)/9. + 0.25*R_B*std::pow(R_HP,2) - 5./36.*std::pow(R_B,3))};
  constexpr double c2{T_B*(R_B-R_HP)*k/q0 + 5./36.*std::pow(R_B,3) - c1*std::log(R_B)};

  if (r < R_HP) return T_HP;
  return q0/k / (R_B - R_HP) * (std::pow(r,3)/9. - 0.25*R_B*std::pow(r,2) + c1*std::log(r) + c2);
}

void temperature_heat_conduction(const Input& inp,
                                 const Material_library& mat_lib,
                                 const ParGridFunction& power,
                                 ParGridFunction& temperature)
{

  ParFiniteElementSpace* const parfespace{power.ParFESpace()};
  const ParMesh* const pmesh{parfespace->GetParMesh()};

  Array<int> ess_bdr, ess_tdof_list;
  ess_bdr.SetSize(pmesh->bdr_attributes.Max());

  /*
  // all boundaries are essential (all external boundaries for now)
  // if we want to solve a problem with reflective symmetry and impose
  // reflective bcs, would need to do something about this
  ess_bdr = 1;
  */

  // For now I'm allowing reflective (insulating) boundaries. This is nice for
  // running the vvercan model. There is a potential problem that this system
  // would have no solution if there is no essential boundary and also no heat
  // pipes but... just don't let that happen...
  for (int i{0}; (i < inp.boundary_conditions.Size()) && (i < ess_bdr.Size()); ++i)
    ess_bdr[i] = inp.boundary_conditions[i];

  parfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // TODO need some other method for this...
  // bwr 5 is poison rods
  // vvercan 4 is poison rods
  constexpr int heat_pipe_id{1};

  std::unordered_set<int> heat_pipe_set;
  for (int e{0}; e < parfespace->GetNE(); ++e)
  {
    if (parfespace->GetAttribute(e) == heat_pipe_id)
    {
      // found a heat pipe
      Array<int> dofs;
      parfespace->GetElementDofs(e, dofs);
      for (int i{0}; i < dofs.Size(); ++i)
      {
        const int ldof{parfespace->GetLocalTDofNumber(dofs[i])};
        if (ldof >= 0)
        {
          heat_pipe_set.insert(ldof);
        }
      }
    }
  }

  for (int val : heat_pipe_set)
    ess_tdof_list.Append(val);

  ConstantCoefficient temp_boundary_coeff{inp.temp_boundary}; // constant temperature of boundary
  ConstantCoefficient heat_pipe_coeff{inp.temp_heat_pipe}; // constant temperature of heat pipes
  temperature.ProjectCoefficient(heat_pipe_coeff); // heat pipe temperature is used as initial guess everywhere
  temperature.ProjectBdrCoefficient(temp_boundary_coeff, ess_bdr);

  ThermalConductivityCoefficient thermal_conductivity_coeff{mat_lib, &temperature};
  ConstantCoefficient analytic_coeff{0.1}; // used for verification problems

  ParBilinearForm a{parfespace};
  switch (inp.sol)
  {
    case (Solver_choice::conduction_linear):
    case (Solver_choice::conduction_radial):
      a.AddDomainIntegrator(new DiffusionIntegrator{analytic_coeff});
      break;
    default:
      a.AddDomainIntegrator(new DiffusionIntegrator{thermal_conductivity_coeff});
      break;
  }
  // For Robin bc, add BoundaryMassIntegrator

  FunctionCoefficient linear_source_coeff{linear_source};
  FunctionCoefficient radial_source_coeff{radial_source};
  GridFunctionCoefficient power_coeff{&power};

  ParLinearForm b{parfespace};
  switch (inp.sol)
  {
    case (Solver_choice::conduction_linear):
        b.AddDomainIntegrator(new DomainLFIntegrator{linear_source_coeff});
        break;
    case (Solver_choice::conduction_radial):
        b.AddDomainIntegrator(new DomainLFIntegrator{radial_source_coeff});
        break;
    default:
        b.AddDomainIntegrator(new DomainLFIntegrator{power_coeff});
        break;
  }

  // For Robin bc, add BoundaryLFIntegrator
  b.Assemble();

  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  HyprePCG cg(MPI_COMM_WORLD);
  cg.SetPreconditioner(amg);
  cg.SetTol(inp.tol.temp_inner_rel);
  //cg.SetAbsTol(inp.tol.temp_inner_abs); // TODO implement with new wrapper function i wrote
  cg.SetMaxIter(inp.tol.temp_max_inner);
  cg.SetPrintLevel(inp.hypre_print_level());

  double temp_conv{inp.tol.temp*1e2+1.0};
  for (int iter{0}; (iter < inp.tol.temp_max_outer) &&
      (temp_conv > inp.tol.temp); ++iter)
  {

    ParGridFunction temperature_old{temperature};

    a.Update();
    a.Assemble();
    a.Finalize();
    
    b.Assemble(); // necessary for bcs

    OperatorPtr A;
    Vector B, X;

    a.FormLinearSystem(ess_tdof_list, temperature, b, A, X, B);
    amg.SetOperator(*A);
    cg.SetOperator(*A);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, temperature);

    temperature_old -= temperature;
    temperature_old.HostRead(); // necessary to sync memory for GPU problems
    const double local_temp_conv{temperature_old.Normlinf()};
    MPI_Allreduce(&local_temp_conv, &temp_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (parfespace->GetMyRank() == 0)
      std::cout << "temperature_iter=" << iter << " temp_conv=" << temp_conv << std::endl;

  }

  switch (inp.sol)
  {
    case (Solver_choice::conduction_linear):
      {
        FunctionCoefficient solution{linear_solution};
        conduction_error_analysis(solution, temperature);
        break;
      }
    case (Solver_choice::conduction_radial):
      {
        FunctionCoefficient solution{radial_solution}; // TODO
        conduction_error_analysis(solution, temperature);
        break;
      }
    default:
      break; // do nothing
  }

  return;
}

void conduction_error_analysis(Coefficient& coeff, GridFunction& sol)
{
  const double l2error{sol.ComputeL2Error(coeff)};
  const double lmaxerror{sol.ComputeMaxError(coeff)};
  std::cout << "===TEMPERATURE_ERROR===" << std::endl;
  std::cout << "l2error=" << l2error << std::endl;
  std::cout << "lmaxerror=" << lmaxerror << std::endl;
  return;
}
