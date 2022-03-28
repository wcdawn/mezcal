#include "conduction.hpp"

#include "solver_choice.hpp"
#include "thermal_conductivity_coefficient.hpp"

#include <iostream>

#include <cmath>

using namespace mfem;

// this is so that the routines are called only once
// there seems to be a leak somewhere in the temperature update
// but i'm not sure where
namespace conduction_geom_data
{

  Array<int> ess_bdr, ess_tdof_list;
  bool first = true;

  void set_geometry(ParFiniteElementSpace *parfespace,
                    const Array<int> &bc_arr,
                    const int heat_pipe_id)
  {

    if (!first)
      return;

    const ParMesh * const pmesh = parfespace->GetParMesh();

    ess_bdr.SetSize(pmesh->bdr_attributes.Max());

    for (int i = 0; (i < bc_arr.Size()) && (i < ess_bdr.Size()); ++i)
      ess_bdr[i] = bc_arr[i];

    parfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    Array<int> ess_dofs_hp, true_ess_dofs_hp;
    ess_dofs_hp.SetSize(parfespace->GetVSize());
    ess_dofs_hp = 0;

    for (int e = 0; e < parfespace->GetNE(); ++e)
    {
      if (parfespace->GetAttribute(e) == heat_pipe_id)
      {
        // found a heat pipe
        Array<int> dofs;
        parfespace->GetElementDofs(e, dofs);
        for (int i = 0; i < dofs.Size(); ++i)
          ess_dofs_hp[dofs[i]] = -1;
      }
    }

    parfespace->Synchronize(ess_dofs_hp);

    parfespace->GetRestrictionMatrix()->BooleanMult(ess_dofs_hp, true_ess_dofs_hp);

    Array<int> ess_tdof_list_hp;
    ParFiniteElementSpace::MarkerToList(true_ess_dofs_hp, ess_tdof_list_hp);

    for (int i = 0; i < ess_tdof_list_hp.Size(); ++i)
      ess_tdof_list.Append(ess_tdof_list_hp[i]);

    first = false;
    return;
  }

  void initialize_temperature(const double temp_boundary,
                              const double temp_hp,
                              ParGridFunction &temperature)
  {
    temperature = temp_hp;
    ConstantCoefficient temp_boundary_coeff{temp_boundary};
    temperature.ProjectBdrCoefficient(temp_boundary_coeff, ess_bdr);
    return;
  }

}

double linear_source(const Vector &v)
{
  const double x{v(0)-0.2};
  constexpr double L{1.0};
  constexpr double q0{100.};
  return q0 * std::sin(M_PI * x / L);
}

double radial_source(const Vector &v)
{
  const double x{v(0)};
  const double y{v(1)};
  const double r{std::sqrt(std::pow(x,2) + std::pow(y,2))};

  constexpr double R_HP{0.5};
  constexpr double R_B{2.0};

  constexpr double q0{100.};
  return q0 * (R_B - r) / (R_B - R_HP);
}

double var_cond_source(const Vector &v)
{
  const double x{v(0)-0.2};
  constexpr double L{5.0};
  constexpr double q0{200.0};
  return q0*std::cos(0.5*M_PI*x/L);
}

double var_cond_radial_source(const Vector &v)
{
  const double r{std::sqrt(std::pow(v(0),2) + std::pow(v(1),2))};
  constexpr double R{5.0};
  constexpr double q0{400.0};
  return q0*(1.0-r/R);
}

double linear_solution(const Vector &v)
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

double radial_solution(const Vector &v)
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

  const double c1{1.0/std::log(R_HP/R_B) * (
    (T_HP-T_B)*(R_B-R_HP)*k/q0 - std::pow(R_HP,3)/9. + 0.25*R_B*std::pow(R_HP,2) - 5./36.*std::pow(R_B,3))};
  const double c2{T_B*(R_B-R_HP)*k/q0 + 5./36.*std::pow(R_B,3) - c1*std::log(R_B)};

  if (r < R_HP) return T_HP;
  return q0/k / (R_B - R_HP) * (std::pow(r,3)/9. - 0.25*R_B*std::pow(r,2) + c1*std::log(r) + c2);
}

double var_cond_solution(const Vector &v)
{
  const double x{v(0)-0.2};

  constexpr double q0{200.0};
  constexpr double k0{10.0};
  constexpr double kbar{5.100359646753139};
  constexpr double alpha{1e-2};
  constexpr double L{5.0};
  constexpr double T0{600.0};
  constexpr double TL{300.0};

  const double c2{q0*std::pow(2.0*L/M_PI, 2) - 0.5/alpha*(std::pow(k0, 2) -
    std::pow(alpha*T0-k0, 2))};
  const double c1{-1.0/L * (0.5/alpha*(std::pow(k0, 2) - std::pow(alpha*TL-k0, 2)) + c2)};

  constexpr bool constant{false};

  if (x < 0.0) return T0;
  if (constant) return -(q0/kbar) * std::pow(2.0*L/M_PI, 2) * (1.0 - x/L -
      std::cos(0.5*M_PI*x/L)) + (TL-T0)/L*x + T0;
  return (k0 - std::sqrt(std::pow(k0, 2) - 2.0*alpha * (
    q0*std::pow(2.0*L/M_PI, 2)*std::cos(0.5*M_PI*x/L) - c1*x - c2)))/alpha;
}

double var_cond_radial_solution(const Vector &v)
{

  const double r = std::sqrt(v(0)*v(0) + v(1)*v(1));

  const double q0 = 400.0;
  const double k0 = 10.0;
  const double alpha = 1e-2;
  const double R = 5.0;
  const double TR = 400.0;

  const double c2 = 0.5*(k0*k0 - std::pow(alpha*TR-k0,2))/alpha + 5.0/36.0*q0*R*R;

  return (k0 - std::sqrt(k0*k0 - 2.0*alpha*(q0*(r*r*r/(9.0*R) - 0.25*r*r) + c2)))/alpha;

}

void temperature_heat_conduction(const Input &inp,
                                 const Material_library &mat_lib,
                                 QuadratureFunction &power,
                                 ParGridFunction &temperature)
{

  constexpr bool thermal_conductivity_plot = false;

  ParFiniteElementSpace *parfespace = temperature.ParFESpace();
  ParMesh *pmesh = parfespace->GetParMesh();
  QuadratureSpace *qspace = power.GetSpace();
  const int myid = parfespace->GetMyRank();

  conduction_geom_data::set_geometry(parfespace, inp.boundary_conditions, inp.heat_pipe_idx);
  conduction_geom_data::initialize_temperature(inp.temp_boundary, inp.temp_heat_pipe, temperature);

  QuadratureFunction thermal_conductivity_quad{qspace};
  QuadratureFunctionCoefficient thermal_conductivity_quad_func{thermal_conductivity_quad};

  ParBilinearForm a{parfespace};
  a.AddDomainIntegrator(new DiffusionIntegrator{thermal_conductivity_quad_func});
  // For Robin bc, add BoundaryMassIntegrator

  FunctionCoefficient linear_source_coeff{linear_source};
  FunctionCoefficient radial_source_coeff{radial_source};
  FunctionCoefficient var_cond_source_coeff{var_cond_source};
  FunctionCoefficient var_cond_radial_source_coeff{var_cond_radial_source};
  QuadratureFunctionCoefficient power_coeff{power};

  ParLinearForm b{parfespace};
  switch (inp.sol)
  {
    case (Solver_choice::conduction_linear):
        b.AddDomainIntegrator(new DomainLFIntegrator{linear_source_coeff});
        break;
    case (Solver_choice::conduction_radial):
        b.AddDomainIntegrator(new DomainLFIntegrator{radial_source_coeff});
        break;
    case (Solver_choice::conduction_var_cond):
        b.AddDomainIntegrator(new DomainLFIntegrator{var_cond_source_coeff});
        break;
    case (Solver_choice::conduction_var_cond_radial):
        b.AddDomainIntegrator(new DomainLFIntegrator{var_cond_radial_source_coeff});
        break;
    default:
        b.AddDomainIntegrator(new DomainLFIntegrator{power_coeff});
        break;
  }

  // For Robin bc, add BoundaryLFIntegrator
  b.Assemble();

  double temp_conv{inp.tol.temp*1e2+1.0};
  for (int iter{0}; (iter < inp.tol.temp_max_outer) &&
      (temp_conv > inp.tol.temp); ++iter)
  {

    ParGridFunction temperature_old{temperature};

    calc_thermal_conductivity(inp, mat_lib, temperature,
      thermal_conductivity_quad);

    if (thermal_conductivity_plot)
    {
      ParGridFunction k{parfespace};
      k.ProjectCoefficient(thermal_conductivity_quad_func);
      ParaViewDataCollection out{"thermal_conductivity", pmesh};
      out.RegisterField("thermal_conductivity", &k);
      out.SetFormat(DataCollection::PARALLEL_FORMAT);
      out.Save();
    }

    a.Update(); // dont understand but needs to be there
    a.Assemble();
    a.Finalize();

    OperatorHandle A;
    Vector X, B;
    a.FormLinearSystem(conduction_geom_data::ess_tdof_list, temperature, b, A, X, B);

    HypreBoomerAMG amg{*A.As<HypreParMatrix>()};
    amg.SetPrintLevel(0);

    HyprePCG cg{MPI_COMM_WORLD};
    cg.SetPrintLevel(inp.verbose ? 2 : 0);
    cg.SetPreconditioner(amg);
    cg.SetOperator(*A);
    cg.SetTol(inp.tol.temp_inner_rel);
    cg.SetAbsTol(inp.tol.temp_inner_rel);
    cg.SetMaxIter(inp.tol.temp_max_inner);

    cg.Mult(B, X);

    a.RecoverFEMSolution(X, b, temperature);

    temperature_old -= temperature;
    temperature_old.HostRead(); // necessary to sync memory for GPU problems
    const double list_temp_conv{temperature_old.Normlinf()};
    MPI_Allreduce(&list_temp_conv, &temp_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    //if ((inp.verbose) && (myid == 0))
    if (myid == 0)
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
        FunctionCoefficient solution{radial_solution};
        conduction_error_analysis(solution, temperature);
        break;
      }
    case (Solver_choice::conduction_var_cond):
      {
        FunctionCoefficient solution{var_cond_solution};
        conduction_error_analysis(solution, temperature);
        break;
      }
    case (Solver_choice::conduction_var_cond_radial):
      {
        FunctionCoefficient solution{var_cond_radial_solution};
        conduction_error_analysis(solution, temperature);
        std::cout << "Tmax=" << temperature.Normlinf() << std::endl;
        break;
      }
    default:
      break; // do nothing
  }

  return;
}

void conduction_error_analysis(Coefficient &coeff, ParGridFunction &sol)
{
  const double l2error{sol.ComputeL2Error(coeff)};
  const double lmaxerror{sol.ComputeMaxError(coeff)};

  if (sol.ParFESpace()->GetMyRank() == 0)
  {
    std::cout << "===TEMPERATURE_ERROR===" << std::endl;
    std::cout << "l2error=" << l2error << std::endl;
    std::cout << "lmaxerror=" << lmaxerror << std::endl;
  }

  ParGridFunction exact{sol};
  exact.ProjectCoefficient(coeff);
  Vector nval_exact;
  exact.GetNodalValues(nval_exact);

  Vector nval_sol;
  sol.GetNodalValues(nval_sol);

  ParMesh *mesh = sol.ParFESpace()->GetParMesh();
  Vector node;
  mesh->GetVertices(node);

  std::ofstream os{"solution.out"};

  const int dim = mesh->SpaceDimension();

  // i, x, y, z, sol, exact, diff

  const int nv = mesh->GetNV();

  for (int i{0}; i < nv; ++i)
  {
    os << i << ",";
    Vector coord(3);
    coord = 0.0;
    for(int j{0}; j < dim; ++j)
      coord(j) = node(j*nv + i);
    for (int j{0}; j < coord.Size(); ++j)
      os << coord(j) << ",";
    os << nval_sol(i) << ","
       << nval_exact(i) << ","
       << nval_sol(i) - nval_exact(i) << std::endl;
  }


  return;
}

void calc_thermal_conductivity(const Input &inp,
                               const Material_library &mat_lib,
                               const ParGridFunction &temperature,
                               QuadratureFunction &thermal_conductivity_quad)
{

  constexpr double analytic_coeff = 0.1;

  constexpr bool kconst = false;
  constexpr double k0 = 10.0;
  constexpr double alpha = 1e-2;
  constexpr double kbar = 5.100359646753139;

  const QuadratureSpace* const qspace = thermal_conductivity_quad.GetSpace();
  const ParFiniteElementSpace* const fespace = temperature.ParFESpace();
  const ParMesh* const mesh = fespace->GetParMesh();

  switch (inp.sol)
  {
    case (Solver_choice::conduction_linear):
    case (Solver_choice::conduction_radial):
      thermal_conductivity_quad = analytic_coeff;
      return;
    default:
      thermal_conductivity_quad = 0.0;
  }

  int idx = 0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = mesh->GetAttribute(e) - 1;

    std::function<double(double)> k_T;
    if (inp.sol == Solver_choice::conduction_var_cond)
      k_T = [k0, alpha](double T){return (kconst) ? kbar : (k0 - alpha*T);};
    else if (inp.sol == Solver_choice::conduction_var_cond_radial)
      k_T = [k0, alpha](double T){return k0 - alpha*T;};
    else
    {
      const Material_record::Thermal_material kmat = mat_lib.mat[attribute].k_mat;
      k_T = [kmat](double T){ return Material_record::thermal_conductivity(kmat, T);} ;
    }

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
      const IntegrationPoint &ip = ir.IntPoint(i);
      const double T = temperature.GetValue(e, ip);
      const double k = k_T(T);
      thermal_conductivity_quad[idx] = k;
      ++idx;
    }

  }

  return;
}
