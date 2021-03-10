#include "conduction.hpp"

#include "thermal_conductivity_coefficient.hpp"

#include <iostream>

using namespace mfem;

void temperature_heat_conduction(const Input& inp,
                                 const Material_library& mat_lib,
                                 const Array<int>& ess_bdr,
                                 const ParGridFunction& power,
                                 ParGridFunction& temperature)
{

  ParFiniteElementSpace* parfespace{power.ParFESpace()};
  const ParMesh* const pmesh{parfespace->GetParMesh()};

  ThermalConductivityCoefficient thermal_conductivity_coeff{mat_lib, &temperature};
  ParBilinearForm a{parfespace};
  a.AddDomainIntegrator(new DiffusionIntegrator{thermal_conductivity_coeff});
  // For Robin bc, add BoundaryMassIntegrator

  GridFunctionCoefficient power_coeff{&power};

  ParLinearForm b{parfespace};
  b.AddDomainIntegrator(new DomainLFIntegrator{power_coeff});
  // For Robin bc, add BoundaryLFIntegrator
  b.Assemble();

  Array<int> ess_tdof_list;
  parfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.temp_inner_rel);
  cg.SetAbsTol(inp.tol.temp_inner_abs);
  cg.SetMaxIter(inp.tol.temp_max_inner);
  cg.SetPrintLevel(inp.mfem_print_level());
  cg.SetPreconditioner(amg);

  OperatorPtr A;
  Vector B, X;

  ParGridFunction temperature_old{temperature};

  double temp_conv{inp.tol.temp*1e2+1.0};
  for (int iter{0}; (iter < inp.tol.temp_max_outer) && 
      (inp.tol.temp > temp_conv); ++iter)
  {

    a.Assemble();

    a.FormLinearSystem(ess_tdof_list, temperature, b, A, X, B);
    cg.SetOperator(*A);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, temperature);

    temperature_old -= temperature;
    temperature_old.HostRead(); // necessary to sync memory for GPU problems
    const double local_temp_conv{temperature_old.Normlinf()};
    MPI_Allreduce(&local_temp_conv, &temp_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    std::cout << "iter=" << iter << "temp_conv=" << temp_conv << std::endl;

    temperature_old = temperature;
  }

  return;
}
