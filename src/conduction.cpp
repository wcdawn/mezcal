#include "conduction.hpp"

#include "thermal_conductivity_coefficient.hpp"

#include <iostream>
#include <unordered_set>

using namespace mfem;

void temperature_heat_conduction(const Input& inp,
                                 const Material_library& mat_lib,
                                 const ParGridFunction& power,
                                 ParGridFunction& temperature)
{

  ParFiniteElementSpace* const parfespace{power.ParFESpace()};
  const ParMesh* const pmesh{parfespace->GetParMesh()};

  Array<int> ess_bdr, ess_tdof_list;
  ess_bdr.SetSize(pmesh->bdr_attributes.Max());
  ess_bdr = 1; // all boundaries are essential (all external boundaries for now)
  parfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  constexpr int heat_pipe_id{5}; // TODO need some other method for this...

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
  temperature.ProjectCoefficient(heat_pipe_coeff);
  temperature.ProjectBdrCoefficient(temp_boundary_coeff, ess_bdr);

  ThermalConductivityCoefficient thermal_conductivity_coeff{mat_lib, &temperature};
  ParBilinearForm a{parfespace};
  a.AddDomainIntegrator(new DiffusionIntegrator{thermal_conductivity_coeff});
  // For Robin bc, add BoundaryMassIntegrator

  GridFunctionCoefficient power_coeff{&power};

  ParLinearForm b{parfespace};
  b.AddDomainIntegrator(new DomainLFIntegrator{power_coeff});
  // For Robin bc, add BoundaryLFIntegrator
  b.Assemble();

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
      (temp_conv > inp.tol.temp); ++iter)
  {

    a.Update();
    a.Assemble();
    a.Finalize();
    
    b.Assemble(); // necessary for bcs

    a.FormLinearSystem(ess_tdof_list, temperature, b, A, X, B);
    cg.SetOperator(*A);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, temperature);

    temperature_old -= temperature;
    temperature_old.HostRead(); // necessary to sync memory for GPU problems
    const double local_temp_conv{temperature_old.Normlinf()};
    MPI_Allreduce(&local_temp_conv, &temp_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    //if ((inp.verbose) && (parfespace->GetMyRank() == 0))
    if (parfespace->GetMyRank() == 0)
      std::cout << "temperature_iter=" << iter << " temp_conv=" << temp_conv << std::endl;

    temperature_old = temperature;
  }

  return;
}
