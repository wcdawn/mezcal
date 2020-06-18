#include "solver.hpp"

#include <limits>

using namespace mfem;
using std::numeric_limits;

using std::cout;
using std::endl;
using std::flush;

Result diffusion_lobpcg(const Input& inp, const XSLibrary& xs,
                        const Array<int>& ess_bdr, const ParMesh* const pmesh,
                        ParFiniteElementSpace* const fespace)
{

  //////////////////////////////////////////////////////////////////////////////
  // assemble A (LHS) matrix
  //////////////////////////////////////////////////////////////////////////////
  auto a{new ParBilinearForm(fespace)};
  if (inp.partial_assembly)
    a->SetAssemblyLevel(AssemblyLevel::PARTIAL);

  // assemble leakage coefficient
  Vector leakage(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * xs.rec[i].sigma_tr[0]);
  PWConstCoefficient leakage_func(leakage);

  // assemble removal coefficient
  Vector removal(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i) removal[i] = xs.rec[i].sigma_r[0];
  PWConstCoefficient removal_func(removal);

  a->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a->AddDomainIntegrator(new MassIntegrator(removal_func));
  if (inp.static_condensation)
    a->EnableStaticCondensation();

  a->Assemble();
  a->EliminateEssentialBCDiag(ess_bdr, 1.0);
  a->Finalize();

  HypreParMatrix* A = a->ParallelAssemble();
  delete a;

  //////////////////////////////////////////////////////////////////////////////
  // assemble M (RHS) matrix
  //////////////////////////////////////////////////////////////////////////////
  auto m{new ParBilinearForm(fespace)};
  if (inp.partial_assembly)
    m->SetAssemblyLevel(AssemblyLevel::PARTIAL);

  // assemble fission coefficient
  Vector fission(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i) fission[i] = xs.rec[i].nusf[0];
  PWConstCoefficient fission_func(fission);

  m->AddDomainIntegrator(new MassIntegrator(fission_func));
  if (inp.static_condensation)
    m->EnableStaticCondensation();

  m->Assemble();
  // shift the eigenvalue corresponding to eliminated dofs to a large value
  m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  m->Finalize();

  HypreParMatrix* M = m->ParallelAssemble();
  delete m;

  // 11. Setup the preconditioner.
  Solver* prec{nullptr};
  auto amg{new HypreBoomerAMG(*A)};
  const int hypre_print_level{inp.verbose ? 1 : 0};
  amg->SetPrintLevel(hypre_print_level);
  prec = amg;

  auto lobpcg{new HypreLOBPCG(MPI_COMM_WORLD)};
  lobpcg->SetNumModes(1); // solve for nev eigenmodes
  // Random seed is set used to initialize the initial eigenvalues for the
  // lobpcg algorithm. This can significantly affect convergence time but
  // convergence time remains bounded. No default provided by MFEM but this is
  // used in one of the tutorials.
  lobpcg->SetRandomSeed(730);
  lobpcg->SetPreconditioner(*prec);
  lobpcg->SetMaxIter(inp.tol.max_outer);
  lobpcg->SetTol(inp.tol.k * inp.tol.phi);
  lobpcg->SetPrecondUsageMode(1);
  lobpcg->SetPrintLevel(hypre_print_level);
  lobpcg->SetMassMatrix(*M);
  lobpcg->SetOperator(*A);

  // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
  //    parallel grid function to represent each of the eigenmodes returned by
  //    the solver.
  lobpcg->Solve();

  delete A;
  delete M;

  Result res(fespace);

  // store solution
  Array<double> eigenvalues;
  lobpcg->GetEigenvalues(eigenvalues);
  res.keff = 1.0 / eigenvalues[0];
  res.phi =
    lobpcg->GetEigenvector(0); // this is the local solution on this process

  // free memory from solver
  delete lobpcg;
  delete prec; // don't also delete amg or that would be double delete

  return res;
}

// this will not work AT ALL for parallel problems
Result diffusion_power_iteration(const Input& inp, const XSLibrary& xs,
                                 const Array<int>& ess_bdr,
                                 const ParMesh* const pmesh,
                                 ParFiniteElementSpace* const fespace)
{
  const int hypre_print_level{inp.verbose ? 1 : 0};
  const int mfem_print_level{inp.verbose ? 1 : 0};

  auto a{new ParBilinearForm(fespace)};
  if (inp.partial_assembly)
    a->SetAssemblyLevel(AssemblyLevel::PARTIAL);

  Vector leakage(pmesh->attributes.Max());
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * xs.rec[i].sigma_tr[0]);
  PWConstCoefficient leakage_func{leakage};

  Vector removal(pmesh->attributes.Max());
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = xs.rec[i].sigma_r[0];
  PWConstCoefficient removal_func{removal};

  a->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a->AddDomainIntegrator(new MassIntegrator(removal_func));

  if (inp.static_condensation)
    a->EnableStaticCondensation();

  a->Assemble();

  Solver* prec{nullptr};
  auto amg{new HypreBoomerAMG};
  amg->SetPrintLevel(hypre_print_level);
  prec = amg;

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(mfem_print_level);
  if (prec)
    cg.SetPreconditioner(*prec);

  OperatorPtr A;
  Vector B;
  Vector X;

  Vector fission(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < fission.Size(); ++i) fission[i] = xs.rec[i].nusf[0];
  PWConstCoefficient fiss_func(fission);

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  Result res(fespace);

  ParGridFunction phi_old(fespace);
  phi_old = 1.0;

  GridFunctionCoefficient phi_func(&phi_old);
  ProductCoefficient fiss_term(fiss_func, phi_func);

  auto b{new ParLinearForm(fespace)};
  b->AddDomainIntegrator(new DomainLFIntegrator(fiss_term));
    
  auto fiss{new ParLinearForm(fespace)};
  fiss->AddDomainIntegrator(new DomainLFIntegrator(fiss_term));

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    
  double keff_old{res.keff};
  double fiss_sum{1.0};
  double local_fiss_sum{1.0};
  double fiss_sum_old{1.0};

  double k_conv{inp.tol.k * 1e1 + 1e-5};
  double phi_conv{inp.tol.phi * 1e1 + 1e-5};
  double local_phi_conv{phi_conv};

  int iter{0};
  for (iter = 0; (iter < inp.tol.max_outer) && ((k_conv > inp.tol.k)
                 || (phi_conv > inp.tol.phi)); ++iter)
  {
    MPI_Barrier(MPI_COMM_WORLD);

    // construct source
    b->Assemble();
    MPI_Barrier(MPI_COMM_WORLD);

    *b /= res.keff;
    MPI_Barrier(MPI_COMM_WORLD);

    // Solve this fixed source problem.
    // Note that FormLinearSystem is designed to be called multiple times.
    a->FormLinearSystem(ess_tdof_list, res.phi, *b, A, X, B);
    MPI_Barrier(MPI_COMM_WORLD);

    cg.SetOperator(*A);
    MPI_Barrier(MPI_COMM_WORLD);

    cg.Mult(B, X); // solve
    MPI_Barrier(MPI_COMM_WORLD);

    a->RecoverFEMSolution(X, *b, res.phi);
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate fission sum.
    if (iter == 0)
    {
      // Don't update keff.
      // This is important for multi-state problems but also avoids extreme
      // values of keff.
      fiss_sum = fiss_sum_old;
      // Use res.phi for all subsequent fission integrations.
      phi_func.SetGridFunction(&res.phi);
    }
    else
    {
      fiss->Assemble();
      MPI_Barrier(MPI_COMM_WORLD);

      local_fiss_sum = fiss->Sum();
      MPI_Barrier(MPI_COMM_WORLD);

      MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // update keff
    res.keff = res.keff * (fiss_sum / fiss_sum_old);
    MPI_Barrier(MPI_COMM_WORLD);

    // update convergence criteria
    k_conv = std::abs(res.keff - keff_old);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&res.keff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&k_conv, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    phi_old -= res.phi;
    MPI_Barrier(MPI_COMM_WORLD);

    local_phi_conv = phi_old.Normlinf();
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    phi_old = res.phi;
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&phi_conv, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // copy old values
    keff_old = res.keff;
    MPI_Barrier(MPI_COMM_WORLD);

    fiss_sum_old = fiss_sum;
    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0)
    {
      cout << "iter=" << iter << " keff=" << res.keff 
           << " dk=" << k_conv << " dphi=" << phi_conv << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  delete fiss;
  //delete b;
  delete prec;
  delete a;

  if ((iter == inp.tol.max_outer) && (myid == 0))
  {
    cout << "Failed to converge in maximum allowed power iterations.\n"
         << "max_outer = " << inp.tol.max_outer << "\n"
         << "(k_tol, k_conv) = "
         << "(" << inp.tol.k << ", " << k_conv << ")\n"
         << "(phi_tol, phi_conv) = "
         << "(" << inp.tol.phi << ", " << phi_conv << ")\n";
  }

  return res;
}

Result diffusion_fixed_src(const Input& inp, const XSLibrary& xs,
                           const Array<int>& ess_bdr,
                           const ParMesh* const pmesh,
                           ParFiniteElementSpace* const fespace)
{

  const int mfem_print_level{inp.verbose ? 1 : 0};
  const int hypre_print_level{inp.verbose ? 1 : 0};

  auto a{new ParBilinearForm(fespace)};
  if (inp.partial_assembly)
    a->SetAssemblyLevel(AssemblyLevel::PARTIAL);

  Vector leakage(pmesh->attributes.Max());
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * xs.rec[i].sigma_tr[0]);
  PWConstCoefficient leakage_func(leakage);

  Vector removal(pmesh->attributes.Max());
  for (int i{0}; i < removal.Size(); ++i) removal[i] = xs.rec[i].sigma_r[0];
  PWConstCoefficient removal_func(removal);

  a->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a->AddDomainIntegrator(new MassIntegrator(removal_func));

  if (inp.static_condensation)
    a->EnableStaticCondensation();

  a->Assemble();

  auto b{new ParLinearForm(fespace)};

  Vector source(pmesh->attributes.Max());
  for (int i{0}; i < source.Size(); ++i) source[i] = xs.rec[i].source[0];
  PWConstCoefficient source_func{source};

  b->AddDomainIntegrator(new DomainLFIntegrator(source_func));

  b->Assemble();

  OperatorPtr A;
  Vector B;
  Vector X;

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  Result res{fespace};

  a->FormLinearSystem(ess_tdof_list, res.phi, *b, A, X, B);

  Solver* prec{nullptr};
  auto amg{new HypreBoomerAMG};
  amg->SetPrintLevel(hypre_print_level);
  prec = amg;

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(mfem_print_level);
  if (prec)
    cg.SetPreconditioner(*prec);
  cg.SetOperator(*A);
  cg.Mult(B, X);
  delete prec;

  a->RecoverFEMSolution(X, *b, res.phi);

  delete b;
  delete a;

  return res;
}
