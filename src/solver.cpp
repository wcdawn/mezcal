#include "solver.hpp"

#include "scattering_coefficient.hpp"
#include "fission_coefficient.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace mfem;
using std::numeric_limits;

using std::vector;
using std::min;

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
  amg->SetPrintLevel(inp.hypre_print_level());
  prec = amg;

  auto lobpcg{new HypreLOBPCG(MPI_COMM_WORLD)};
  lobpcg->SetNumModes(1); // solve for nev eigenmodes
  // Random seed is set used to initialize the initial eigenvalues for the
  // lobpcg algorithm. This can significantly affect convergence time but
  // convergence time remains bounded.
  lobpcg->SetRandomSeed(730);
  lobpcg->SetPreconditioner(*prec);
  lobpcg->SetMaxIter(inp.tol.max_outer);
  lobpcg->SetTol(min(inp.tol.phi * inp.tol.phi, inp.tol.k * inp.tol.k));
  lobpcg->SetPrecondUsageMode(1);
  lobpcg->SetPrintLevel(inp.hypre_print_level());
  lobpcg->SetMassMatrix(*M);
  lobpcg->SetOperator(*A);

  // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
  //    parallel grid function to represent each of the eigenmodes returned by
  //    the solver.
  lobpcg->Solve();

  delete A;
  delete M;

  Result res{fespace, xs.ngroup()};

  constexpr int g{0};

  // store solution
  Array<double> eigenvalues;
  lobpcg->GetEigenvalues(eigenvalues);
  res.keff = 1.0 / eigenvalues[0];
  res.phi[g] = lobpcg->GetEigenvector(0);

  // free memory from solver
  delete lobpcg;
  delete prec; // don't also delete amg or that would be double delete

  return res;
}

Result diffusion_power_iteration(const Input& inp, const XSLibrary& xs,
                                 const Array<int>& ess_bdr,
                                 const ParMesh* const pmesh,
                                 ParFiniteElementSpace* const fespace)
{

  // this is necessary to specify bcs for the inner solver
  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // build streaming and removal operator (LHS) for each energy group
  vector<ParBilinearForm*> a(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
  {
    a[g] = new ParBilinearForm(fespace);
    if (inp.partial_assembly) a[g]->SetAssemblyLevel(AssemblyLevel::PARTIAL);

    Vector leakage(pmesh->attributes.Max());
    for (int i{0}; i < leakage.Size(); ++i)
      leakage[i] = 1.0 / (3.0 * xs.rec[i].sigma_tr[g]);
    PWConstCoefficient leakage_func{leakage};

    Vector removal(pmesh->attributes.Max());
    for (int i{0}; i < removal.Size(); ++i)
      removal[i] = xs.rec[i].sigma_r[g];
    PWConstCoefficient removal_func{removal};

    a[g]->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
    a[g]->AddDomainIntegrator(new MassIntegrator(removal_func));

    if (inp.static_condensation) a[g]->EnableStaticCondensation();

    a[g]->Assemble();
  }

  // create a coefficientfor the nu*fission cross section in each energy group
  vector<PWConstCoefficient> nusf_coeff(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
  {
    Vector fission(pmesh->attributes.Max()); // length is number of materials
    for (int i{0}; i < fission.Size(); ++i)
      fission[i] = xs.rec[i].nusf[g];
    nusf_coeff[g].UpdateConstants(fission);
  }

  // set initial flux profile to be used for fission source
  // currently we use a uniform "flat" flux profile
  ParGridFunction phi_old(fespace);
  phi_old = 1.0;

  // prepare solution that will be returned
  Result res{fespace, xs.ngroup()};
  for (auto& flux : res.phi)
    flux = 1.0;

  // set the grid function coefficient for each energy group flux
  vector<GridFunctionCoefficient> phi_coeff(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
    phi_coeff[g].SetGridFunction(&res.phi[g]);

  // note that this one does not have a default constructor so we have to do
  // this a bit differently
  vector<ProductCoefficient> fiss_product_coeff;
  fiss_product_coeff.reserve(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
    fiss_product_coeff.push_back(
      ProductCoefficient{nusf_coeff[g], phi_coeff[g]});

  // build "outline" of fission source integrator for each energy group
  vector<ParLinearForm*> fiss(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); g++)
  {
    fiss[g] = new ParLinearForm(fespace);
    fiss[g]->AddDomainIntegrator(new DomainLFIntegrator(fiss_product_coeff[g]));
  }

  // setup preconditioner
  Solver* prec{nullptr};
  auto amg{new HypreBoomerAMG};
  amg->SetPrintLevel(inp.hypre_print_level());
  prec = amg;

  // setup solver
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(inp.mfem_print_level());
  if (prec) cg.SetPreconditioner(*prec);

  // these are necessary for the solver
  OperatorPtr A;
  Vector B;
  Vector X;

  // get the process id so that only the root process writes to terminal
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // initialize "old" data
  double keff_old{res.keff};
  double fiss_sum{1.0};
  double fiss_sum_old{1.0};

  // initialize convergence critera (make sure we enter the loop at least once)
  double k_conv{inp.tol.k * 1e1 + 1e-5};
  double phi_conv{inp.tol.phi * 1e1 + 1e-5};

  vector<FissionCoefficient> fiss_term;
  fiss_term.reserve(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
    fiss_term.push_back(FissionCoefficient{g, res.keff, xs, res.phi});

  vector<ScatteringCoefficient> scat_term;
  scat_term.reserve(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
    scat_term.push_back(ScatteringCoefficient{g, xs, res.phi});

  vector<ParLinearForm*> b(xs.ngroup());
  for (int g{0}; g < xs.ngroup(); ++g)
  {
    b[g] = new ParLinearForm(fespace);
    b[g]->AddDomainIntegrator(new DomainLFIntegrator(fiss_term[g]));
    b[g]->AddDomainIntegrator(new DomainLFIntegrator(scat_term[g]));
  }

  int iter{0};
  for (iter = 0; (iter < inp.tol.max_outer) && ((k_conv > inp.tol.k)
                 || (phi_conv > inp.tol.phi)); ++iter)
  {
    // Solve this fixed source problem for each energy group.
    // Note that FormLinearSystem is designed to be called multiple times.
    double local_phi_conv{0.0};
    for (int g{0}; g < xs.ngroup(); ++g)
    {
      // assemble source for this group
      b[g]->Assemble();

      // necessary to enforce zero-flux boundary conditions
      if (iter == 0) res.phi[g] = 0.0;

      // store the old flux value for this group
      phi_old = res.phi[g];

      // form, solve, and recover the linear system
      a[g]->FormLinearSystem(ess_tdof_list, res.phi[g], *b[g], A, X, B);
      cg.SetOperator(*A);
      cg.Mult(B, X); // solve
      a[g]->RecoverFEMSolution(X, *b[g], res.phi[g]);
      
      // determine the convergence of this energy group and store the maximum
      // value over all energy groups
      phi_old -= res.phi[g];
      local_phi_conv = std::max(local_phi_conv, phi_old.Normlinf());
    }

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Calculate fission sum.
    if (iter == 0)
    {
      // Don't update keff.
      // This is important for multi-state problems but also avoids extreme
      // values of keff.
      fiss_sum = fiss_sum_old;
    }
    else
    {
      double local_fiss_sum{0.0};
      for (int g{0}; g < xs.ngroup(); ++g)
      {
        fiss[g]->Assemble();
        local_fiss_sum += fiss[g]->Sum();
      }
      MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    
    // update keff
    res.keff = res.keff * (fiss_sum / fiss_sum_old);

    // update convergence criteria
    k_conv = std::abs(res.keff - keff_old);

    // copy old values
    keff_old = res.keff;
    fiss_sum_old = fiss_sum;

    if (myid == 0)
    {
      cout << "iter=" << iter << " keff=" << res.keff 
           << " dk=" << k_conv << " dphi=" << phi_conv << endl;
    }
  }

  delete prec;

  // TODO consider creating a special class to adhere better to RAII
  for (int g{0}; g < xs.ngroup(); ++g)
  {
    delete a[g];
    delete b[g];
    delete fiss[g];
  }

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

  Result res{fespace, xs.ngroup()};

  // TODO we must require that the fixed source problem has a single energy group
  a->FormLinearSystem(ess_tdof_list, res.phi[0], *b, A, X, B);

  Solver* prec{nullptr};
  auto amg{new HypreBoomerAMG};
  amg->SetPrintLevel(inp.hypre_print_level());
  prec = amg;

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(inp.mfem_print_level());
  if (prec)
    cg.SetPreconditioner(*prec);
  cg.SetOperator(*A);
  cg.Mult(B, X);
  delete prec;

  a->RecoverFEMSolution(X, *b, res.phi[0]);

  delete b;
  delete a;

  return res;
}
