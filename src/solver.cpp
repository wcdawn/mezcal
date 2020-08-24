#include "solver.hpp"

#include "fission_coefficient.hpp"
#include "scattering_coefficient.hpp"
#include "make_unique.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>


using namespace mfem;
using std::numeric_limits;

using std::max;
using std::string;
using std::vector;

using std::cout;
using std::flush;

using std::unique_ptr;
using std::make_unique;

#if __cplusplus >= 201703L
#include <filesystem>
using std::filesystem::exists; // requires C++17
#else
#include <fstream>
bool exists(const std::string& fname)
{
  std::ifstream ifs{fname};
  return ifs.good();
}
#endif

PetscParMatrix* bilinear_to_petscparmatrix(ParBilinearForm& pbf);

Result diffusion_hypre_lobpcg(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const Array<int>& ess_bdr,
                              const ParMesh* const pmesh,
                              ParFiniteElementSpace* const fespace)
{

  time.start(time.matrix_assembly);

  //////////////////////////////////////////////////////////////////////////////
  // assemble A (LHS) matrix
  //////////////////////////////////////////////////////////////////////////////
  ParBilinearForm a{fespace};

  // assemble leakage coefficient
  Vector leakage(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[0]);
  PWConstCoefficient leakage_func(leakage);

  // assemble removal coefficient
  Vector removal(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = mat_lib.mat[i].rec.sigma_r[0];
  PWConstCoefficient removal_func(removal);

  a.AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a.AddDomainIntegrator(new MassIntegrator(removal_func));
  if (inp.static_condensation) a.EnableStaticCondensation();

  a.Assemble();
  a.EliminateEssentialBCDiag(ess_bdr, 1.0);
  a.Finalize();

  const unique_ptr<HypreParMatrix> A{a.ParallelAssemble()};

  //////////////////////////////////////////////////////////////////////////////
  // assemble M (RHS) matrix
  //////////////////////////////////////////////////////////////////////////////
  ParBilinearForm m{fespace};

  // assemble fission coefficient
  Vector fission(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i)
    fission[i] = mat_lib.mat[i].rec.nusf[0];
  PWConstCoefficient fission_func(fission);

  m.AddDomainIntegrator(new MassIntegrator(fission_func));
  if (inp.static_condensation) m.EnableStaticCondensation();

  m.Assemble();
  // shift the eigenvalue corresponding to eliminated dofs to a large value
  m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  m.Finalize();

  const unique_ptr<HypreParMatrix> M{m.ParallelAssemble()};

  time.stop();

  // 11. Setup the preconditioner.
  HypreBoomerAMG amg(*A);
  amg.SetPrintLevel(inp.hypre_print_level());

  HypreLOBPCG lobpcg{MPI_COMM_WORLD};
  lobpcg.SetNumModes(1); // solve for nev eigenmodes
  // Random seed is set used to initialize the initial eigenvalues for the
  // lobpcg algorithm. This can significantly affect convergence time but
  // convergence time remains bounded.
  lobpcg.SetRandomSeed(730);
  lobpcg.SetPreconditioner(amg);
  lobpcg.SetMaxIter(inp.tol.max_outer);
  lobpcg.SetTol(inp.tol.phi * inp.tol.k);
  lobpcg.SetPrecondUsageMode(1);
  lobpcg.SetPrintLevel(inp.hypre_print_level());
  lobpcg.SetMassMatrix(*M);
  lobpcg.SetOperator(*A);

  time.start(time.solver);
  lobpcg.Solve();
  time.stop();

  Result res{fespace, mat_lib.ngroup()};

  // store solution
  Array<double> eigenvalues;
  lobpcg.GetEigenvalues(eigenvalues);
  res.keff   = 1.0 / eigenvalues[0];
  res.phi[0] = lobpcg.GetEigenvector(0);

  return res;
}

Result diffusion_slepc(Timer& time, const Input& inp,
                       const Material_library& mat_lib,
                       const Array<int>& ess_bdr, const ParMesh* const pmesh,
                       ParFiniteElementSpace* const fespace)
{
  constexpr int g{0};

  const string slepcrc_fname{inp.concatenate_path("slepcrc")};
  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(NULL, NULL, slepcrc_fname.c_str(), NULL);
  else
    MFEMInitializeSlepc(NULL, NULL, NULL, NULL);

  time.start(time.matrix_assembly);

  Vector leakage(pmesh->attributes.Max());
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[g]);
  PWConstCoefficient leakage_func{leakage};

  Vector removal(pmesh->attributes.Max());
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = mat_lib.mat[i].rec.sigma_r[g];
  PWConstCoefficient removal_func{removal};

  ParBilinearForm a{fespace};
  a.AddDomainIntegrator(new DiffusionIntegrator{leakage_func});
  a.AddDomainIntegrator(new MassIntegrator{removal_func});
  a.Assemble();
  a.EliminateEssentialBCDiag(ess_bdr, 1.0);
  a.Finalize();

  Vector fission(pmesh->attributes.Max());
  for (int i{0}; i < fission.Size(); ++i)
    fission[i] = mat_lib.mat[i].rec.nusf[g];
  PWConstCoefficient fission_func{fission};

  ParBilinearForm m{fespace};
  m.AddDomainIntegrator(new MassIntegrator{fission_func});
  m.Assemble();
  m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  m.Finalize();

  time.stop();

  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(inp.tol.phi * inp.tol.k);
  slepc.SetMaxIter(inp.tol.max_outer);

  // spectral transformation will not be useful for Generalized Davidson solver
  // as it uses a preconditioner instead of a spectral transformation
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);

  // set PETSc operators appropriately
  // not sure of a better way to do this
  PetscParMatrix* A{nullptr};
  PetscParMatrix* M{nullptr};

  {
    OperatorHandle Ah{Operator::PETSC_MATAIJ};
    OperatorHandle Mh{Operator::PETSC_MATAIJ};
    a.ParallelAssemble(Ah);
    Ah.Get(A);
    Ah.SetOperatorOwner(false);
    m.ParallelAssemble(Mh);
    Mh.Get(M);
    Mh.SetOperatorOwner(false);
  }

  slepc.SetOperators(*A, *M);

  time.start(time.solver);
  slepc.Solve();
  time.stop();

  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  Result res{fespace, mat_lib.ngroup()};

  slepc.GetEigenvalue(0, res.keff);
  res.keff = 1.0 / res.keff;
  Vector tmp(fespace->GetTrueVSize());
  slepc.GetEigenvector(0, tmp);
  res.phi[g].Distribute(tmp);

  return res;
}

Result diffusion_power_iteration(Timer& time, const Input& inp,
                                 const Material_library& mat_lib,
                                 const Array<int>& ess_bdr,
                                 const ParMesh* const pmesh,
                                 ParFiniteElementSpace* const fespace)
{

  // this is necessary to specify bcs for the inner solver
  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // NOTE: rebuilding the RHS matrix is not included in matrix assembly.
  time.start(time.matrix_assembly);

  // build streaming and removal operator (LHS) for each energy group
  // note ParBilinearForm has neither copy nor move constructor

  vector<unique_ptr<ParBilinearForm>> a(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    a[g] = make_unique<ParBilinearForm>(fespace);

    Vector leakage(pmesh->attributes.Max());
    for (int i{0}; i < leakage.Size(); ++i)
      leakage[i] = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[g]);
    PWConstCoefficient leakage_func{leakage};

    Vector removal(pmesh->attributes.Max());
    for (int i{0}; i < removal.Size(); ++i)
      removal[i] = mat_lib.mat[i].rec.sigma_r[g];
    PWConstCoefficient removal_func{removal};

    a[g]->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
    a[g]->AddDomainIntegrator(new MassIntegrator(removal_func));

    if (inp.static_condensation) a[g]->EnableStaticCondensation();

    a[g]->Assemble();
  }

  time.stop();

  // create a coefficientfor the nu*fission cross section in each energy group
  // this will be used for the flux integral at the end of each loop
  vector<PWConstCoefficient> nusf_coeff;
  nusf_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector fission(pmesh->attributes.Max()); // length is number of materials
    for (int i{0}; i < fission.Size(); ++i)
      fission[i] = mat_lib.mat[i].rec.nusf[g];
    nusf_coeff.emplace_back(fission);
  }

  // set initial flux profile to be used for fission source
  // currently we use a uniform "flat" flux profile
  ParGridFunction phi_old(fespace);
  phi_old = 1.0;

  // prepare solution that will be returned
  Result res{fespace, mat_lib.ngroup()};
  for (auto& flux : res.phi) flux = 1.0;

  // set the grid function coefficient for each energy group flux
  vector<GridFunctionCoefficient> phi_coeff;
  phi_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g) phi_coeff.emplace_back(&res.phi[g]);

  // note that this one does not have a default constructor so we have to do
  // this a bit differently
  vector<ProductCoefficient> fiss_product_coeff;
  fiss_product_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    fiss_product_coeff.emplace_back(nusf_coeff[g], phi_coeff[g]);

  // build "outline" of fission source integrator for each energy group
  vector<ParLinearForm> fiss(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); g++)
  {
    fiss[g].Update(fespace);
    fiss[g].AddDomainIntegrator(new DomainLFIntegrator(fiss_product_coeff[g]));
  }

  // setup preconditioner
  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  // setup solver
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(inp.mfem_print_level());
  cg.SetPreconditioner(amg);

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
  fiss_term.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    fiss_term.emplace_back(g, res.keff, mat_lib, res.phi, inp.flux_free_chi);

  vector<ScatteringCoefficient> scat_term;
  scat_term.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scat_term.emplace_back(g, mat_lib, res.phi);

  vector<ParLinearForm> b(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    b[g].Update(fespace);
    b[g].AddDomainIntegrator(new DomainLFIntegrator(fiss_term[g]));
    b[g].AddDomainIntegrator(new DomainLFIntegrator(scat_term[g]));
  }

  time.start(time.solver);
  int iter{0};
  for (iter = 0; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi));
       ++iter)
  {
    // Solve this fixed source problem for each energy group.
    // Note that FormLinearSystem is designed to be called multiple times.
    double local_phi_conv{0.0};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      // assemble source for this group
      b[g].Assemble();

      // necessary to enforce zero-flux boundary conditions
      if (iter == 0) res.phi[g] = 0.0;

      // store the old flux value for this group
      phi_old = res.phi[g];

      // form, solve, and recover the linear system
      a[g]->FormLinearSystem(ess_tdof_list, res.phi[g], b[g], A, X, B);
      cg.SetOperator(*A);
      cg.Mult(B, X); // solve
      a[g]->RecoverFEMSolution(X, b[g], res.phi[g]);

      // determine the convergence of this energy group and store the maximum
      // value over all energy groups
      phi_old -= res.phi[g];

      // need to sync memory for gpu problems
      phi_old.GetMemory().Read(MemoryClass::HOST, phi_old.Size());

      local_phi_conv =
      max(local_phi_conv, phi_old.Normlinf() / res.phi[g].Max());
    }

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

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
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        fiss[g].Assemble();
        local_fiss_sum += fiss[g].Sum();
      }
      MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    }

    // update keff
    res.keff = res.keff * (fiss_sum / fiss_sum_old);

    // update convergence criteria
    k_conv = std::abs(res.keff - keff_old);

    // copy old values
    keff_old     = res.keff;
    fiss_sum_old = fiss_sum;

    if (myid == 0)
    {
      cout << "iter=" << iter << " keff=" << res.keff << " dk=" << k_conv
           << " dphi=" << phi_conv << '\n' << flush;
    }
  }
  time.stop();

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

Result diffusion_fixed_src(Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const Array<int>& ess_bdr,
                           const ParMesh* const pmesh,
                           ParFiniteElementSpace* const fespace)
{
  time.start(time.matrix_assembly);
  ParBilinearForm a{fespace};

  Vector leakage(pmesh->attributes.Max());
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[0]);
  PWConstCoefficient leakage_func(leakage);

  Vector removal(pmesh->attributes.Max());
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = mat_lib.mat[i].rec.sigma_r[0];
  PWConstCoefficient removal_func(removal);

  a.AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a.AddDomainIntegrator(new MassIntegrator(removal_func));

  if (inp.static_condensation) a.EnableStaticCondensation();

  a.Assemble();

  ParLinearForm b{fespace};

  Vector source(pmesh->attributes.Max());
  for (int i{0}; i < source.Size(); ++i)
    source[i] = mat_lib.mat[i].rec.source[0];
  PWConstCoefficient source_func{source};

  b.AddDomainIntegrator(new DomainLFIntegrator(source_func));

  b.Assemble();

  time.stop();

  OperatorPtr A;
  Vector B;
  Vector X;

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  Result res{fespace, mat_lib.ngroup()};

  // ASSUMPTION we require that the fixed source problem has a single energy
  // group
  a.FormLinearSystem(ess_tdof_list, res.phi[0], b, A, X, B);

  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(inp.mfem_print_level());
  cg.SetPreconditioner(amg);
  cg.SetOperator(*A);
  time.start(time.solver);
  cg.Mult(B, X);
  time.stop();

  a.RecoverFEMSolution(X, b, res.phi[0]);

  return res;
}

Result diffusion_slepc_vector(Timer& time, const Input& inp,
                       const Material_library& mat_lib,
                       const Array<int>& ess_bdr, const ParMesh* const pmesh,
                       ParFiniteElementSpace* const fespace)
{

  const string slepcrc_fname{inp.concatenate_path("slepcrc")};
  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(NULL, NULL, slepcrc_fname.c_str(), NULL);
  else
    MFEMInitializeSlepc(NULL, NULL, NULL, NULL);

  time.start(time.matrix_assembly);

  ParBilinearForm diffusion_bilinear{fespace};
  ParBilinearForm fission_bilinear{fespace};

  DenseMatrix leakage(mat_lib.ngroup()); // zero initialized
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    leakage(g,g) = 1.0 / (3.0 * mat_lib.mat[0].rec.sigma_tr[g]);
  MatrixConstantCoefficient leakage_func{leakage};

  DenseMatrix removal(mat_lib.ngroup());
  mat_lib.mat[0].rec.sigma_s.ToDenseMatrix(removal);
  removal.Neg(); // negate (for subtraction)
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    removal(g,g) += mat_lib.mat[0].rec.sigma_t[g];
  MatrixConstantCoefficient removal_func{removal};

  // TODO i need to write a custom coefficient for a MatrixPWConstCoefficient

  DenseMatrix fission(mat_lib.ngroup()); // zero initialized
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    {
      // fission(gprime,g) = mat_lib.mat[0].rec.chi[gprime] * mat_lib.mat[0].rec.nusf[g];
      const auto& mat{mat_lib.mat[0]};
      for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
      {
        const auto& isotope{mat.iso[iso_idx]};
        const auto& record{isotope.rec};
        const double nden{isotope.nden};
        fission(gprime,g) = record.chi[g] * nden * record.nusf[gprime];
      }
    }
  }
  MatrixConstantCoefficient fission_func{fission};

  // TODO figure out BCs

  diffusion_bilinear.AddDomainIntegrator(new DiffusionIntegrator{leakage_func});
  diffusion_bilinear.AddDomainIntegrator(new VectorMassIntegrator{removal_func});
  diffusion_bilinear.Assemble();
  diffusion_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
  diffusion_bilinear.Finalize();

  fission_bilinear.AddDomainIntegrator(new VectorMassIntegrator{fission_func});
  fission_bilinear.Assemble();
  fission_bilinear.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  fission_bilinear.Finalize();

  time.stop();

  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(inp.tol.phi * inp.tol.k);
  slepc.SetMaxIter(inp.tol.max_outer);

  // spectral transformation will not be useful for Generalized Davidson solver
  // as it uses a preconditioner instead of a spectral transformation
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);

  // set PETSc operators appropriately
  PetscParMatrix* A{nullptr};
  PetscParMatrix* M{nullptr};

  {

    OperatorHandle Ah{Operator::PETSC_MATAIJ};
    diffusion_bilinear.ParallelAssemble(Ah);
    Ah.Get(A);
    Ah.SetOperatorOwner(false);

    OperatorHandle Mh{Operator::PETSC_MATAIJ};
    fission_bilinear.ParallelAssemble(Mh);
    Mh.Get(M);
    Mh.SetOperatorOwner(false);

  }

  slepc.SetOperators(*A, *M);

  A->Print("petsca");
  M->Print("petscm");

  time.start(time.solver);
  slepc.Solve();
  time.stop();

  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  Result res{fespace, mat_lib.ngroup()};

  slepc.GetEigenvalue(0, res.keff);
  res.keff = 1.0 / res.keff;
  Vector tmp(fespace->GetTrueVSize());
  slepc.GetEigenvector(0, tmp);
  // TODO figure this out...

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int idx{g}; idx < tmp.Size(); idx += mat_lib.ngroup())
      std::cout << tmp[idx] << std::endl;
    std::cout << std::endl;
  }

  res.phi[0].Distribute(tmp);

  return res;
}

Result diffusion_slepc_manual(Timer& time, const Input& inp,
                       const Material_library& mat_lib,
                       const Array<int>& ess_bdr, const ParMesh* const pmesh,
                       ParFiniteElementSpace* const fespace)
{

  // NOTE: this call to MFEMInitializeSlepc() leaks.
  // It leaks 112 bytes in 2 blocks. (definitely lost)
  // Small, but really annoying.
  // It seems to probably be a PETSc issue.
  const string slepcrc_fname{inp.concatenate_path("slepcrc")};
  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(NULL, NULL, slepcrc_fname.c_str(), NULL);
  else
    MFEMInitializeSlepc();

  time.start(time.matrix_assembly);

  const int offset{fespace->TrueVSize()}; // needed for parallel
  Array<int> offset_list(mat_lib.ngroup() + 1);
  offset_list[0] = 0;
  for (int g{1}; g < mat_lib.ngroup() + 1; ++g)
    offset_list[g] = offset_list[g-1] + offset;

  BlockOperator diffusion_block{offset_list};
  BlockOperator fission_block{offset_list};

  vector<unique_ptr<PetscParMatrix>> matrix_list;
  matrix_list.reserve(mat_lib.ngroup() * mat_lib.ngroup());

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Vector leakage(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      leakage[i] = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[g]);
    PWConstCoefficient leakage_func{leakage};

    Vector removal(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      removal[i] = mat_lib.mat[i].rec.sigma_r[g];
    PWConstCoefficient removal_func{removal};

    ParBilinearForm diffusion_bilinear{fespace};
    diffusion_bilinear.AddDomainIntegrator(new DiffusionIntegrator{leakage_func});
    diffusion_bilinear.AddDomainIntegrator(new MassIntegrator{removal_func});
    diffusion_bilinear.Assemble();
    diffusion_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
    diffusion_bilinear.Finalize();

    matrix_list.emplace_back(bilinear_to_petscparmatrix(diffusion_bilinear));
    diffusion_block.SetDiagonalBlock(g, matrix_list.back().get());

    for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    {

      Vector fission(pmesh->attributes.Max());
      fission = 0.0;
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        const auto& mat{mat_lib.mat[i]};
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const auto& isotope{mat.iso[iso_idx]};
          const auto& record{isotope.rec};
          const double nden{isotope.nden};
          fission[i] += record.chi[g] * nden * record.nusf[gprime];
        }
      }
      PWConstCoefficient fission_func{fission};

      ParBilinearForm fission_bilinear{fespace};
      fission_bilinear.AddDomainIntegrator(new MassIntegrator{fission_func});
      fission_bilinear.Assemble();
      fission_bilinear.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
      fission_bilinear.Finalize();

      matrix_list.emplace_back(bilinear_to_petscparmatrix(fission_bilinear));
      fission_block.SetBlock(g, gprime, matrix_list.back().get());

      if (gprime != g)
      {
        Vector scatter(pmesh->attributes.Max());
        scatter = 0.0; // not zero initialized by default
        for (int i{0}; i < pmesh->attributes.Max(); ++i)
        {
          Array<int> col_idx;
          Vector data;
          mat_lib.mat[i].rec.sigma_s.GetRow(g, col_idx, data);
          const auto gprime_idx{col_idx.FindSorted(gprime)};
          if (gprime_idx >= 0)
            scatter[i] -= data[gprime_idx];
        }
        PWConstCoefficient scatter_func{scatter};

        ParBilinearForm scatter_bilinear{fespace};
        scatter_bilinear.AddDomainIntegrator(new MassIntegrator{scatter_func});
        scatter_bilinear.Assemble();
        scatter_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
        scatter_bilinear.Finalize();

        matrix_list.emplace_back(bilinear_to_petscparmatrix(scatter_bilinear));
        diffusion_block.SetBlock(g, gprime, matrix_list.back().get());

      }

    }

  }

  time.stop();

  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(inp.tol.phi * inp.tol.k);
  slepc.SetMaxIter(inp.tol.max_outer);

  // spectral transformation will not be useful for Generalized Davidson solver
  // as it uses a preconditioner instead of a spectral transformation
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);

  // set PETSc operators appropriately
  PetscParMatrix Atmp{MPI_COMM_WORLD, &diffusion_block, Operator::PETSC_MATAIJ};
  PetscParMatrix A{&Atmp, Operator::PETSC_MATAIJ};

  PetscParMatrix Mtmp{MPI_COMM_WORLD, &fission_block, Operator::PETSC_MATAIJ};
  PetscParMatrix M{&Mtmp, Operator::PETSC_MATAIJ};

  slepc.SetOperators(A, M);

  // NOTE: this call to SlepcEigenSolver::Solve() leaks.
  // It leaks 124 bytes in 6 blocks. (definitely lost)
  // It leaks 1,408 bytes in 4 blocks. (indirecly lost)
  // It seems to probably be a SLEPc issue.
  time.start(time.solver);
  slepc.Solve();
  time.stop();

  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  Result res{fespace, mat_lib.ngroup()};

  slepc.GetEigenvalue(0, res.keff);
  res.keff = 1.0 / res.keff;
  Vector tmp(fespace->GetTrueVSize()*mat_lib.ngroup());
  slepc.GetEigenvector(0, tmp);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector group(fespace->GetTrueVSize());
    for (int idx{0}; idx < fespace->GetTrueVSize(); ++idx)
      group[idx] = tmp[idx + g * fespace->GetTrueVSize()];
    res.phi[g].Distribute(group);
  }

  time.start("return");
  return res;
}

PetscParMatrix* bilinear_to_petscparmatrix(ParBilinearForm& pbf)
{
  OperatorHandle oh{Operator::PETSC_MATAIJ};
  PetscParMatrix* A{nullptr};
  pbf.ParallelAssemble(oh);
  oh.Get(A);
  oh.SetOperatorOwner(false);
  return A;
}
