#include "solver.hpp"

#include "fission_coefficient.hpp"
#include "scattering_coefficient.hpp"
#include "spn_source_coefficient.hpp"
#include "make_unique.hpp"
#include "vector_pwconst_coefficient.hpp"
#include "matrix_pwconst_coefficient.hpp"
#include "incoming_flux_coefficient.hpp"
#include "stability_tau_coefficient.hpp"

#include "vector_integrator_wrapper.hpp"
#include "../addon/tensor.hpp"

#include "memory.hpp"

#include "slepceps.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "../mfem/linalg/petscinternals.hpp" // PCHKERRQ

using namespace mfem;

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

double par_norml2(const Vector &x);

Memory_stats mem(Memory_stats::megabyte);

double omega_x;
double omega_y;
double omega_z;

double psi_analytic_source(const mfem::Vector& v)
{
  const double& x{v(0)};
  const double& y{v(1)};
  //const double& z{v(2)};

  constexpr double L{5.0};
  constexpr double N{static_cast<double>(4)}; // for level_symmetric order2
  constexpr double A{1.0}; // normalization constant
  constexpr double pil{M_PI/L};
  constexpr double sigma_t{0.1};

  return A * (N+1.0)*0.5 * (
    pil * omega_x * std::cos(pil*x) * std::sin(pil*y) + 
    pil * omega_y * std::sin(pil*x) * std::cos(pil*y) +
    sigma_t * std::sin(pil*x) * std::sin(pil*y));
}

double point_source(const mfem::Vector& v)
{
  const double& x{v(0)};
  const double& y{v(1)};
  //const double& z{v(2)};

  constexpr double point_x{2.5};
  constexpr double point_y{2.5};
  constexpr double source{0.01};

  constexpr double tol{0.25};

  double dist{std::sqrt(std::pow(point_x-x,2) + std::pow(point_y-y,2))};

  if (dist < tol)
    return source;

  return 0.0;

}

void diffusion_hypre_lobpcg(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const Array<int>& ess_bdr,
                              const ParMesh* const pmesh,
                              ParFiniteElementSpace* const fespace, Result& res)
{

  static constexpr int ell{0}; // for P0 xs

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
    removal[i] = mat_lib.mat[i].rec.sigma_r(0,ell);
  PWConstCoefficient removal_func(removal);

  a.AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a.AddDomainIntegrator(new MassIntegrator(removal_func));
  if (inp.static_condensation) a.EnableStaticCondensation();

  a.Assemble();
  a.EliminateEssentialBCDiag(ess_bdr, 1.0);
  a.Finalize();

  const std::unique_ptr<HypreParMatrix> A{a.ParallelAssemble()};

  //////////////////////////////////////////////////////////////////////////////
  // assemble M (RHS) matrix
  //////////////////////////////////////////////////////////////////////////////
  ParBilinearForm m{fespace};

  // assemble fission coefficient
  Vector fission(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < fission.Size(); ++i)
    fission[i] = mat_lib.mat[i].rec.nusf[0];
  PWConstCoefficient fission_func(fission);

  m.AddDomainIntegrator(new MassIntegrator(fission_func));
  if (inp.static_condensation) m.EnableStaticCondensation();

  m.Assemble();
  // shift the eigenvalue corresponding to eliminated dofs to a large value
  m.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
  m.Finalize();

  const std::unique_ptr<HypreParMatrix> M{m.ParallelAssemble()};

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

  A->Print("hypre_lobpcg_a");
  M->Print("hypre_lobpcg_m");

  time.start(time.solver);
  lobpcg.Solve();
  time.stop();

  // store solution
  Array<double> eigenvalues;
  lobpcg.GetEigenvalues(eigenvalues);
  res.keff = 1.0 / eigenvalues[0];
  res(0)   = lobpcg.GetEigenvector(0);

}

void spn_power_iteration(Timer& time, const Input& inp,
                         Material_library& mat_lib,
                         Array<int>& ess_bdr,
                         const ParMesh* const pmesh,
                         ParFiniteElementSpace* const fespace, Result& res)
{

  constexpr bool isWielandt{false};
  constexpr double delta_shift{0.04};
  constexpr int free_iterations{3};
  double shift{1e2};

  // need to be able to update on-the-fly so must use PWConstCoefficient
  PWConstCoefficient shift_coefficient(pmesh->attributes.Max());
  shift_coefficient=-1.0/shift;

  mat_lib.set_nmoment(inp.spn_order()+1);

  // this is necessary to specify bcs for the inner solver
  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // NOTE: rebuilding the RHS matrix is not included in matrix assembly.
  time.start(time.matrix_assembly);

  SPNWrapper<PWConstCoefficient> leakage_func(mat_lib.ngroup(), inp.spn_eq());
  SPNWrapper<PWConstCoefficient> removal_func(mat_lib.ngroup(), inp.spn_eq());
  leakage_func.reserve();
  removal_func.reserve();

  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    const double dble_n{2.0*n};
    const double xmult_next{(dble_n+1.0)*(dble_n+1.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};
    const double xmult_prev{dble_n*dble_n / (4.0*dble_n*dble_n - 1.0)};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      Vector leakage(pmesh->attributes.Max());
      for (int i{0}; i < leakage.Size(); ++i)
      {
        leakage(i) = xmult_next / mat_lib.mat[i].rec.sigma_r(g,2*n+1);
        if (n > 0)
          leakage(i) += xmult_prev / mat_lib.mat[i].rec.sigma_r(g,2*n-1);
      }
      leakage_func.emplace_back(leakage);

      Vector removal(pmesh->attributes.Max());
      for (int i{0}; i < removal.Size(); ++i)
        removal(i) = mat_lib.mat[i].rec.sigma_r(g,2*n);
      removal_func.emplace_back(removal);

    }
  }

  // build streaming and removal operator (LHS) for each energy group
  // note ParBilinearForm has neither copy nor move constructor
  SPNWrapper<std::unique_ptr<ParBilinearForm>> a(mat_lib.ngroup(), inp.spn_eq());
  a.resize();
  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      a(g,n) = std::make_unique<ParBilinearForm>(fespace);
      a(g,n)->AddDomainIntegrator(new DiffusionIntegrator{leakage_func(g,n)});
      a(g,n)->AddDomainIntegrator(new MassIntegrator{removal_func(g,n)});
    }
  }

  time.stop();

  // set initial flux profile to be used for fission source
  // currently we use a uniform "flat" flux profile
  ParGridFunction phi_old{fespace};
  phi_old = 1.0;

  // prepare solution that will be returned for first iteration
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      if (res(g,n).Normlinf() < std::numeric_limits<double>::min())
      {
        res(g,n) = 1.0;
      }
      ConstantCoefficient zero{0.0};
      res(g,n).ProjectBdrCoefficient(zero, ess_bdr);
    }
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

  // get the process id so that only the root process writes to terminal
  const int myid{fespace->GetMyRank()};

  // initialize "old" data
  double keff_old{res.keff};
  double fiss_sum{1.0};
  double fiss_sum_old{1.0};

  // initialize convergence criteria (make sure we enter the loop at least once)
  double k_conv{inp.tol.k * 1e1 + 1e-5};
  double phi_conv{inp.tol.phi * 1e1 + 1e-5};

  std::vector<QuadratureFunction> fiss_quad_func;
  std::vector<QuadratureFunction> upscatter_quad_func, dnscatter_quad_func;
  fiss_quad_func.reserve(mat_lib.ngroup());
  upscatter_quad_func.reserve(mat_lib.ngroup());
  dnscatter_quad_func.reserve(mat_lib.ngroup());

  std::vector<QuadratureFunctionCoefficient> fiss_quad_coeff;
  std::vector<QuadratureFunctionCoefficient> upscatter_quad_coeff, dnscatter_quad_coeff;
  fiss_quad_coeff.reserve(mat_lib.ngroup());
  upscatter_quad_coeff.reserve(mat_lib.ngroup());
  dnscatter_quad_coeff.reserve(mat_lib.ngroup());

  const int integration_order{fespace->GetOrder(0)+1};
  QuadratureSpace qspace{const_cast<ParMesh*>(pmesh), integration_order};

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_quad_func.emplace_back(&qspace);
    upscatter_quad_func.emplace_back(&qspace);
    dnscatter_quad_func.emplace_back(&qspace);
    fiss_quad_coeff.emplace_back(fiss_quad_func.back());
    upscatter_quad_coeff.emplace_back(upscatter_quad_func.back());
    dnscatter_quad_coeff.emplace_back(dnscatter_quad_func.back());
  }

  SPNWrapper<QuadratureFunction> spn_next_quad_func(mat_lib.ngroup(), inp.spn_eq());
  SPNWrapper<QuadratureFunction> spn_prev_quad_func(mat_lib.ngroup(), inp.spn_eq());
  spn_next_quad_func.reserve();
  spn_prev_quad_func.reserve();

  SPNWrapper<VectorQuadratureFunctionCoefficient> spn_next_quad_coeff(mat_lib.ngroup(), inp.spn_eq());
  SPNWrapper<VectorQuadratureFunctionCoefficient> spn_prev_quad_coeff(mat_lib.ngroup(), inp.spn_eq());
  spn_next_quad_coeff.reserve();
  spn_prev_quad_coeff.reserve();

  const int dim{pmesh->Dimension()};

  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      spn_next_quad_func.emplace_back(&qspace, dim);
      spn_prev_quad_func.emplace_back(&qspace, dim);
      spn_next_quad_coeff.emplace_back(spn_next_quad_func(g,n));
      spn_prev_quad_coeff.emplace_back(spn_prev_quad_func(g,n));
    }
  }

  SPNWrapper<std::unique_ptr<ParLinearForm>> b(mat_lib.ngroup(), inp.spn_eq());
  b.resize();
  for (int n{0}; n < inp.spn_eq(); ++ n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      b(g,n) = std::make_unique<ParLinearForm>(fespace);

      if ((n+1) < inp.spn_eq())
        b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{spn_next_quad_coeff(g,n)});

      if (n > 0)
        b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{spn_prev_quad_coeff(g,n)});

      if (n == 0)
      {
        b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{fiss_quad_coeff[g], nullptr});
        b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{upscatter_quad_coeff[g], nullptr});
        b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{dnscatter_quad_coeff[g], nullptr});
      }
    }
  }

  std::vector<GridFunctionCoefficient> scalar_flux;
  std::vector<PowerCoefficient> inv_scalar_flux;
  scalar_flux.reserve(mat_lib.ngroup());
  inv_scalar_flux.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    scalar_flux.emplace_back(&res(g));
    inv_scalar_flux.emplace_back(scalar_flux.back(), -1.0);
  }

  std::vector<ProductCoefficient> intermediate_wielandt_sutton_coeff, wielandt_sutton_coeff;
  intermediate_wielandt_sutton_coeff.reserve(mat_lib.ngroup());
  wielandt_sutton_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    intermediate_wielandt_sutton_coeff.emplace_back(inv_scalar_flux[g], fiss_quad_coeff[g]);
    wielandt_sutton_coeff.emplace_back(shift_coefficient, intermediate_wielandt_sutton_coeff.back());
  }

  if (isWielandt)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      a(g,0)->AddDomainIntegrator(new MassIntegrator{wielandt_sutton_coeff[g]});
    }
  }

////////////////////////////////////////////////////////////////////////////////
  std::vector<std::unique_ptr<ParLinearForm>> fiss(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss[g] = std::make_unique<ParLinearForm>(fespace);
    
    Vector nusf_vec{pmesh->attributes.Max()};
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      nusf_vec(i) = mat_lib.mat[i].rec.nusf[g];
    PWConstCoefficient nusf_operator{nusf_vec};

    fiss[g]->AddDomainIntegrator(new DomainLFIntegrator{nusf_operator});
    fiss[g]->Assemble();
  }
////////////////////////////////////////////////////////////////////////////////

  // assemble all higher-order moments as these are not shifted
  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      a(g,n)->Assemble();
      a(g,n)->Finalize();
    }
  }

  double beta{1.0};
  time.start(time.solver);
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {
    // Solve this fixed source problem for each energy group and each moment.
    // Note that FormLinearSystem is designed to be called multiple times.

    double local_phi_conv{0.0};

    if ((isWielandt) && (iter >= free_iterations)) // dynamic shift
    {
      shift = res.keff + delta_shift;
      shift_coefficient = -1.0/shift;
    }

    if (isWielandt)
      beta = shift * res.keff / (shift - res.keff);
    else
      beta = res.keff;

    time.start("fiss&up");
    calc_fiss_upscatter(res, mat_lib, fiss_quad_func, upscatter_quad_func, inp.flux_free_chi);

    if (isWielandt)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        time.start("AAssemble");
        a(g,0)->Update();
        a(g,0)->Assemble();
        a(g,0)->Finalize();
        time.stop("AAssemble");
      }
    }

    for (QuadratureFunction& qf : fiss_quad_func)
      qf /= beta;

    time.stop();
    if (inp.spn_eq() > 1)
    {
      time.start("spn_next");
      calc_spn_next(res, mat_lib, spn_next_quad_func);
      time.stop();
    }

    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {

        if (n == 0)
        {
          time.start("down");
          calc_downscatter(g, res, mat_lib, dnscatter_quad_func[g]);
          time.stop();
        }
        else
        {
          time.start("spn_prev");
          calc_spn_prev(g, n, res, mat_lib, spn_prev_quad_func(g,n));
          time.stop();
        }

        time.start("BAssemble");
        // assemble source for this group
        b(g,n)->Assemble();
        time.stop("BAssemble");

        // store the old flux value for this group
        phi_old = res(g,n);

        // these are necessary for the solver
        OperatorPtr A;
        Vector B, X;

        // form, solve, and recover the linear system
        a(g,n)->FormLinearSystem(ess_tdof_list, res(g,n), *b(g,n), A, X, B);
        cg.SetOperator(*A);
        time.start("Mult");
        cg.Mult(B, X); // solve
        time.stop("Mult");
        a(g,n)->RecoverFEMSolution(X, *b(g,n), res(g,n));

        // flux convergence critria on all moments
        // determine the convergence of this energy group and store the maximum
        // value over all energy groups
        phi_old -= res(g,n);

        // need to sync memory for gpu problems
        phi_old.GetMemory().Read(MemoryClass::HOST, phi_old.Size());

        local_phi_conv =
          std::max(local_phi_conv, phi_old.Normlinf() / res(g,n).Normlinf());

      }
    }

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    time.start("FissSum");
    // Calculate fission sum.
    double local_fiss_sum{0.0};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      local_fiss_sum += (*fiss[g])(res(g));
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    time.stop("FissSum");

    // update keff
    if (iter > 0) beta *= fiss_sum / fiss_sum_old;

    if (isWielandt)
      res.keff = beta * shift / (beta + shift);
    else
      res.keff = beta;

    // update convergence criteria
    k_conv = std::abs(res.keff - keff_old);

    // copy old values
    keff_old     = res.keff;
    fiss_sum_old = fiss_sum;

    if (myid == 0)
    {
      std::cout << "iter=" << iter << " keff=" << res.keff << " dk=" << k_conv
                << " dphi=" << phi_conv << '\n' << std::flush;
    }

    if ((iter == inp.tol.max_outer) && (myid == 0))
    {
      std::cout << "Failed to converge in maximum allowed power iterations.\n"
                << "max_outer = " << inp.tol.max_outer << "\n"
                << "(k_tol, k_conv) = "
                << "(" << inp.tol.k << ", " << k_conv << ")\n"
                << "(phi_tol, phi_conv) = "
                << "(" << inp.tol.phi << ", " << phi_conv << ")\n";
    }

  }
  time.stop();
  return;

}

void spn_amgx_power_iteration(Timer& time, const Input& inp,
                              Material_library& mat_lib,
                              Array<int>& ess_bdr,
                              const ParMesh* const pmesh,
                              ParFiniteElementSpace* const fespace, Result& res)
{

  /*
   * 1) Efficiency of bassemble. Begin with quadrature functions.
   * 2) Boundary conditions.
   * 3) bassemble on GPU.
   */

  constexpr int skip_zeros{2};
  const int myid{fespace->GetMyRank()};
  if (myid == 0) std::cout << "__spn_order__=" << inp.spn_order() << std::endl;

  mat_lib.set_nmoment(inp.spn_order()+1);

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  time.start(time.matrix_assembly);

  SPNWrapper<AmgXSolver> sol(mat_lib.ngroup(), inp.spn_eq());
  SPNWrapper<std::unique_ptr<HypreParMatrix>> amat(mat_lib.ngroup(), inp.spn_eq());
  sol.resize();
  amat.resize();

  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    const double dble_n{2.0*n};
    const double xmult_next{(dble_n+1.0)*(dble_n+1.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};
    const double xmult_prev{dble_n*dble_n / (4.0*dble_n*dble_n - 1.0)};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      ParBilinearForm a{fespace};

      Vector leakage(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        leakage(i) = xmult_next / mat_lib.mat[i].rec.sigma_r(g,2*n+1);
        if (n > 0) leakage(i) += xmult_prev / mat_lib.mat[i].rec.sigma_r(g,2*n-1);
      }
      PWConstCoefficient leakage_func{leakage};
      a.AddDomainIntegrator(new DiffusionIntegrator{leakage_func});

      Vector removal(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
        removal(i) = mat_lib.mat[i].rec.sigma_r(g,2*n);
      PWConstCoefficient removal_func{removal};
      a.AddDomainIntegrator(new MassIntegrator{removal_func});

      a.Assemble(skip_zeros);
      a.Finalize(skip_zeros);

      // TODO figure out how to impose boundary conditions.
      // FormSystemMatrix() looks promising.
      amat(g,n).reset(a.ParallelAssemble());

      sol(g,n).DefaultParameters(AmgXSolver::SOLVER, inp.verbose); // setup solver
      sol(g,n).ReadParameters("amgxrc", AmgXSolver::EXTERNAL); // read solver properties from file
      sol(g,n).InitExclusiveGPU(fespace->GetComm()); // mandate one MPI per gpu
      sol(g,n).SetOperator(*amat(g,n));
    }
  }

  time.stop(time.matrix_assembly);

  std::vector<ScatteringCoefficient> scat_coeff;
  std::vector<FissionCoefficient> fiss_coeff;
  std::vector<SPNSourceCoefficient> spn_coeff;
  scat_coeff.reserve(mat_lib.ngroup());
  fiss_coeff.reserve(mat_lib.ngroup());
  spn_coeff.reserve(mat_lib.ngroup()*inp.spn_eq());

  SPNWrapper<std::unique_ptr<ParLinearForm>> b(mat_lib.ngroup(), inp.spn_eq());
  b.resize();

  // TODO massive improvement needed. Needs to use quadrature function, etc.
  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      b(g,n) = std::make_unique<ParLinearForm>(fespace);
      b(g,n)->UseDevice(true);

      if (n == 0)
      {
        scat_coeff.emplace_back(g, mat_lib, res);
        fiss_coeff.emplace_back(g, res.keff, mat_lib, res);
        b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{scat_coeff.back()});
        b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{fiss_coeff.back()});
      }

      spn_coeff.emplace_back(g, n, inp.spn_eq(), mat_lib, res);
      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{spn_coeff.back()});

    }
  }

////////////////////////////////////////////////////////////////////////////////
  std::vector<std::unique_ptr<ParLinearForm>> fiss(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss[g] = std::make_unique<ParLinearForm>(fespace);
    Vector nusf(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      nusf(i) = mat_lib.mat[i].rec.nusf[g];
    PWConstCoefficient nusf_func{nusf};
    fiss[g]->AddDomainIntegrator(new DomainLFIntegrator{nusf_func});
    fiss[g]->Assemble();
  }
////////////////////////////////////////////////////////////////////////////////

  // initialize "old" data
  double keff_old{res.keff};
  double fiss_sum{1.0};
  double fiss_sum_old{1.0};

  // initialize convergence criteria (make sure we enter the loop at least once)
  double k_conv{inp.tol.k * 1e1 + 1e-5};
  double phi_conv{inp.tol.phi * 1e1 + 1e-5};

  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      if (n == 0) res(g) = 1.0;
      ConstantCoefficient zero{0.0};
      res(g,n).ProjectBdrCoefficient(zero, ess_bdr);
    }
  }

  for (int n{0}; n < inp.spn_eq(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      res(g,n).UseDevice(true);

  for (int iter{0}; (iter < inp.tol.max_outer) && 
      ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    double local_phi_conv{0.0};

    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {

        time.start("bassemble");
        b(g,n)->Assemble(); // All of the time is here...
        time.stop("bassemble");
        time.start("bassemble_hypre");
        std::unique_ptr<HypreParVector> bvec{b(g,n)->ParallelAssemble()};
        time.stop("bassemble_hypre");

        time.start("phi_copy");
        ParGridFunction phi_old{res(g,n)};
        phi_old.UseDevice(true);
        time.stop("phi_copy");

        time.start("mult");
        sol(g,n).Mult(*bvec, res(g,n));
        time.stop("mult");

        time.start("local_phi_conv");
        phi_old -= res(g,n);
        phi_old.HostRead(); // TODO needs a lot of work (maybe?)
        local_phi_conv = std::max(local_phi_conv, phi_old.Normlinf() / res(g,n).Normlinf());
        time.stop("local_phi_conv");

      }
    }

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    time.start("fiss_sum");
    double local_fiss_sum{0.0};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      local_fiss_sum += (*fiss[g])(res(g));
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time.stop("fiss_sum");

    if (iter > 0) res.keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(res.keff - keff_old); // = std::abs(1.0 - fiss_sum/fiss_sum_old);

    // copy old data
    keff_old = res.keff;
    fiss_sum_old = fiss_sum;

    if (myid == 0)
    {
      std::cout << "iter=" << iter << " keff=" << res.keff << " dk=" << k_conv
                << " dphi=" << phi_conv << '\n' << std::flush;
    }

  }

  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      sol(g,n).Finalize();
    }
  }

  return;
}

void diffusion_fixed_src(Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const Array<int>& ess_bdr,
                           const ParMesh* const pmesh,
                           ParFiniteElementSpace* const fespace, Result& res)
{
  static constexpr int ell{0}; // for P0 xs

  time.start(time.matrix_assembly);
  ParBilinearForm a{fespace};

  Vector leakage(pmesh->attributes.Max());
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[0]);
  PWConstCoefficient leakage_func(leakage);

  Vector removal(pmesh->attributes.Max());
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = mat_lib.mat[i].rec.sigma_r(0,ell);
  PWConstCoefficient removal_func(removal);

  a.AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a.AddDomainIntegrator(new MassIntegrator(removal_func));

  if (inp.static_condensation) a.EnableStaticCondensation();

  a.Assemble();
  a.Finalize();

  ParLinearForm b{fespace};

  Vector source(pmesh->attributes.Max());
  for (int i{0}; i < source.Size(); ++i)
    source[i] = mat_lib.mat[i].rec.source[0];
  PWConstCoefficient source_func{source};

  b.AddDomainIntegrator(new DomainLFIntegrator(source_func));

  b.Assemble();

  time.stop();

  OperatorPtr A;
  Vector B, X;

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // ASSUMPTION we require that the fixed source problem has a single energy
  // group
  a.FormLinearSystem(ess_tdof_list, res(0), b, A, X, B);

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

  a.RecoverFEMSolution(X, b, res(0));

}

void spn_slepc_vector(Timer& time, const Input& inp,
                        Material_library& mat_lib,
                        const Array<int>& ess_bdr,
                        const ParMesh* const pmesh,
                        ParFiniteElementSpace* const fespace,
                        Result& res,
                        Result& scalar_flux)
{

  constexpr int skip_zeros{2}; // skip all zeros regardless of symmetry

  // get the process id so that only the root process writes to terminal
  const int myid{fespace->GetMyRank()};

  if (myid == 0) std::cout << "__spn_order__ = " << inp.spn_order() << std::endl;

  mat_lib.set_nmoment(inp.spn_order()+1);

  slepc_davidson_hypre_boomeramg_initialize(inp, mat_lib.ngroup()*inp.spn_eq());

  HypreParMatrix *Ahypre{nullptr}, *Mhypre{nullptr};

  time.start(time.matrix_assembly);

  { // sink construction

  ParBilinearForm diffusion_bilinear{fespace};

  std::vector<DenseMatrix> leakage(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    leakage[i].SetSize(mat_lib.ngroup()*inp.spn_eq());
    const XSRecord& record{mat_lib.mat[i].rec};
    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const int idx_dia{g + mat_lib.ngroup() * n};
        const double dble_n{2.0*n};
        const double xmult_next{(dble_n+1.0)*(dble_n+1.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};

        leakage[i](idx_dia,idx_dia) = xmult_next / record.sigma_r(g,2*n+1);

        if (n > 0)
        {
          const int idx_sub{idx_dia - mat_lib.ngroup()};
          const double xmult_prev{dble_n*dble_n / (4.0*dble_n*dble_n - 1.0)};
          const double ymult{(dble_n*dble_n - dble_n) / (4.0*dble_n*dble_n - 1.0)};
          leakage[i](idx_dia,idx_dia) += xmult_prev / record.sigma_r(g,2*n-1);
          leakage[i](idx_dia,idx_sub) = ymult / record.sigma_r(g,2*n-1);
        }

        if ((n+1) < inp.spn_eq())
        {
          const int idx_sup{idx_dia + mat_lib.ngroup()};
          const double zmult{(dble_n+1.0)*(dble_n+2.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};
          leakage[i](idx_dia,idx_sup) = zmult / record.sigma_r(g,2*n+1);
        }

      }
    }
  }
  MatrixPWConstCoefficient leakage_func{leakage};
  diffusion_bilinear.AddDomainIntegrator(new VectorDiffusionIntegrator{leakage_func});
  if (myid == 0) std::cout << "MATRIXPWCONSTCOEFFICIENT::leakage_func" << std::endl;

  std::vector<DenseMatrix> removal(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    removal[i].SetSize(mat_lib.ngroup()*inp.spn_eq());
    const XSRecord& record{mat_lib.mat[i].rec};

    DenseMatrix scat_mat;
    record.sigma_s.ToDenseMatrix(scat_mat);
    scat_mat.Neg();
    removal[i].AddMatrix(scat_mat, 0, 0);

    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const int idx_dia{g + n*mat_lib.ngroup()};
        removal[i](idx_dia,idx_dia) = record.sigma_r(g,n);
      }
    }
  }
  MatrixPWConstCoefficient removal_func{removal};
  diffusion_bilinear.AddDomainIntegrator(new VectorMassIntegrator{removal_func});
  if (myid == 0) std::cout << "MATRIXPWCONSTCOEFFICIENT::removal_func" << std::endl;

  if (myid == 0) std::cout << "DIFFUSION_ASSEMBLE_BEGIN" << std::endl;
  time.start("diffusion_assemble");
  diffusion_bilinear.Assemble(skip_zeros);
  diffusion_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
  diffusion_bilinear.Finalize(skip_zeros);
  time.stop("diffusion_assemble");
  if (myid == 0) std::cout << "DIFFUSION_ASSEMBLE_END" << std::endl;

  time.start("diffusion_par_assemble");
  Ahypre = diffusion_bilinear.ParallelAssemble();
  time.stop("diffusion_par_assemble");

  } // sink construction

  { // fiss construction

  std::vector<DenseMatrix> fission(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    fission[i].SetSize(mat_lib.ngroup()*inp.spn_eq()); // zero initialized
    const Material_record& mat{mat_lib.mat[i]};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const Material_isotope& isotope{mat.iso[iso_idx]};
          const XSRecord& record{isotope.rec};
          const double nden{isotope.nden};
          fission[i](g,gprime) += record.chi[g] * nden * record.nusf[gprime];
        }
      }
    }
  }

  ParBilinearForm fission_bilinear{fespace};
  MatrixPWConstCoefficient fission_func{fission};
  fission_bilinear.AddDomainIntegrator(new VectorMassIntegrator{fission_func});
  if (myid == 0) std::cout << "MATRIXPWCONSTCOEFFICIENT::fission_func" << std::endl;

  if (myid == 0) std::cout << "FISSION_ASSEMBLE_BEGIN" << std::endl;
  time.start("fission_assemble");
  fission_bilinear.Assemble(skip_zeros);
  fission_bilinear.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
  fission_bilinear.Finalize(skip_zeros);
  time.start("fission_assemble");
  if (myid == 0) std::cout << "FISSION_ASSEMBLE_END" << std::endl;

  time.start("fission_par_assemble");
  Mhypre = fission_bilinear.ParallelAssemble();
  time.stop("fission_par_assemble");

  } // fiss construction

  time.stop(time.matrix_assembly);

  constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};

  if (myid == 0) std::cout << "PETSC_ASSEMBLE_BEGIN" << std::endl;
  time.start("petsc_assemble");
  PetscParMatrix A{Ahypre, petsc_format};
  PetscParMatrix M{Mhypre, petsc_format};
  time.stop("petsc_assemble");
  if (myid == 0) std::cout << "PETSC_ASSEMBLE_END" << std::endl;

  A.Print("petsca");
  M.Print("petscm");

  const int vdim{mat_lib.ngroup()*inp.spn_eq()};
  PetscErrorCode ierr;
  ierr = MatSetBlockSize(A, vdim); PCHKERRQ(A, ierr);
  ierr = MatSetBlockSize(M, vdim); PCHKERRQ(M, ierr);

  if ((petsc_format != Operator::PETSC_MATHYPRE) && 
      (petsc_format != Operator::PETSC_MATSHELL))
  {
    delete Ahypre;
    delete Mhypre;
  }

  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(inp.tol.phi * inp.tol.k);
  slepc.SetMaxIter(inp.tol.max_outer);
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
  slepc.SetOperators(A, M);

  /* RESTART
  // set based on the previous state
  Vector true_dof;
  res(0).GetTrueDofs(true_dof);
  const PetscParVector tv{MPI_COMM_WORLD, true_dof, true};
  const std::unique_ptr<Vec[]> array{std::make_unique<Vec[]>(1)};
  array[0] = tv;
  EPSSetInitialSpace(slepc, 1, array.get());
  */

  if (myid == 0) std::cout << "SOLVE_BEGIN" << std::endl;
  time.start(time.solver);
  slepc.Solve();
  time.stop();
  if (myid == 0) std::cout << "SOLVE_END" << std::endl;

  if ((petsc_format == Operator::PETSC_MATHYPRE) ||
      (petsc_format == Operator::PETSC_MATSHELL))
  {
    delete Ahypre;
    delete Mhypre;
  }

  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  slepc.GetEigenvalue(0, res.keff);
  res.keff = 1.0 / res.keff;
  Vector tmp(fespace->GetTrueVSize());
  slepc.GetEigenvector(0, tmp);

  std::ofstream ofs{"sol"};
  tmp.Print(ofs, 1);

  // each group/moment is currently stored as a vector quantity
  // parsing this is then handled by ParaView
  // for dimensions <= 3, the vector is listed {x,y,z}
  // for higher dimensions, the vector is listed {0,1,2,3,4,...}
  res(0).Distribute(tmp);

  // flip the sign of the solution such that the scalar flux in group=0 is positive
  tmp.Destroy();
  res(0).GetNodalValues(tmp);
  if (std::abs(tmp.Max()) < std::abs(tmp.Min()))
    res(0).Neg();

  scalar_flux.keff = res.keff;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    GridFunctionCoefficient gfc{&res(0), g+1};
    scalar_flux(g).ProjectCoefficient(gfc);
  }

  return;
}

void spn_amgx_vector(Timer& time, const Input& inp,
                     Material_library& mat_lib,
                     const Array<int>& ess_bdr,
                     const ParMesh* const pmesh,
                     ParFiniteElementSpace* const fespace,
                     Result& res)
{

  constexpr int skip_zeros{2}; // skip all zeros regardless of symmetry

  // get the process id so that only the root process writes to terminal
  const int myid{fespace->GetMyRank()};

  if (myid == 0) std::cout << "__spn_order__ = " << inp.spn_order() << std::endl;

  mat_lib.set_nmoment(inp.spn_order()+1);

  HypreParMatrix *Ahypre{nullptr}, *Mhypre{nullptr};

  time.start(time.matrix_assembly);

  { // sink construction

  ParBilinearForm diffusion_bilinear{fespace};

  std::vector<DenseMatrix> leakage(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    leakage[i].SetSize(mat_lib.ngroup()*inp.spn_eq());
    const XSRecord& record{mat_lib.mat[i].rec};
    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const int idx_dia{g + mat_lib.ngroup() * n};
        const double dble_n{2.0*n};
        const double xmult_next{(dble_n+1.0)*(dble_n+1.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};

        leakage[i](idx_dia,idx_dia) = xmult_next / record.sigma_r(g,2*n+1);

        if (n > 0)
        {
          const int idx_sub{idx_dia - mat_lib.ngroup()};
          const double xmult_prev{dble_n*dble_n / (4.0*dble_n*dble_n - 1.0)};
          const double ymult{(dble_n*dble_n - dble_n) / (4.0*dble_n*dble_n - 1.0)};
          leakage[i](idx_dia,idx_dia) += xmult_prev / record.sigma_r(g,2*n-1);
          leakage[i](idx_dia,idx_sub) = ymult / record.sigma_r(g,2*n-1);
        }

        if ((n+1) < inp.spn_eq())
        {
          const int idx_sup{idx_dia + mat_lib.ngroup()};
          const double zmult{(dble_n+1.0)*(dble_n+2.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};
          leakage[i](idx_dia,idx_sup) = zmult / record.sigma_r(g,2*n+1);
        }

      }
    }
  }
  MatrixPWConstCoefficient leakage_func{leakage};
  diffusion_bilinear.AddDomainIntegrator(new VectorDiffusionIntegrator{leakage_func});
  if (myid == 0) std::cout << "MATRIXPWCONSTCOEFFICIENT::leakage_func" << std::endl;

  std::vector<DenseMatrix> removal(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    removal[i].SetSize(mat_lib.ngroup()*inp.spn_eq());
    const XSRecord& record{mat_lib.mat[i].rec};

    DenseMatrix scat_mat;
    record.sigma_s.ToDenseMatrix(scat_mat);
    scat_mat.Neg();
    removal[i].AddMatrix(scat_mat, 0, 0);

    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const int idx_dia{g + n*mat_lib.ngroup()};
        removal[i](idx_dia,idx_dia) = record.sigma_r(g,n);
      }
    }
  }
  MatrixPWConstCoefficient removal_func{removal};
  diffusion_bilinear.AddDomainIntegrator(new VectorMassIntegrator{removal_func});
  if (myid == 0) std::cout << "MATRIXPWCONSTCOEFFICIENT::removal_func" << std::endl;

  if (myid == 0) std::cout << "DIFFUSION_ASSEMBLE_BEGIN" << std::endl;
  time.start("diffusion_assemble");
  diffusion_bilinear.Assemble(skip_zeros);
  diffusion_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
  diffusion_bilinear.Finalize(skip_zeros);
  time.stop("diffusion_assemble");
  if (myid == 0) std::cout << "DIFFUSION_ASSEMBLE_END" << std::endl;

  time.start("diffusion_par_assemble");
  Ahypre = diffusion_bilinear.ParallelAssemble();
  time.stop("diffusion_par_assemble");

  } // sink construction

  { // fiss construction

  std::vector<DenseMatrix> fission(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    fission[i].SetSize(mat_lib.ngroup()*inp.spn_eq()); // zero initialized
    const Material_record& mat{mat_lib.mat[i]};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const Material_isotope& isotope{mat.iso[iso_idx]};
          const XSRecord& record{isotope.rec};
          const double nden{isotope.nden};
          fission[i](g,gprime) += record.chi[g] * nden * record.nusf[gprime];
        }
      }
    }
  }

  ParBilinearForm fission_bilinear{fespace};
  MatrixPWConstCoefficient fission_func{fission};
  fission_bilinear.AddDomainIntegrator(new VectorMassIntegrator{fission_func});
  if (myid == 0) std::cout << "MATRIXPWCONSTCOEFFICIENT::fission_func" << std::endl;

  if (myid == 0) std::cout << "FISSION_ASSEMBLE_BEGIN" << std::endl;
  time.start("fission_assemble");
  fission_bilinear.Assemble(skip_zeros);
  fission_bilinear.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
  fission_bilinear.Finalize(skip_zeros);
  time.stop("fission_assemble");
  if (myid == 0) std::cout << "FISSION_ASSEMBLE_END" << std::endl;

  time.start("fission_par_assemble");
  Mhypre = fission_bilinear.ParallelAssemble();
  time.stop("fission_par_assemble");

  } // fiss construction

  time.stop(time.matrix_assembly);

  constexpr bool compute_residual{false};

  time.start("amgx_setup");
  AmgXSolver Aamgx;
  Aamgx.DefaultParameters(AmgXSolver::SOLVER, inp.verbose);
  Aamgx.ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);
  //Aamgx.InitExclusiveGPU(fespace->GetComm());
  constexpr int ndevice{1};
  Aamgx.InitMPITeams(fespace->GetComm(), ndevice);
  Aamgx.SetOperator(*Ahypre);
  if (!compute_residual) 
  {
    delete Ahypre;
    Ahypre = nullptr;
  }
  time.stop("amgx_setup");

  HypreParVector x{*Mhypre};
  x.UseDevice(true);
  x.Randomize(730); // if default seed is used, the seed is actually the time and we lose reproducability

  double k_conv{inp.tol.k*1e2+1.0};
  double x_conv{inp.tol.phi*1e2+1.0};

  double k{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer) && 
      ((k_conv > inp.tol.k) || (x_conv > inp.tol.phi)); ++iter)
  {

    time.start("old_copy");
    const double k_old{k};
    HypreParVector x_old{x}; // constructor doesn't perform deep copy
    x_old.UseDevice(true);
    x_old = x;
    time.stop("old_copy");

    time.start("calc_v");
    // v = x / \| x \|
    HypreParVector v{x}; // constructor doesn't perform deep copy
    v.UseDevice(true);
    v = x;
    const double local_x_norml2{std::pow(x.Norml2(),2)};
    double x_norml2{0.0};
    MPI_Allreduce(&local_x_norml2, &x_norml2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    x_norml2 = std::sqrt(x_norml2);
    v /= x_norml2;
    time.stop("calc_v");

    time.start("matvec");
    // b = M*v
    HypreParVector b{v}; // constructor doesn't perform deep copy
    b.UseDevice(true);
    //Mhypre->Mult(v, b); // this is performed on CPU
    Mhypre->Mult(v, b, 1.0, 0.0); // make sure we use the proper function signature
    time.stop("matvec");
    
    time.start("amgx_solve");
    // x = A^-1 * b
    Aamgx.Mult(b, x);
    time.stop("amgx_solve");

    time.start("dot_product");
    // k = v^T * x
    const double local_k{v*x};
    MPI_Allreduce(&local_k, &k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time.stop("dot_product");

    // convergence criteria
    k_conv = std::abs(k - k_old);
    x_old -= x;
    const double local_conv_norml2{std::pow(x_old.Norml2(),2)}; // NOTE: Norml2inf is not on the GPU
    MPI_Allreduce(&local_conv_norml2, &x_conv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    x_conv = std::sqrt(x_conv);

    if (compute_residual)
    {
      time.start("residual");
      // r = A*x - M*x/k
      HypreParVector r0{x};
      r0.UseDevice(true);
      Ahypre->Mult(x, r0);

      HypreParVector r1{x};
      r1.UseDevice(true);
      Mhypre->Mult(x, r1);
      r1 /= k;

      r0 -= r1;
      const double local_residual{r0.Norml2()};
      double residual{0.0};
      MPI_Allreduce(&local_residual, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      std::cout << "residual=" << residual << std::endl;
      time.stop("residual");
    }

    if (myid == 0) std::cout << "iter=" << iter
                             << " inner=" << Aamgx.GetNumIterations()
                             << " k=" << k 
                             << " dk=" << k_conv 
                             << " dx=" << x_conv 
                             << std::endl;
  }

  if (myid == 0) std::cout << "ITERATIONS_FINISHED!!!" << std::endl;

  delete Ahypre;
  delete Mhypre;

  res.keff = k;
  time.start("distribute");
  res(0).Distribute(x);
  time.stop("distribute");
  return;
}


void spn_slepc_manual(Timer& time, const Input& inp,
                       Material_library& mat_lib,
                       const Array<int>& ess_bdr, const ParMesh* const pmesh,
                       ParFiniteElementSpace* const fespace, Result& res)
{

  mat_lib.set_nmoment(inp.spn_order()+1);

  slepc_davidson_hypre_boomeramg_initialize(inp, mat_lib.ngroup()*inp.spn_eq());

  time.start(time.matrix_assembly);

  const int block_rank{mat_lib.ngroup() * inp.spn_eq()};
  const int offset{fespace->TrueVSize()}; // needed for parallel
  Array<int> offset_list(block_rank + 1);
  offset_list[0] = 0;
  for (int i{1}; i < block_rank+1; ++i)
    offset_list[i] = offset_list[i-1] + offset;

  BlockOperator diffusion_block{offset_list};
  BlockOperator fission_block{offset_list};

  std::vector<std::unique_ptr<PetscParMatrix>> matrix_list;
  matrix_list.reserve(block_rank*block_rank);

  for (int n{0}; n < inp.spn_eq(); ++n)
  {

    const double dble_n{2.0*n};
    const double xmult_next{(dble_n+1.0)*(dble_n+1.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};
    const double xmult_prev{dble_n*dble_n / (4.0*dble_n*dble_n - 1.0)};
    const double ymult{(dble_n*dble_n - dble_n) / (4.0*dble_n*dble_n - 1.0)};
    const double zmult{(dble_n+1.0)*(dble_n+2.0) / ((2.0*dble_n+1.0)*(2.0*dble_n+3.0))};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      const int idx_dia{g + mat_lib.ngroup() * n};
      const int idx_sub{idx_dia - mat_lib.ngroup()};
      const int idx_sup{idx_dia + mat_lib.ngroup()};

      Vector leakage(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        leakage(i) = xmult_next / mat_lib.mat[i].rec.sigma_r(g,2*n+1);
        if (n > 0)
          leakage(i) += xmult_prev / mat_lib.mat[i].rec.sigma_r(g,2*n-1);
      }
      PWConstCoefficient leakage_func{leakage};

      Vector removal(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
        removal[i] = mat_lib.mat[i].rec.sigma_r(g,2*n);
      PWConstCoefficient removal_func{removal};

      ParBilinearForm self_bilinear{fespace};
      self_bilinear.AddDomainIntegrator(new DiffusionIntegrator{leakage_func});
      self_bilinear.AddDomainIntegrator(new MassIntegrator{removal_func});
      self_bilinear.Assemble();
      self_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
      self_bilinear.Finalize();

      matrix_list.emplace_back(bilinear_to_petscparmatrix(self_bilinear));
      diffusion_block.SetDiagonalBlock(idx_dia, matrix_list.back().get());

      if ((n+1) < inp.spn_eq())
      {

        Vector next_leakage(pmesh->attributes.Max());
        for (int i{0}; i < pmesh->attributes.Max(); ++i)
          next_leakage(i) = zmult / mat_lib.mat[i].rec.sigma_r(g,2*n+1);
        PWConstCoefficient next_leakage_func{next_leakage};

        ParBilinearForm next_bilinear{fespace};
        next_bilinear.AddDomainIntegrator(new DiffusionIntegrator{next_leakage_func});
        next_bilinear.Assemble();
        next_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
        next_bilinear.Finalize();
        matrix_list.emplace_back(bilinear_to_petscparmatrix(next_bilinear));
        diffusion_block.SetBlock(idx_dia, idx_sup, matrix_list.back().get());
      }

      if (n > 0)
      {

        Vector prev_leakage(pmesh->attributes.Max());
        for (int i{0}; i < pmesh->attributes.Max(); ++i)
          prev_leakage(i) = ymult / mat_lib.mat[i].rec.sigma_r(g,2*n-1);
        PWConstCoefficient prev_leakage_func{prev_leakage};

        ParBilinearForm prev_bilinear{fespace};
        prev_bilinear.AddDomainIntegrator(new DiffusionIntegrator{prev_leakage_func});
        prev_bilinear.Assemble();
        prev_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
        prev_bilinear.Finalize();
        matrix_list.emplace_back(bilinear_to_petscparmatrix(prev_bilinear));
        diffusion_block.SetBlock(idx_dia, idx_sub, matrix_list.back().get());
      }

      if (n == 0)
      {

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
          fission_bilinear.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
          fission_bilinear.Finalize();

          matrix_list.emplace_back(bilinear_to_petscparmatrix(fission_bilinear));
          // block index formula valid for zeroth moment only
          fission_block.SetBlock(idx_dia, gprime, matrix_list.back().get());

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
            // block index formula valid for zeroth moment only
            diffusion_block.SetBlock(idx_dia, gprime, matrix_list.back().get());

          }

        } // gprime

      } // zeroth-moment (scatter & fiss)

    } // ngroup

  } // spn_eq

  // TODO check if the length of matrix_list is correct...

  if (inp.spn_eq() > 1)
  {
    // tricking petsc
    ConstantCoefficient zero{0.0};
    ParBilinearForm zero_bilinear{fespace};
    zero_bilinear.AddDomainIntegrator(new MassIntegrator{zero});
    zero_bilinear.Assemble();
    zero_bilinear.Finalize();
    matrix_list.emplace_back(bilinear_to_petscparmatrix(zero_bilinear));
    const int idx_fixed{mat_lib.ngroup()+1};
    for (int n{0}; n < inp.spn_eq(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const int idx_moving{g + n*mat_lib.ngroup()};
        fission_block.SetBlock(idx_fixed, idx_moving, matrix_list.back().get());
        fission_block.SetBlock(idx_moving, idx_fixed, matrix_list.back().get());
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
  PetscParMatrix M{&Mtmp, Operator::PETSC_MATAIJ}; // TODO why?

  slepc.SetOperators(A, M);

  /*
  A.Print("petsca_manual");
  M.Print("petscm_manual");
  */

  // NOTE: this call to SlepcEigenSolver::Solve() leaks.
  // It leaks 124 bytes in 6 blocks. (definitely lost)
  // It leaks 1,408 bytes in 4 blocks. (indirecly lost)
  // It seems to probably be a SLEPc issue.
  std::cout << "START" << std::endl;
  time.start(time.solver);
  slepc.Solve();
  time.stop();
  std::cout << "STOP" << std::endl;

  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  slepc.GetEigenvalue(0, res.keff);
  res.keff = 1.0 / res.keff;
  Vector tmp(fespace->GetTrueVSize()*block_rank);
  slepc.GetEigenvector(0, tmp);

  for (int n{0}; n < inp.spn_eq(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      const int base_offset{(g + mat_lib.ngroup()*n) * fespace->GetTrueVSize()};
      Vector this_sol(fespace->GetTrueVSize());
      for (int idx{0}; idx < fespace->GetTrueVSize(); ++idx)
        this_sol(idx) = tmp(idx + base_offset);
      res(g,n).Distribute(this_sol);
    }
  }

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

void epdo_power_iteration(const Quadrature* const quad, Timer& time, const Input& inp,
                          const Material_library& mat_lib,
                          const mfem::Array<int>& ess_bdr,
                          mfem::ParMesh* const pmesh,
                          mfem::ParFiniteElementSpace* const fespace, Result& angular_flux, Result& scalar_flux)
{

  // there are no essential boundary conditions for EPDO neutron transport
  Array<int> ess_tdof_list(0);

  // need a mask to list where reflective boundary conditions need to be applied
  for (int bc : inp.boundary_conditions)
  {
    if (bc == 0)
    {
      std::cout << "vacuum BCs unsupported with EPDO. not solving." << std::endl;
      return;
    }
  }

  std::vector<VectorConstantCoefficient> omega;
  omega.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    omega.emplace_back(quad->omega(n));

  std::vector<PWConstCoefficient> inv_tot;
  inv_tot.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector inv_tot_vec{pmesh->attributes.Max()};
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      inv_tot_vec(i) = 1.0/mat_lib.mat[i].rec.sigma_t[g];
    inv_tot.emplace_back(inv_tot_vec);
  }

  time.start(time.matrix_assembly);

  SPNWrapper<std::unique_ptr<ParBilinearForm>> a(mat_lib.ngroup(), quad->nd());
  a.resize();

  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      a(g,n) = std::make_unique<ParBilinearForm>(fespace);

      DenseMatrix streaming_operator{quad->nd()};
      MultVVt(quad->omega(n), streaming_operator);
      MatrixConstantCoefficient streaming_matrix{streaming_operator};
      ScalarMatrixProductCoefficient streaming{inv_tot[g], streaming_matrix};

      Vector total_operator{pmesh->attributes.Max()};
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
        total_operator(i) = mat_lib.mat[i].rec.sigma_t[g];
      PWConstCoefficient total{total_operator};

      a(g,n)->AddDomainIntegrator(new DiffusionIntegrator{streaming});
      a(g,n)->AddDomainIntegrator(new MassIntegrator{total});

      a(g,n)->AddBdrFaceIntegrator(new DGTraceIntegrator{omega[n], 0.0, 1.0});

      a(g,n)->Assemble();
      a(g,n)->Finalize();

    }
  }

  time.stop();

  std::vector<ScatteringCoefficient> scat_term;
  scat_term.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scat_term.emplace_back(g, mat_lib, scalar_flux, true); // include self-scatter

  ConstantCoefficient source{1.0};

  SPNWrapper<std::unique_ptr<ParLinearForm>> b(mat_lib.ngroup(), quad->nd());
  b.resize();
  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      b(g,n) = std::make_unique<ParLinearForm>(fespace);

      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{scat_term[g]});
      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{source});
    }
  }

  // setup preconditioner
  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  // setup solver
  GMRESSolver sol(MPI_COMM_WORLD);
  // TODO
  // sol.SetKDim(100); // need more but makes more slower
  sol.SetRelTol(inp.tol.inner_rel);
  sol.SetAbsTol(inp.tol.inner_abs);
  sol.SetMaxIter(inp.tol.max_inner);
  sol.SetPrintLevel(inp.mfem_print_level());
  sol.SetPreconditioner(amg);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < quad->nd(); ++n)
    {
      angular_flux(g,n) = 1.0;
    }
    scalar_flux(g) = 1.0;
  }

////////////////////////////////////////////////////////////////////////////////
  std::vector<std::unique_ptr<ParLinearForm>> fiss(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss[g] = std::make_unique<ParLinearForm>(fespace);

    Vector nusf_vec{pmesh->attributes.Max()};
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      nusf_vec(i) = mat_lib.mat[i].rec.nusf[g];
    PWConstCoefficient nusf_operator{nusf_vec};

    fiss[g]->AddDomainIntegrator(new DomainLFIntegrator{nusf_operator});
    fiss[g]->Assemble();
  }
////////////////////////////////////////////////////////////////////////////////

  const int myid{fespace->GetMyRank()};

  double k_conv{inp.tol.k * 10 + 0.1};
  double phi_conv{inp.tol.phi * 10 + 0.1};
  double fiss_sum_old{1.0};
  double keff_old{1.0};

  time.start(time.solver);

  for (int iter{0}; (iter < inp.tol.max_outer)
      && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)) ; ++iter)
  {

    double local_phi_conv{0.0};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      for (int n{0}; n < quad->nd(); ++n)
      {

        b(g,n)->Assemble();

        OperatorPtr A;
        Vector B, X;

        a(g,n)->FormLinearSystem(ess_tdof_list, angular_flux(g,n), *b(g,n), A, X, B);
        sol.SetOperator(*A);
        sol.Mult(B, X);
        a(g,n)->RecoverFEMSolution(X, *b(g,n), angular_flux(g,n));

      }

      // only converging the scalar flux
      ParGridFunction phi_old{scalar_flux(g)};
      calc_scalar_flux(g, angular_flux, quad, scalar_flux);
      phi_old -= scalar_flux(g);

      // need to sync memory for gpu problems
      phi_old.GetMemory().Read(MemoryClass::HOST, phi_old.Size());

      local_phi_conv = std::max(local_phi_conv, phi_old.Normlinf() / scalar_flux(g).Max());

    }

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // calculate fission sum
    double fiss_sum{0.0}, local_fiss_sum{0.0};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      local_fiss_sum += (*fiss[g])(scalar_flux(g)); // LinearForm::operator()
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (iter > 0) angular_flux.keff *= (fiss_sum / fiss_sum_old);

    k_conv = std::abs(angular_flux.keff - keff_old);

    if (myid == 0)
    {
      std::cout << iter
                << " keff=" << angular_flux.keff 
                << " dk=" << k_conv 
                << " dphi=" << phi_conv 
                << std::endl;
    }

    scalar_flux.keff = angular_flux.keff;
    keff_old = angular_flux.keff;
    fiss_sum_old = fiss_sum;

  }

  time.stop();

}

void sfent_power_iteration(const Quadrature* const quad, Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const mfem::Array<int>& ess_bdr,
                           mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace,
                           Result& angular_flux, Result& scalar_flux)
{

  std::vector<VectorConstantCoefficient> omega_term;
  omega_term.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    omega_term.emplace_back(quad->omega(n));

  std::vector<StabilityTauCoefficient> tau_term;
  tau_term.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    tau_term.emplace_back(pmesh, g, mat_lib);

  time.start(time.matrix_assembly);

  SPNWrapper<std::unique_ptr<ParBilinearForm>> a{mat_lib.ngroup(), quad->nd()};
  a.resize();
  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      a(g,n) = std::make_unique<ParBilinearForm>(fespace);

      std::vector<Vector> total_omega(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        total_omega[i].SetSize(quad->dim());
        total_omega[i] = quad->omega(n);
        total_omega[i] *= mat_lib.mat[i].rec.sigma_t[g];
      }
      VectorPWConstCoefficient total_omega_term{total_omega};
      ScalarVectorProductCoefficient total_omega_func{tau_term[g], total_omega_term};

      // streaming term

      DenseMatrix omega_matrix(quad->dim()); // TODO move this to the outer loop
      MultVVt(quad->omega(n), omega_matrix);
      MatrixConstantCoefficient omega_matrix_term{omega_matrix};
      ScalarMatrixProductCoefficient omega_matrix_func{tau_term[g], omega_matrix_term};

      a(g,n)->AddDomainIntegrator(new ConvectionIntegrator{omega_term[n], -1.0});
      a(g,n)->AddDomainIntegrator(new DiffusionIntegrator{omega_matrix_func}); // stability (SUPG+GLS)
      //a(g,n)->AddDomainIntegrator(new ConvectionIntegrator{total_omega_func}); // stability (GLS)

      // removal term

      Vector total{pmesh->attributes.Max()};
      Vector total_total{pmesh->attributes.Max()};
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        total(i) = mat_lib.mat[i].rec.sigma_t[g];
        total_total(i) = total(i) * mat_lib.mat[i].rec.sigma_t[g];
      }
      PWConstCoefficient total_func{total};
      PWConstCoefficient total_total_func{total_total};
      ProductCoefficient total_total_stab_func{tau_term[g], total_total_func};

      a(g,n)->AddDomainIntegrator(new MassIntegrator{total_func});
      a(g,n)->AddDomainIntegrator(
          new TransposeIntegrator(
          new ConvectionIntegrator(total_omega_func))); // stability (SUPG+GLS)
      //a(g,n)->AddDomainIntegrator(new MassIntegrator{total_total_stab_func}); // stability (GLS)


      // TODO I think the TransposeIntegrator is unnecessary
      a(g,n)->AddBdrFaceIntegrator(
        new TransposeIntegrator{
        new DGTraceIntegrator{omega_term[n], 0.5, 0.5}});

      a(g,n)->Assemble();
      a(g,n)->Finalize();
    }
  }

  time.stop();

  std::vector<FissionCoefficient> fiss_term;
  fiss_term.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    fiss_term.emplace_back(g, angular_flux.keff, mat_lib, scalar_flux, inp.flux_free_chi);

  std::vector<ScatteringCoefficient> scat_term;
  scat_term.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scat_term.emplace_back(g, mat_lib, scalar_flux, true); // include self-scatter

  SPNWrapper<IncomingFluxCoefficient> incoming_flux_term{mat_lib.ngroup(), quad->nd()};
  incoming_flux_term.reserve();
  for (int n{0}; n < quad->nd(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      incoming_flux_term.emplace_back(g, n, angular_flux, quad);

  SPNWrapper<ScalarVectorProductCoefficient> fiss_stab(mat_lib.ngroup(), quad->nd());
  SPNWrapper<ScalarVectorProductCoefficient> scat_stab(mat_lib.ngroup(), quad->nd());
  fiss_stab.reserve();
  scat_stab.reserve();
  SPNWrapper<ScalarVectorProductCoefficient> fiss_stab_term(mat_lib.ngroup(), quad->nd());
  SPNWrapper<ScalarVectorProductCoefficient> scat_stab_term(mat_lib.ngroup(), quad->nd());
  fiss_stab_term.reserve();
  scat_stab_term.reserve();
  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      fiss_stab.emplace_back(fiss_term[g], omega_term[n]);
      scat_stab.emplace_back(scat_term[g], omega_term[n]);
      fiss_stab_term.emplace_back(tau_term[g], fiss_stab(g,n));
      scat_stab_term.emplace_back(tau_term[g], scat_stab(g,n));
    }
  }

  ConstantCoefficient source{1.0}; // note: 1.0 is the default value
  //FunctionCoefficient source{psi_analytic_source};
  //FunctionCoefficient source{point_source};

  std::vector<ProductCoefficient> source_stab;
  source_stab.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    source_stab.emplace_back(source, tau_term[g]);
  SPNWrapper<ScalarVectorProductCoefficient> source_stab_term(mat_lib.ngroup(), quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      source_stab_term.emplace_back(source_stab[g], omega_term[n]);

  std::vector<Vector> total_vec(mat_lib.ngroup());
  std::vector<PWConstCoefficient> total_vec_func;
  std::vector<ProductCoefficient> total_vec_stab_func;
  std::vector<ProductCoefficient> total_vec_stab_source_func;
  total_vec_func.reserve(mat_lib.ngroup());
  total_vec_stab_func.reserve(mat_lib.ngroup());
  total_vec_stab_source_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    total_vec[g].SetSize(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      total_vec[g](i) = mat_lib.mat[i].rec.sigma_t[g];
    }
    total_vec_func.emplace_back(total_vec[g]);
    total_vec_stab_func.emplace_back(total_vec_func.back(), tau_term[g]);
    total_vec_stab_source_func.emplace_back(total_vec_stab_func.back(), source);
  }


  SPNWrapper<std::unique_ptr<ParLinearForm>> b{mat_lib.ngroup(), quad->nd()};
  b.resize();

  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      b(g,n) = std::make_unique<ParLinearForm>(fespace);

//      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{fiss_term[g]});
//      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{fiss_stab_term(g,n)});

      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{scat_term[g]});
      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{scat_stab_term(g,n)}); // stability (SUPG+GLS)

      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{source});
      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{source_stab_term(g,n)}); // stability (SUPG+GLS)
//      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{total_vec_stab_source_func[g]}); // stability (GLS)

      /*
      // TODO restrict to reflective boundaries only
      b(g,n)->AddBdrFaceIntegrator(
        new BoundaryFlowIntegrator{incoming_flux_term(g,n), omega_term[n], -1.0, -0.5});
      */

    }
  }

  // setup preconditioner
  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  // setup linear solver
  GMRESSolver gmres(MPI_COMM_WORLD);
  gmres.SetRelTol(inp.tol.inner_rel);
  gmres.SetAbsTol(inp.tol.inner_abs);
  gmres.SetMaxIter(inp.tol.max_inner);
  gmres.SetPrintLevel(inp.mfem_print_level());
  gmres.SetPreconditioner(amg);
  gmres.SetKDim(10000); // number of iterations to perform between restarts, default=50
  
  // get the process id so that only the zero-process writes to the terminal
  const int myid{fespace->GetMyRank()};

  // initialize "old" data
  double keff_old{angular_flux.keff};
  double fiss_sum{1.0};
  double fiss_sum_old{1.0};

  // initialize convergence criteria,
  // making sure we enter the loop at least once
  double k_conv{inp.tol.k * 1e1 + 1e-5};
  double phi_conv{inp.tol.phi * 1e1 + 1e-5};

  // TODO this should *really* go away! It costs a lot of extra memory.
  // build "outline" of fission source integrator for each energy group
  std::vector<PWConstCoefficient> nusf_coeff;
  std::vector<GridFunctionCoefficient> phi_coeff;
  std::vector<ProductCoefficient> fiss_product_coeff;
  std::vector<std::unique_ptr<ParLinearForm>> fiss(mat_lib.ngroup());

  nusf_coeff.reserve(mat_lib.ngroup());
  phi_coeff.reserve(mat_lib.ngroup());
  fiss_product_coeff.reserve(mat_lib.ngroup());

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Vector fission(pmesh->attributes.Max());
    for (int i{0}; i < fission.Size(); ++i)
      fission[i] = mat_lib.mat[i].rec.nusf[g];
    nusf_coeff.emplace_back(fission);

    phi_coeff.emplace_back(&scalar_flux(g));

    fiss_product_coeff.emplace_back(nusf_coeff[g], phi_coeff[g]);

    fiss[g] = std::make_unique<ParLinearForm>(fespace);
    fiss[g]->AddDomainIntegrator(new DomainLFIntegrator{fiss_product_coeff[g]});

  }

  // TODO move this
  Array<int> ess_tdof_list(0); // there are no essential bcs

  // TODO initialize better...
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < quad->nd(); ++n)
    {
      angular_flux(g,n) = 0.0;
    }
    scalar_flux(g) = 1.0;
  }


  time.start(time.solver);
  for (int iter{0}; (iter < inp.tol.max_outer) && 
      ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    double local_fiss_sum{0.0};
    double local_phi_conv{0.0};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {

        omega_x = quad->omega(n)(0);
        omega_y = quad->omega(n)(1);
        omega_z = quad->omega(n)(2);

        b(g,n)->Assemble();

        OperatorPtr A;
        Vector B;
        Vector X;

        a(g,n)->FormLinearSystem(ess_tdof_list, angular_flux(g,n), *b(g,n), A, X, B);
        gmres.SetOperator(*A);
        gmres.Mult(B, X);
        a(g,n)->RecoverFEMSolution(X, *b(g,n), angular_flux(g,n));

      }

      // only converging the scalar flux
      ParGridFunction phi_old{scalar_flux(g)};
      calc_scalar_flux(g, angular_flux, quad, scalar_flux);

      // calc (local) scalar flux convergence criteria
      phi_old -= scalar_flux(g);
      phi_old.GetMemory().Read(MemoryClass::HOST, phi_old.Size());
      local_phi_conv = std::max(local_phi_conv, phi_old.Normlinf() / scalar_flux(g).Max());

      // calculate fission sum
      fiss[g]->Assemble();
      local_fiss_sum += fiss[g]->Sum();

    }

    // update keff
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (iter > 0) angular_flux.keff *= fiss_sum / fiss_sum_old;

    // update scalar flux convergence criteria
    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // update keff convergence criteria
    k_conv = std::abs(angular_flux.keff - keff_old);

    if (myid == 0)
    {
      std::cout << "iter=" << iter << " keff=" << angular_flux.keff 
                << " dk=" << k_conv << " dphi=" << phi_conv << std::endl;
    }

    keff_old = angular_flux.keff;
    fiss_sum_old = fiss_sum;

  }

  time.stop();

  return;
}

void saaf_power_iteration(const Quadrature* const quad,
                          Timer& time,
                          const Input& inp,
                          const Material_library& mat_lib,
                          ParMesh* const pmesh,
                          ParFiniteElementSpace* const fespace,
                          Result& angular_flux,
                          Result& scalar_flux)
{

  std::vector<VectorConstantCoefficient> omega;
  omega.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    omega.emplace_back(quad->omega(n));

  std::vector<PWConstCoefficient> inv_tot;
  inv_tot.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector inv_tot_vec{pmesh->attributes.Max()};
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      inv_tot_vec(i) = 1.0/mat_lib.mat[i].rec.sigma_t[g];
    inv_tot.emplace_back(inv_tot_vec);
  }

  SPNWrapper<ScalarVectorProductCoefficient> inv_tot_omega{mat_lib.ngroup(), quad->nd()};
  inv_tot_omega.reserve();
  for (int n{0}; n < quad->nd(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      inv_tot_omega.emplace_back(inv_tot[g], omega[n]);

  time.start(time.matrix_assembly);

  SPNWrapper<std::unique_ptr<ParBilinearForm>> a{mat_lib.ngroup(), quad->nd()};
  a.resize();

  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      a(g,n) = std::make_unique<ParBilinearForm>(fespace);

      DenseMatrix streaming_operator{quad->dim()};
      MultVVt(quad->omega(n), streaming_operator);
      MatrixConstantCoefficient streaming_matrix{streaming_operator};
      ScalarMatrixProductCoefficient streaming{inv_tot[g], streaming_matrix};

      Vector total_operator{pmesh->attributes.Max()};
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
        total_operator(i) = mat_lib.mat[i].rec.sigma_t[g];
      PWConstCoefficient total{total_operator};

      a(g,n)->AddDomainIntegrator(new DiffusionIntegrator{streaming});
      a(g,n)->AddDomainIntegrator(new MassIntegrator{total});

      a(g,n)->AddBdrFaceIntegrator(new DGTraceIntegrator{omega[n], 1.0});

      a(g,n)->Assemble();
      a(g,n)->Finalize();

    }
  }

  const int integration_order{fespace->GetOrder(0)+1};
  QuadratureSpace qspace{const_cast<ParMesh*>(pmesh), integration_order};

  std::vector<QuadratureFunction> fiss_quad_func;
  std::vector<QuadratureFunction> upscatter_quad_func, dnscatter_quad_func;
  fiss_quad_func.reserve(mat_lib.ngroup());
  upscatter_quad_func.reserve(mat_lib.ngroup());
  dnscatter_quad_func.reserve(mat_lib.ngroup());

  std::vector<QuadratureFunctionCoefficient> fiss_quad_coeff;
  std::vector<QuadratureFunctionCoefficient> upscatter_quad_coeff, dnscatter_quad_coeff;
  fiss_quad_coeff.reserve(mat_lib.ngroup());
  upscatter_quad_coeff.reserve(mat_lib.ngroup());
  dnscatter_quad_coeff.reserve(mat_lib.ngroup());

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_quad_func.emplace_back(&qspace);
    upscatter_quad_func.emplace_back(&qspace);
    dnscatter_quad_func.emplace_back(&qspace);
    fiss_quad_coeff.emplace_back(fiss_quad_func.back());
    upscatter_quad_coeff.emplace_back(upscatter_quad_func.back());
    dnscatter_quad_coeff.emplace_back(dnscatter_quad_func.back());
  }

  std::vector<GridFunctionCoefficient> scalar_flux_coeff;
  scalar_flux_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux_coeff.emplace_back(&scalar_flux(g));

  std::vector<PWConstCoefficient> self_scatter_xs;
  self_scatter_xs.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector self_scatter(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      self_scatter(i) = mat_lib.mat[i].rec.sigma_s.Elem(g,g);
    self_scatter_xs.emplace_back(self_scatter);
  }

  std::vector<ProductCoefficient> self_scatter_src;
  self_scatter_src.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    self_scatter_src.emplace_back(scalar_flux_coeff[g], self_scatter_xs[g]);

  std::vector<SumCoefficient> intermediate_scat_operator, scat_operator;
  intermediate_scat_operator.reserve(mat_lib.ngroup());
  scat_operator.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    intermediate_scat_operator.emplace_back(upscatter_quad_coeff[g], dnscatter_quad_coeff[g]);
    scat_operator.emplace_back(intermediate_scat_operator.back(), self_scatter_src[g]);
  }

  SPNWrapper<ScalarVectorProductCoefficient> inv_tot_omega_scat_operator{mat_lib.ngroup(), quad->nd()};
  inv_tot_omega_scat_operator.reserve();
  for (int n{0}; n < quad->nd(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      inv_tot_omega_scat_operator.emplace_back(scat_operator[g], inv_tot_omega(g,n));

  SPNWrapper<ScalarVectorProductCoefficient> inv_tot_omega_fiss_operator{mat_lib.ngroup(), quad->nd()};
  inv_tot_omega_fiss_operator.reserve();
  for (int n{0}; n < quad->nd(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      inv_tot_omega_fiss_operator.emplace_back(fiss_quad_coeff[g], inv_tot_omega(g,n));

  SPNWrapper<IncomingFluxCoefficient> incoming_flux_coeff{mat_lib.ngroup(), quad->nd()};
  incoming_flux_coeff.reserve();
  for (int n{0}; n < quad->nd(); ++n)
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      incoming_flux_coeff.emplace_back(g, n, angular_flux, quad);

  Array<int> ess_tdof_list(0); // no essential bcs
  Array<int> reflective_condition(pmesh->bdr_attributes.Max());
  reflective_condition = 0;
  for (int i{0}; (i < inp.boundary_conditions.Size()) && (i < reflective_condition.Size()); ++i)
  {
    if (inp.boundary_conditions[i] == 0) // zero represents reflective
      reflective_condition[i] = 1;
  }

  SPNWrapper<std::unique_ptr<ParLinearForm>> b{mat_lib.ngroup(), quad->nd()};
  b.resize();

  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      b(g,n) = std::make_unique<ParLinearForm>(fespace);

      b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{upscatter_quad_coeff[g], nullptr});
      b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{dnscatter_quad_coeff[g], nullptr});
      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{self_scatter_src[g]});
      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{inv_tot_omega_scat_operator(g,n)});

      b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{fiss_quad_coeff[g], nullptr});
      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{inv_tot_omega_fiss_operator(g,n)});

      b(g,n)->AddBdrFaceIntegrator(
        new BoundaryFlowIntegrator{incoming_flux_coeff(g,n), omega[n], -1.0}, 
        reflective_condition);

    }
  }

  // setup preconditioner
  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  // setup solver
  CGSolver sol(MPI_COMM_WORLD);
  sol.SetRelTol(inp.tol.inner_rel);
  sol.SetAbsTol(inp.tol.inner_abs);
  sol.SetMaxIter(inp.tol.max_inner);
  sol.SetPrintLevel(inp.mfem_print_level());
  sol.SetPreconditioner(amg);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < quad->nd(); ++n)
    {
      angular_flux(g,n) = 1.0;
    }
    scalar_flux(g) = 1.0;
  }

////////////////////////////////////////////////////////////////////////////////
  std::vector<std::unique_ptr<ParLinearForm>> fiss(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss[g] = std::make_unique<ParLinearForm>(fespace);

    Vector nusf_vec{pmesh->attributes.Max()};
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      nusf_vec(i) = mat_lib.mat[i].rec.nusf[g];
    PWConstCoefficient nusf_operator{nusf_vec};

    fiss[g]->AddDomainIntegrator(new DomainLFIntegrator{nusf_operator});
    fiss[g]->Assemble();
  }
////////////////////////////////////////////////////////////////////////////////

  const int myid{fespace->GetMyRank()};

  // these variables need to be carried between iterations
  double k_conv{inp.tol.k * 1e1 + 1e-6};
  double phi_conv{inp.tol.phi * 1e1 + 1e-6};
  double keff_old{angular_flux.keff};
  double fiss_sum_old{1.0};

  QuadratureFunction fiss_src_old{&qspace};
  fiss_src_old = 1.0;

  for (int iter{0}; (iter < inp.tol.max_outer) && 
      ((phi_conv > inp.tol.phi) || (k_conv > inp.tol.k)) ; ++iter)
  {

    calc_fiss_upscatter(scalar_flux, mat_lib, 
      fiss_quad_func, upscatter_quad_func, inp.flux_free_chi);
    for (QuadratureFunction& qf : fiss_quad_func)
      qf /= angular_flux.keff;

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      calc_downscatter(g, scalar_flux, mat_lib, dnscatter_quad_func[g]);

      // TODO this criteria should be in user input
      double scattering_conv{1e6};
      for (int scattering_iter{0};
          (scattering_iter < inp.tol.max_inner) && 
          (scattering_conv > inp.tol.inner_rel);
          ++scattering_iter)
      {

        for (int n{0}; n < quad->nd(); ++n)
        {
          b(g,n)->Assemble();

          OperatorPtr A;
          Vector B, X;

          a(g,n)->FormLinearSystem(ess_tdof_list, angular_flux(g,n), *b(g,n), A, X, B);
          sol.SetOperator(*A);
          sol.Mult(B, X);
          a(g,n)->RecoverFEMSolution(X, *b(g,n), angular_flux(g,n));
        } // ndir

        ParGridFunction phi_old{scalar_flux(g)};
        calc_scalar_flux(g, angular_flux, quad, scalar_flux);
        phi_old -= scalar_flux(g);

        // need to sync memory for gpu problems
        phi_old.HostRead();
        const double local_scattering_conv{
          phi_old.Normlinf() / scalar_flux(g).Normlinf()};
        MPI_Allreduce(&local_scattering_conv, &scattering_conv, 1, MPI_DOUBLE, 
          MPI_MAX, MPI_COMM_WORLD);

        if (myid == 0) 
        {
          std::cout << "iter=" << iter 
                    << " g=" << g 
                    << " scattering_iter=" << scattering_iter 
                    << " scattering_conv=" << scattering_conv 
                    << std::endl;
        }

      } // scattering iterations

    } // ngroup

    // calculate fission sum
    double fiss_sum{0.0}, local_fiss_sum{0.0};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      local_fiss_sum += (*fiss[g])(scalar_flux(g)); // LinearForm::operator()
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    QuadratureFunction fiss_src{&qspace};
    calc_fiss_src(scalar_flux, mat_lib, fiss_src);
    fiss_src_old -= fiss_src;
    const double local_phi_conv{fiss_src_old.Normlinf() / fiss_src.Normlinf()};
    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (iter > 0) angular_flux.keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(angular_flux.keff - keff_old);

    if (myid == 0)
    {
      std::cout << iter 
                << " keff=" << angular_flux.keff
                << " dk=" << k_conv 
                << " dphi=" << phi_conv 
                << std::endl;
    }

    scalar_flux.keff = angular_flux.keff; // copy for convenience
    // store old values
    keff_old = angular_flux.keff;
    fiss_sum_old = fiss_sum;
    fiss_src_old = fiss_src;

  }

  return;
}

void saaf_slepc_vector(const Quadrature* const quad,
                       Timer& time,
                       const Input& inp,
                       const Material_library& mat_lib,
                       ParMesh* const pmesh,
                       ParFiniteElementSpace* const fespace,
                       Result& angular_flux,
                       Result& scalar_flux)
{

  const int myid{fespace->GetMyRank()};
  const int block_rank{mat_lib.ngroup()*quad->nd()};

  slepc_davidson_hypre_boomeramg_initialize(inp, quad->nd()*mat_lib.ngroup());

  std::vector<VectorConstantCoefficient> omega;
  omega.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    omega.emplace_back(quad->omega(n));

  std::vector<MatrixConstantCoefficient> omega_matrix;
  omega_matrix.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    DenseMatrix omega_operator(quad->dim());
    MultVVt(quad->omega(n), omega_operator);
    omega_matrix.emplace_back(omega_operator);
  }

  std::vector<ConstantCoefficient> weight;
  weight.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    weight.emplace_back(quad->weight(n));

  std::vector<PWConstCoefficient> tot;
  tot.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector tot_vec(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      tot_vec(i) = mat_lib.mat[i].rec.sigma_t[g];
    tot.emplace_back(tot_vec);
  }

  std::vector<PowerCoefficient> inv_tot;
  inv_tot.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    inv_tot.emplace_back(tot[g], -1.0);
  }

  std::vector<ScalarVectorProductCoefficient> inv_tot_omega;
  inv_tot_omega.reserve(quad->nd()*mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      inv_tot_omega.emplace_back(inv_tot[g], omega[n]);

  std::vector<ScalarMatrixProductCoefficient> inv_tot_omega_matrix;
  inv_tot_omega_matrix.reserve(quad->nd()*mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      inv_tot_omega_matrix.emplace_back(inv_tot[g], omega_matrix[n]);

  time.start(time.matrix_assembly);

  ParBilinearForm sink_bilinear{fespace}, fission_bilinear{fespace};

  std::vector<MassIntegrator> tot_integrator;
  tot_integrator.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    tot_integrator.emplace_back(tot[g]);

  std::vector<DGTraceIntegrator> omega_bdr_integrator;
  omega_bdr_integrator.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    omega_bdr_integrator.emplace_back(omega[n], 1.0);

  std::vector<DiffusionIntegrator> leakage_integrator;
  leakage_integrator.reserve(quad->nd()*mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      leakage_integrator.emplace_back(inv_tot_omega_matrix[n+g*quad->nd()]);

  std::unique_ptr<VectorIntegratorWrapper> sink_integrator{
    std::make_unique<VectorIntegratorWrapper>(block_rank, 2*block_rank)};
  std::unique_ptr<VectorIntegratorWrapper> sink_bdr_integrator{
    std::make_unique<VectorIntegratorWrapper>(block_rank, block_rank)};

  // This ordering is advantageous for problems with more angles than energy
  // groups. This is the case for the vast majority of transport problems.
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < quad->nd(); ++n)
    {
      const int idx_dia{n + g*quad->nd()};
      sink_integrator->AddIntegrator(&leakage_integrator[idx_dia], idx_dia, false);
      sink_integrator->AddIntegrator(&tot_integrator[g], idx_dia, false);
      sink_bdr_integrator->AddIntegrator(&omega_bdr_integrator[n], idx_dia, false);
    }
  }

  std::vector<Vector> sigma_s_vec(mat_lib.ngroup()*mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    {
      const int gidx{gprime + g*mat_lib.ngroup()};
      sigma_s_vec[gidx].SetSize(pmesh->attributes.Max());
      sigma_s_vec[gidx] = 0.0;
    }
  }

  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    const SparseMatrix& scat_mat{mat_lib.mat[i].rec.sigma_s};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      Vector scat_dat;
      Array<int> scat_col;
      scat_mat.GetRow(g, scat_col, scat_dat);
      for (int gidx{0}; gidx < scat_col.Size(); ++gidx)
      {
        const int gprime{scat_col[gidx]};
        sigma_s_vec[gprime + g*mat_lib.ngroup()](i) = -scat_dat(gidx);
      }
    }
  }

  std::vector<PWConstCoefficient> sigma_s;
  sigma_s.reserve(mat_lib.ngroup()*mat_lib.ngroup());
  for (Vector& vec : sigma_s_vec)
    sigma_s.emplace_back(vec);

  std::vector<ProductCoefficient> scat;
  std::vector<MassIntegrator> scat_integrator;
  scat.reserve(mat_lib.ngroup()*mat_lib.ngroup()*quad->nd());
  scat_integrator.reserve(mat_lib.ngroup()*mat_lib.ngroup()*quad->nd());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        scat.emplace_back(weight[n], sigma_s[gprime + g*mat_lib.ngroup()]); // TODO i just swapped indices
        scat_integrator.emplace_back(scat.back());
      }
    }
  }

  std::vector<ScalarVectorProductCoefficient> scat_saaf;
  scat_saaf.reserve(block_rank*block_rank);
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
        for (int nprime{0}; nprime < quad->nd(); ++nprime)
          scat_saaf.emplace_back(
            scat[nprime + gprime*quad->nd() + g*quad->nd()*mat_lib.ngroup()],
            inv_tot_omega[n + g*quad->nd()]);

  std::unique_ptr<VectorIntegratorWrapper> scat_bfi{
    std::make_unique<VectorIntegratorWrapper>(block_rank, block_rank*block_rank)};
  std::unique_ptr<VectorIntegratorWrapper> scat_saaf_bfi{
    std::make_unique<VectorIntegratorWrapper>(block_rank, block_rank*block_rank)};
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < quad->nd(); ++n)
    {
      const int this_row{n + g*quad->nd()};
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        for (int nprime{0}; nprime < quad->nd(); ++nprime)
        {
          const int this_col{nprime + gprime*quad->nd()};
          const int idx{nprime + gprime*quad->nd() + 
            n*quad->nd()*mat_lib.ngroup() + 
            g*quad->nd()*mat_lib.ngroup()*quad->nd()};
          scat_bfi->AddIntegrator(
            &scat_integrator[nprime + gprime*quad->nd() + g*quad->nd()*mat_lib.ngroup()],
            this_row, this_col, false);
          scat_saaf_bfi->AddIntegrator(new ConvectionIntegrator{scat_saaf[idx]},
            this_row, this_col);
        }
      }
    }
  }

  sink_bilinear.AddDomainIntegrator(sink_integrator.release());
  sink_bilinear.AddBdrFaceIntegrator(sink_bdr_integrator.release());

  sink_bilinear.AddDomainIntegrator(scat_bfi.release());
  sink_bilinear.AddDomainIntegrator(scat_saaf_bfi.release());

  std::cout << "SINK_ASSENBLE_BEGIN id=" << myid << std::endl;
  sink_bilinear.Assemble();
  sink_bilinear.Finalize(); // no essential boundary conditions (maybe)
  std::cout << "SINK_ASSEMBLE_END id=" << myid << std::endl;

  /*
  sigma_s_vec.clear(); // TODO offload
  sigma_s.clear();
  scat.clear();
  scat_saaf.clear();
  */

  std::vector<Vector> chi_nusf_vec(mat_lib.ngroup()*mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    {
      const int gidx{gprime + g*mat_lib.ngroup()};
      chi_nusf_vec[gidx].SetSize(pmesh->attributes.Max());
      chi_nusf_vec[gidx] = 0.0;
    }
  }

  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    const Material_record& mat{mat_lib.mat[i]};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const Material_isotope& isotope{mat.iso[iso_idx]};
          const XSRecord& record{isotope.rec};
          const double nden{isotope.nden};
          chi_nusf_vec[gprime + g*mat_lib.ngroup()](i) += record.chi[g] * nden * record.nusf[gprime];
        }
      }
    }
  }

  std::vector<PWConstCoefficient> chi_nusf;
  chi_nusf.reserve(mat_lib.ngroup()*mat_lib.ngroup());
  for (Vector& vec : chi_nusf_vec)
    chi_nusf.emplace_back(vec);

  std::vector<ProductCoefficient> fiss;
  std::vector<MassIntegrator> fiss_integrator;
  fiss.reserve(mat_lib.ngroup()*mat_lib.ngroup()*quad->nd());
  fiss_integrator.reserve(mat_lib.ngroup()*mat_lib.ngroup()*quad->nd());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        fiss.emplace_back(weight[n], chi_nusf[gprime + g*mat_lib.ngroup()]);
        fiss_integrator.emplace_back(fiss.back());
      }
    }
  }

  std::vector<ScalarVectorProductCoefficient> fiss_saaf;
  fiss_saaf.reserve(block_rank*block_rank);
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
        for (int nprime{0}; nprime < quad->nd(); ++nprime)
          fiss_saaf.emplace_back(
            fiss[nprime + gprime*quad->nd() + g*quad->nd()*mat_lib.ngroup()],
            inv_tot_omega[n + g*quad->nd()]);

  std::unique_ptr<VectorIntegratorWrapper> fiss_bfi{
    std::make_unique<VectorIntegratorWrapper>(block_rank, block_rank*block_rank)};
  std::unique_ptr<VectorIntegratorWrapper> fiss_saaf_bfi{
    std::make_unique<VectorIntegratorWrapper>(block_rank, block_rank*block_rank)};
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int n{0}; n < quad->nd(); ++n)
    {
      const int this_row{n + g*quad->nd()};
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        for (int nprime{0}; nprime < quad->nd(); ++nprime)
        {
          const int this_col{nprime + gprime*quad->nd()};
          const int idx{nprime + gprime*quad->nd() + 
            n*quad->nd()*mat_lib.ngroup() + 
            g*quad->nd()*mat_lib.ngroup()*quad->nd()};
          fiss_bfi->AddIntegrator(
            &fiss_integrator[nprime + gprime*quad->nd() + g*quad->nd()*mat_lib.ngroup()],
            this_row, this_col, false);
          fiss_saaf_bfi->AddIntegrator(new ConvectionIntegrator{fiss_saaf[idx]},
            this_row, this_col);
        }
      }
    }
  }

  fission_bilinear.AddDomainIntegrator(fiss_saaf_bfi.release());
  fission_bilinear.AddDomainIntegrator(fiss_bfi.release());

  std::cout << "FISSION_ASSEMBLY_BEGIN id=" << myid << std::endl;
  fission_bilinear.Assemble();
  fission_bilinear.Finalize();
  std::cout << "FISSION_ASSEMBLY_END id=" << myid << std::endl;

  /*
  chi_nusf_vec.clear(); // TODO offload
  chi_nusf.clear();
  fiss.clear();
  fiss_saaf.clear();
  */

  time.stop();

  if (myid == 0) std::cout << "PETSC_TRANSLATION_BEGIN" << std::endl;
  time.start("petsc translation");
  PetscParMatrix *A{nullptr}, *M{nullptr};
  {
    OperatorHandle Ah{Operator::PETSC_MATAIJ};
    sink_bilinear.ParallelAssemble(Ah);
    Ah.Get(A);
    Ah.SetOperatorOwner(false);

    OperatorHandle Mh{Operator::PETSC_MATAIJ};
    fission_bilinear.ParallelAssemble(Mh);
    Mh.Get(M);
    Mh.SetOperatorOwner(false);
  }
  time.stop();
  if (myid == 0) std::cout << "PETSC_TRANSLATION_END" << std::endl;

  time.start("slepc setup");
  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(inp.tol.phi * inp.tol.k);
  slepc.SetMaxIter(inp.tol.max_outer);
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
  slepc.SetOperators(*A, *M);
  time.stop();

  // TODO consider restarting

  if (myid == 0) std::cout << "SOLVE_BEGIN" << std::endl;
  time.start(time.solver);
  slepc.Solve();
  time.stop();
  if (myid == 0) std::cout << "SOLVE_END" << std::endl;

  delete A;
  delete M;
  
  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  slepc.GetEigenvalue(0, angular_flux.keff);
  angular_flux.keff = 1.0 / angular_flux.keff;
  scalar_flux.keff = angular_flux.keff;
  Vector tmp(fespace->GetTrueVSize());
  slepc.GetEigenvector(0, tmp);

  // each group/moment is currently stored as a vector quantity
  // parsing this is then handled by ParaView
  // for dimensions <= 3, the vector is listed {x,y,z}
  // for higher dimensions, the vector is listed {0,1,2,3,4,...}
  angular_flux(0).Distribute(tmp);

  // flip the sign of the solution such that the scalar flux in group=0 is positive
  tmp.Destroy();
  angular_flux(0).GetNodalValues(tmp);
  if (std::abs(tmp.Max()) < std::abs(tmp.Min()))
    angular_flux(0).Neg();

  // TODO figure out how to calculate scalar flux

  return;
}

void saaf_true_slepc_vector(const Quadrature* const quad,
                       Timer& time,
                       const Input& inp,
                       const Material_library& mat_lib,
                       Result& angular_flux)
{

  ParFiniteElementSpace* const fespace{angular_flux(0).ParFESpace()};
  ParMesh* const pmesh{fespace->GetParMesh()};

  const int myid{fespace->GetMyRank()};
  const int block_rank{mat_lib.ngroup()*quad->nd()}; // aka vdim
  constexpr int skip_zeros{2};

  slepc_davidson_hypre_boomeramg_initialize(inp, quad->nd()*mat_lib.ngroup());

  time.start(time.matrix_assembly);

  HypreParMatrix *Ahypre{nullptr}, *Mhypre{nullptr};

  { // sink construction

  std::vector<DenseMatrix> omega_operator(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    omega_operator[n].SetSize(quad->dim());
    MultVVt(quad->omega(n), omega_operator[n]);
  }

  ParBilinearForm sink_bilinear{fespace};

  std::vector<Tensor<mfem::DenseMatrix>> leakage(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    const XSRecord& rec{mat_lib.mat[i].rec};
    leakage[i].reserve(block_rank);
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        const int dia{n + g*quad->nd()};
        omega_operator[n] *= 1.0/rec.sigma_t[g];
        leakage[i].AddElement(omega_operator[n], dia);
        omega_operator[n] *= rec.sigma_t[g];
      }
    }
  }
  MatrixTensorPWConstCoefficient leakage_func{block_rank, leakage};
  sink_bilinear.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});

  Tensor<mfem::Vector> boundary;
  boundary.reserve(block_rank);
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      boundary.AddElement(quad->omega(n), n+g*quad->nd());
  VectorTensorConstantCoefficient boundary_func{block_rank, boundary};
  sink_bilinear.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{boundary_func, 1.0});

  std::vector<DenseMatrix> removal(pmesh->attributes.Max());
  std::vector<Tensor<mfem::Vector>> removal_saaf(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    removal[i].SetSize(block_rank);
    removal_saaf[i].reserve(block_rank*block_rank);
    const XSRecord& record{mat_lib.mat[i].rec};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      Array<int> col_idx;
      Vector scat_dat;
      record.sigma_s.GetRow(g, col_idx, scat_dat);

      for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
      {
        const int gprime{col_idx[gidx]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int row{n + g*quad->nd()};
          for (int nprime{0}; nprime < quad->nd(); ++nprime)
          {
            const int col{nprime + gprime*quad->nd()};
            const double val{-scat_dat(gidx)*quad->weight(nprime)};
            removal[i](row,col) = val;

            Vector v{quad->omega(n)};
            v /= record.sigma_t[g];
            v *= val;
            removal_saaf[i].AddElement(v, row, col);
          } // nprime < quad->nd()
        } // n < quad->nd()
      } // gidx < col_idx.Size()

      for (int n{0}; n < quad->nd(); ++n)
      {
        const int dia{n + g*quad->nd()};
        removal[i](dia,dia) += record.sigma_t[g];
      } // n < quad->nd()

    } // g < mat_lib.ngroup()
  } // i < pmesh->attributes.Max()
  MatrixPWConstCoefficient removal_func{removal};
  VectorTensorPWConstCoefficient removal_saaf_func{block_rank, removal_saaf};

  sink_bilinear.AddDomainIntegrator(new VectorMassIntegrator{removal_func});
  sink_bilinear.AddDomainIntegrator(new VectorConvectionIntegrator{removal_saaf_func});

  std::cout << "SINK_ASSENBLE_BEGIN id=" << myid << std::endl;
  sink_bilinear.Assemble(skip_zeros);
  sink_bilinear.Finalize(skip_zeros); // no essential boundary conditions (maybe)
  std::cout << "SINK_ASSEMBLE_END id=" << myid << std::endl;

  if (myid == 0) std::cout << "SINK_TRANSLATION_BEGIN" << std::endl;
  Ahypre = sink_bilinear.ParallelAssemble();
  if (myid == 0) std::cout << "SINK_TRANSLATION_END" << std::endl;

  } // sink construction

  { // fiss construction

  std::vector<DenseMatrix> fiss(pmesh->attributes.Max());
  std::vector<Tensor<mfem::Vector>> fiss_saaf(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    fiss[i].SetSize(block_rank);
    fiss_saaf[i].reserve(block_rank*block_rank);
    const Material_record& mat{mat_lib.mat[i]};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {

        double val{0.0};
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const Material_isotope& isotope{mat.iso[iso_idx]};
          const XSRecord& record{isotope.rec};
          const double nden{isotope.nden};
          val += record.chi[g] * nden * record.nusf[gprime];
        } // iso_idx < mat.niso

        for (int n{0}; n < quad->nd(); ++n)
        {
          const int row{n + g*quad->nd()};
          for (int nprime{0}; nprime < quad->nd(); ++nprime)
          {
            const int col{nprime + gprime*quad->nd()};
            fiss[i](row,col) = val*quad->weight(nprime);

            Vector v{quad->omega(n)};
            v /= mat.rec.sigma_t[g];
            v *= val * quad->weight(nprime);
            fiss_saaf[i].AddElement(v, row, col);
          } // nprime < quad->nd()
        } // n < quad->nd()

      } // gprime < mat_lib.ngroup()
    } // g < mat_lib.ngroup()
  } // i < pmesh->attributes.Max()
  MatrixPWConstCoefficient fiss_func{fiss};
  VectorTensorPWConstCoefficient fiss_saaf_func{block_rank, fiss_saaf};

  ParBilinearForm fiss_bilinear{fespace};
  fiss_bilinear.AddDomainIntegrator(new VectorMassIntegrator{fiss_func});
  fiss_bilinear.AddDomainIntegrator(new VectorConvectionIntegrator{fiss_saaf_func});

  std::cout << "FISSION_ASSEMBLY_BEGIN id=" << myid << std::endl;
  fiss_bilinear.Assemble(skip_zeros);
  fiss_bilinear.Finalize(skip_zeros);
  std::cout << "FISSION_ASSEMBLY_END id=" << myid << std::endl;

  if (myid == 0) std::cout << "FISS_TRANSLATION_BEGIN" << std::endl;
  Mhypre = fiss_bilinear.ParallelAssemble();
  if (myid == 0) std::cout << "FISS_TRANSLATION_END" << std::endl;

  } // fiss construction

  constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};

  PetscParMatrix A{Ahypre, petsc_format};
  PetscParMatrix M{Mhypre, petsc_format};

  PetscErrorCode ierr;
  ierr = MatSetBlockSize(A, block_rank); PCHKERRQ(A, ierr);
  ierr = MatSetBlockSize(M, block_rank); PCHKERRQ(M, ierr);

  if ((petsc_format != Operator::PETSC_MATHYPRE) && 
      (petsc_format != Operator::PETSC_MATSHELL))
  {
    delete Ahypre;
    delete Mhypre;
  }

  time.stop();

  time.start("slepc setup");
  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(inp.tol.phi * inp.tol.k);
  slepc.SetMaxIter(inp.tol.max_outer);
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
  slepc.SetOperators(A, M);
  time.stop("slepc setup");

  // TODO consider restarting

  if (myid == 0)
  {
    std::cout << "SOLVE_BEGIN" << std::endl;
    mem.update_and_print();
    mem.init_background(1000);
  }
  time.start(time.solver);
  slepc.Solve();
  time.stop(time.solver);
  if (myid == 0) 
  {
    std::cout << "SOLVE_END" << std::endl;
    mem.stop_background();
  }

  if ((petsc_format == Operator::PETSC_MATHYPRE) || 
      (petsc_format == Operator::PETSC_MATSHELL))
  {
    delete Ahypre;
    delete Mhypre;
  }

  // Note: if SLEPc fails to *fully* converge, the eigenvalues or eigenvectors
  // cannot be accessed, even if they are *nearly* converged. Therefore, to
  // avoid dereferencing bad memory, we must abort.
  if (slepc.GetNumConverged() < 1)
    throw Failed_to_converge{"SLEPc failed to converge.\n"};

  slepc.GetEigenvalue(0, angular_flux.keff);
  angular_flux.keff = 1.0 / angular_flux.keff;
  Vector tmp(fespace->GetTrueVSize());
  slepc.GetEigenvector(0, tmp);

  // each group/moment is currently stored as a vector quantity
  // parsing this is then handled by ParaView
  // for dimensions <= 3, the vector is listed {x,y,z}
  // for higher dimensions, the vector is listed {0,1,2,3,4,...}
  angular_flux(0).Distribute(tmp);

  // flip the sign of the solution such that the scalar flux in group=0 is positive
  tmp.Destroy();
  angular_flux(0).GetNodalValues(tmp);
  if (std::abs(tmp.Max()) < std::abs(tmp.Min()))
    angular_flux(0).Neg();

  return;
}

void saaf_true_amgx_vector(const Quadrature* const quad,
                           Timer &time,
                           const Input &inp,
                           const Material_library &mat_lib,
                           Result &angular_flux)
{
  ParFiniteElementSpace* const fespace{angular_flux(0).ParFESpace()};
  ParMesh* const pmesh{fespace->GetParMesh()};

  const int myid{fespace->GetMyRank()};
  const int block_rank{mat_lib.ngroup()*quad->nd()}; // aka vdim
  constexpr int skip_zeros{2};

  HypreParMatrix *Ahypre{nullptr}, *Mhypre{nullptr};

  time.start(time.matrix_assembly);

  { // sink construction

  std::vector<DenseMatrix> omega_operator(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    omega_operator[n].SetSize(quad->dim());
    MultVVt(quad->omega(n), omega_operator[n]);
  }

  ParBilinearForm sink_bilinear{fespace};

  std::vector<Tensor<mfem::DenseMatrix>> leakage(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    const XSRecord& rec{mat_lib.mat[i].rec};
    leakage[i].reserve(block_rank);
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        const int dia{n + g*quad->nd()};
        omega_operator[n] *= 1.0/rec.sigma_t[g];
        leakage[i].AddElement(omega_operator[n], dia);
        omega_operator[n] *= rec.sigma_t[g];
      }
    }
  }
  MatrixTensorPWConstCoefficient leakage_func{block_rank, leakage};
  sink_bilinear.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});

  Tensor<mfem::Vector> boundary;
  boundary.reserve(block_rank);
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      boundary.AddElement(quad->omega(n), n+g*quad->nd());
  VectorTensorConstantCoefficient boundary_func{block_rank, boundary};
  sink_bilinear.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{boundary_func, 1.0});

  std::vector<DenseMatrix> removal(pmesh->attributes.Max());
  std::vector<Tensor<mfem::Vector>> removal_saaf(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    removal[i].SetSize(block_rank);
    removal_saaf[i].reserve(block_rank*block_rank);
    const XSRecord& record{mat_lib.mat[i].rec};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      Array<int> col_idx;
      Vector scat_dat;
      record.sigma_s.GetRow(g, col_idx, scat_dat);

      for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
      {
        const int gprime{col_idx[gidx]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int row{n + g*quad->nd()};
          for (int nprime{0}; nprime < quad->nd(); ++nprime)
          {
            const int col{nprime + gprime*quad->nd()};
            const double val{-scat_dat(gidx)*quad->weight(nprime)};
            removal[i](row,col) = val;

            Vector v{quad->omega(n)};
            v /= record.sigma_t[g];
            v *= val;
            removal_saaf[i].AddElement(v, row, col);
          } // nprime < quad->nd()
        } // n < quad->nd()
      } // gidx < col_idx.Size()

      for (int n{0}; n < quad->nd(); ++n)
      {
        const int dia{n + g*quad->nd()};
        removal[i](dia,dia) += record.sigma_t[g];
      } // n < quad->nd()

    } // g < mat_lib.ngroup()
  } // i < pmesh->attributes.Max()
  MatrixPWConstCoefficient removal_func{removal};
  VectorTensorPWConstCoefficient removal_saaf_func{block_rank, removal_saaf};

  sink_bilinear.AddDomainIntegrator(new VectorMassIntegrator{removal_func});
  sink_bilinear.AddDomainIntegrator(new VectorConvectionIntegrator{removal_saaf_func});

  std::cout << "SINK_ASSENBLE_BEGIN id=" << myid << std::endl;
  time.start("sink_assemble");
  sink_bilinear.Assemble(skip_zeros);
  sink_bilinear.Finalize(skip_zeros); // no essential boundary conditions (maybe)
  time.stop("sink_assemble");
  std::cout << "SINK_ASSEMBLE_END id=" << myid << std::endl;

  if (myid == 0) std::cout << "SINK_TRANSLATION_BEGIN" << std::endl;
  time.start("sink_par_assemble");
  Ahypre = sink_bilinear.ParallelAssemble();
  time.stop("sink_par_assemble");
  if (myid == 0) std::cout << "SINK_TRANSLATION_END" << std::endl;

  }

  { // fiss construction

  std::vector<DenseMatrix> fiss(pmesh->attributes.Max());
  std::vector<Tensor<mfem::Vector>> fiss_saaf(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    fiss[i].SetSize(block_rank);
    fiss_saaf[i].reserve(block_rank*block_rank);
    const Material_record& mat{mat_lib.mat[i]};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {

        double val{0.0};
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const Material_isotope& isotope{mat.iso[iso_idx]};
          const XSRecord& record{isotope.rec};
          const double nden{isotope.nden};
          val += record.chi[g] * nden * record.nusf[gprime];
        } // iso_idx < mat.niso

        for (int n{0}; n < quad->nd(); ++n)
        {
          const int row{n + g*quad->nd()};
          for (int nprime{0}; nprime < quad->nd(); ++nprime)
          {
            const int col{nprime + gprime*quad->nd()};
            fiss[i](row,col) = val*quad->weight(nprime);

            Vector v{quad->omega(n)};
            v /= mat.rec.sigma_t[g];
            v *= val * quad->weight(nprime);
            fiss_saaf[i].AddElement(v, row, col);
          } // nprime < quad->nd()
        } // n < quad->nd()

      } // gprime < mat_lib.ngroup()
    } // g < mat_lib.ngroup()
  } // i < pmesh->attributes.Max()
  MatrixPWConstCoefficient fiss_func{fiss};
  VectorTensorPWConstCoefficient fiss_saaf_func{block_rank, fiss_saaf};

  ParBilinearForm fiss_bilinear{fespace};
  fiss_bilinear.AddDomainIntegrator(new VectorMassIntegrator{fiss_func});
  fiss_bilinear.AddDomainIntegrator(new VectorConvectionIntegrator{fiss_saaf_func});

  std::cout << "FISSION_ASSEMBLY_BEGIN id=" << myid << std::endl;
  time.start("fiss_assemble");
  fiss_bilinear.Assemble(skip_zeros);
  fiss_bilinear.Finalize(skip_zeros);
  time.stop("fiss_assemble");
  std::cout << "FISSION_ASSEMBLY_END id=" << myid << std::endl;

  if (myid == 0) std::cout << "FISS_TRANSLATION_BEGIN" << std::endl;
  time.start("fiss_par_assemble");
  Mhypre = fiss_bilinear.ParallelAssemble();
  time.stop("fiss_par_assemble");
  if (myid == 0) std::cout << "FISS_TRANSLATION_END" << std::endl;

  }

  constexpr bool compute_residual{false};

  time.start("amgx_setup");
  AmgXSolver Aamgx;
  Aamgx.DefaultParameters(AmgXSolver::SOLVER, inp.verbose);
  Aamgx.ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);
  Aamgx.InitExclusiveGPU(fespace->GetComm());
  Aamgx.SetOperator(*Ahypre);
  if (!compute_residual) 
  {
    delete Ahypre;
    Ahypre = nullptr;
  }
  time.stop("amgx_setup");

  HypreParVector x{*Mhypre};
  x.UseDevice(true);
  x.Randomize(730); // if default seed is used, the seed is actually the time and we lose reproducability

  double k_conv{inp.tol.k*1e2+1.0};
  double x_conv{inp.tol.phi*1e2+1.0};

  double k{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer) && 
      ((k_conv > inp.tol.k) || (x_conv > inp.tol.phi)); ++iter)
  {

    time.start("old_copy");
    const double k_old{k};
    HypreParVector x_old{x}; // constructor doesn't perform deep copy
    x_old.UseDevice(true);
    x_old = x;
    time.stop("old_copy");

    time.start("calc_v");
    // v = x / \| x \|
    HypreParVector v{x}; // constructor doesn't perform deep copy
    v.UseDevice(true);
    v = x;
    v /= par_norml2(x);
    time.stop("calc_v");

    time.start("matvec");
    // b = M*v
    HypreParVector b{v}; // constructor doesn't perform deep copy
    b.UseDevice(true);
    //Mhypre->Mult(v, b); // this is performed on CPU
    Mhypre->Mult(v, b, 1.0, 0.0); // make sure we use the proper function signature
    time.stop("matvec");
    
    time.start("amgx_solve");
    // x = A^-1 * b
    Aamgx.Mult(b, x);
    time.stop("amgx_solve");

    time.start("dot_product");
    // k = v^T * x
    const double local_k{v*x};
    MPI_Allreduce(&local_k, &k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time.stop("dot_product");

    // convergence criteria
    k_conv = std::abs(k - k_old);
    x_old -= x;
    x_conv = par_norml2(x_old);

    if (compute_residual)
    {
      time.start("residual");
      // r = A*x - M*x/k
      HypreParVector r0{x};
      r0.UseDevice(true);
      Ahypre->Mult(x, r0);

      HypreParVector r1{x};
      r1.UseDevice(true);
      Mhypre->Mult(x, r1);
      r1 /= k;

      r0 -= r1;
      std::cout << "residual=" << par_norml2(r0) << std::endl;
      time.stop("residual");
    }

    if (myid == 0) std::cout << "iter=" << iter
                             << " inner=" << Aamgx.GetNumIterations()
                             << " k=" << k 
                             << " dk=" << k_conv 
                             << " dx=" << x_conv 
                             << std::endl;
  }

  if (myid == 0) std::cout << "ITERATIONS_FINISHED!!!" << std::endl;

  delete Ahypre;
  delete Mhypre;

  angular_flux.keff = k;
  time.start("distribute");
  angular_flux(0).Distribute(x);
  time.stop("distribute");

  return;
}

void saaf_angle_power_iteration(const Quadrature* const quad,
                                Timer& time,
                                const Input& inp,
                                const Material_library& mat_lib,
                                ParMesh* const pmesh,
                                ParFiniteElementSpace* const fespace,
                                Result& angular_flux,
                                Result& scalar_flux)
{

  const int myid{fespace->GetMyRank()};

  time.start(time.matrix_assembly);

  std::vector<DenseMatrix> omega_mat(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    omega_mat[n].SetSize(quad->dim());
    MultVVt(quad->omega(n), omega_mat[n]);
  }

  Tensor<Vector> omega;
  omega.reserve(quad->nd());
  for(int n{0}; n < quad->nd(); ++n)
    omega.AddElement(quad->omega(n), n);
  VectorTensorConstantCoefficient omega_func{quad->nd(), omega};
  
  std::vector<std::unique_ptr<ParBilinearForm>> a(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    a[g] = std::make_unique<ParBilinearForm>(fespace);

    std::vector<Vector> total(pmesh->attributes.Max());
    std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      const double tot{mat_lib.mat[i].rec.sigma_t[g]};

      leakage[i].reserve(quad->nd());
      for (int n{0}; n < quad->nd(); ++n)
      {
        omega_mat[n] *= 1.0/tot;
        leakage[i].AddElement(omega_mat[n], n);
        omega_mat[n] *= tot;
      }
      
      total[i].SetSize(quad->nd());
      total[i] = tot;

    }
    VectorPWConstCoefficient total_func{total};
    MatrixTensorPWConstCoefficient leakage_func{quad->nd(), leakage};
    a[g]->AddDomainIntegrator(new VectorMassIntegrator{total_func});
    a[g]->AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
    a[g]->AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});

    a[g]->Assemble(); // note: no essential bcs in saaf
    a[g]->Finalize();
  }

  time.stop(time.matrix_assembly);

  std::vector<GridFunctionCoefficient> scalar_flux_coeff;
  scalar_flux_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux_coeff.emplace_back(&scalar_flux(g));

  std::vector<PWConstCoefficient> self_scatter_xs;
  self_scatter_xs.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector self_scatter(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      self_scatter[i] = mat_lib.mat[i].rec.sigma_s.Elem(g,g);
    self_scatter_xs.emplace_back(self_scatter);
  }

  std::vector<ProductCoefficient> self_scatter_phi;
  self_scatter_phi.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    self_scatter_phi.emplace_back(scalar_flux_coeff[g], self_scatter_xs[g]);

  Vector ones(quad->nd());
  ones = 1.0;
  VectorConstantCoefficient ones_func{ones};

  std::vector<ScalarVectorProductCoefficient> self_scatter_func;
  self_scatter_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    self_scatter_func.emplace_back(self_scatter_phi[g], ones_func);

  Array<int> ess_tdof_list(0); // no essential bcs
  Array<int> reflective_condition(pmesh->bdr_attributes.Max());
  reflective_condition = 0;
  for (int i{0}; (i < inp.boundary_conditions.Size()) && (i < reflective_condition.Size()); ++i)
  {
    if (inp.boundary_conditions[i] == 0) // zero represents reflective
      reflective_condition[i] = 1;
  }

  std::vector<std::vector<Tensor<Vector>>> saaf;
  saaf.resize(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    saaf[g].resize(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      saaf[g][i].reserve(quad->nd());
  }

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      const double tot{mat_lib.mat[i].rec.sigma_t[g]};
      for (int n{0}; n < quad->nd(); ++n)
      {
        Vector tmp{quad->omega(n)};
        tmp *= 1.0/tot;
        saaf[g][i].AddElement(tmp, n);
      }
    }
  }

  std::vector<VectorTensorPWConstCoefficient> saaf_func;
  saaf_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    saaf_func.emplace_back(quad->nd(), saaf[g]);

  const int integration_order{fespace->GetOrder(0)+1};
  QuadratureSpace qspace{const_cast<ParMesh*>(pmesh), integration_order};

  std::vector<QuadratureFunction> fiss_quad_func;
  std::vector<QuadratureFunction> upscatter_quad_func, dnscatter_quad_func;
  fiss_quad_func.reserve(mat_lib.ngroup());
  upscatter_quad_func.reserve(mat_lib.ngroup());
  dnscatter_quad_func.reserve(mat_lib.ngroup());

  std::vector<QuadratureFunctionCoefficient> fiss_quad_coeff;
  std::vector<QuadratureFunctionCoefficient> upscatter_quad_coeff, dnscatter_quad_coeff;
  fiss_quad_coeff.reserve(mat_lib.ngroup());
  upscatter_quad_coeff.reserve(mat_lib.ngroup());
  dnscatter_quad_coeff.reserve(mat_lib.ngroup());

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_quad_func.emplace_back(&qspace);
    upscatter_quad_func.emplace_back(&qspace);
    dnscatter_quad_func.emplace_back(&qspace);
    fiss_quad_coeff.emplace_back(fiss_quad_func.back());
    upscatter_quad_coeff.emplace_back(upscatter_quad_func.back());
    dnscatter_quad_coeff.emplace_back(dnscatter_quad_func.back());
  }

  std::vector<SumCoefficient> scat_quad_coeff;
  scat_quad_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scat_quad_coeff.emplace_back(upscatter_quad_coeff[g], dnscatter_quad_coeff[g]);

  std::vector<ScalarVectorProductCoefficient> scat_quad_func_func;
  scat_quad_func_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scat_quad_func_func.emplace_back(scat_quad_coeff[g], ones_func);

  std::vector<ScalarVectorProductCoefficient> fiss_quad_func_func;
  fiss_quad_func_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    fiss_quad_func_func.emplace_back(fiss_quad_coeff[g], ones_func);

  std::vector<ScalarVectorTensorProductCoefficient> scat_quad_saaf_coeff;
  scat_quad_saaf_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scat_quad_saaf_coeff.emplace_back(scat_quad_coeff[g], saaf_func[g]);

  std::vector<ScalarVectorTensorProductCoefficient> fiss_quad_saaf_coeff;
  fiss_quad_saaf_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    fiss_quad_saaf_coeff.emplace_back(fiss_quad_coeff[g], saaf_func[g]);

  // The sources will be "nd" copies of the regular sources because (for now)
  // everything is isotropic.
  // The saaf sources have n & g dependency so they will be different.
  std::vector<std::unique_ptr<ParLinearForm>> b(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    b[g] = std::make_unique<ParLinearForm>(fespace);

    b[g]->AddDomainIntegrator(new VectorDomainLFIntegrator{scat_quad_func_func[g]});
    b[g]->AddDomainIntegrator(new VectorDomainLFGradIntegrator{scat_quad_saaf_coeff[g]});
    b[g]->AddDomainIntegrator(new VectorDomainLFIntegrator{self_scatter_func[g]});

    b[g]->AddDomainIntegrator(new VectorDomainLFIntegrator{fiss_quad_func_func[g]});
    b[g]->AddDomainIntegrator(new VectorDomainLFGradIntegrator{fiss_quad_saaf_coeff[g]});

    // NOTE: when solving all angles simultaneously, reflective bcs will be
    // essential. we will need to eliminate from lhs (or do something...) and
    // not simply treat rhs.

  }

////////////////////////////////////////////////////////////////////////////////
  std::vector<std::unique_ptr<ParLinearForm>> fiss_src_form(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_src_form[g] = std::make_unique<ParLinearForm>(
      static_cast<ParFiniteElementSpace*>(scalar_flux(g).FESpace())
      );

    Vector nusf(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      nusf(i) = mat_lib.mat[i].rec.nusf[g];
    PWConstCoefficient nusf_func{nusf};

    fiss_src_form[g]->AddDomainIntegrator(new DomainLFIntegrator{nusf_func});
    fiss_src_form[g]->Assemble();
  }
////////////////////////////////////////////////////////////////////////////////

  // setup preconditioner
  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  // setup solver
  CGSolver sol(MPI_COMM_WORLD);
  sol.SetRelTol(inp.tol.inner_rel);
  sol.SetAbsTol(inp.tol.inner_abs);
  sol.SetMaxIter(inp.tol.max_inner);
  sol.SetPrintLevel(inp.mfem_print_level());
  sol.SetPreconditioner(amg);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    angular_flux(g) = 1.0;
    scalar_flux(g) = 1.0;
  }

  double k_conv{inp.tol.k * 1e1 + 1e-6};
  double phi_conv{inp.tol.phi * 1e1 + 1e-6};
  double keff_old{angular_flux.keff};
  double fiss_sum_old{1.0};

  QuadratureFunction fiss_src_old{&qspace};
  fiss_src_old = 1.0;

  time.start(time.solver);
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    time.start("fiss+upscatter");
    calc_fiss_upscatter(scalar_flux, mat_lib, fiss_quad_func, upscatter_quad_func, inp.flux_free_chi);
    for (QuadratureFunction& qf : fiss_quad_func)
      qf /= angular_flux.keff;
    time.stop("fiss+upscatter");

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      double scattering_conv{inp.tol.inner_rel * 1e1 + 1e-6};

      time.start("downscatter");
      calc_downscatter(g, scalar_flux, mat_lib, dnscatter_quad_func[g]);
      time.stop("downscatter");

      for (int scattering_iter{0}; (scattering_iter < inp.tol.max_inner) &&
        (scattering_conv > inp.tol.inner_rel); ++scattering_iter)
      {
        time.start("bassemble");
        b[g]->Assemble();
        time.stop("bassemble");

        OperatorPtr A;
        Vector B, X;
        a[g]->FormLinearSystem(ess_tdof_list, angular_flux(g), *b[g], A, X, B);
        sol.SetOperator(*A);
        time.start("mult");
        sol.Mult(B, X);
        time.stop("mult");
        a[g]->RecoverFEMSolution(X, *b[g], angular_flux(g));

        ParGridFunction phi_old{scalar_flux(g)};
        time.start("scalarflux");
        calc_scalar_flux_vector(g, angular_flux, quad, scalar_flux);
        time.stop("scalarflux");
        phi_old -= scalar_flux(g);


        phi_old.HostRead();
        const double local_scattering_conv{
          phi_old.Normlinf() / scalar_flux(g).Normlinf()};
        MPI_Allreduce(&local_scattering_conv, &scattering_conv, 1, MPI_DOUBLE, 
          MPI_MAX, MPI_COMM_WORLD);
        if (myid == 0)
          std::cout << "..."
                    << scattering_iter
                    << " g=" << g
                    << " conv=" << scattering_conv
                    << '\n';
      } // scattering_iter
    } // ngroup

    time.start("fisssum");
    double fiss_sum{0.0}, local_fiss_sum{0.0};
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      local_fiss_sum += (*fiss_src_form[g])(scalar_flux(g)); // LinearForm::operator()
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time.stop("fisssum");

    time.start("phiconv");
    QuadratureFunction fiss_src{&qspace};
    calc_fiss_src(scalar_flux, mat_lib, fiss_src);
    fiss_src_old -= fiss_src;
    double local_phi_conv{fiss_src_old.Normlinf() / fiss_src.Normlinf()};
    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    time.stop("phiconv");

    if (iter > 0) angular_flux.keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(angular_flux.keff - keff_old);

    if (myid == 0)
      std::cout << iter 
        << " keff=" << angular_flux.keff
        << " dk=" << k_conv 
        << " dphi=" << phi_conv 
        << std::endl;

    scalar_flux.keff = angular_flux.keff;
    // store old values
    keff_old = angular_flux.keff;
    fiss_sum_old = fiss_sum;
    fiss_src_old = fiss_src;
  }
  time.stop(time.solver);

  return;
}


void slepc_davidson_hypre_boomeramg_initialize(const Input& inp, int num_equations)
{

  const std::string slepcrc_fname{inp.concatenate_path("slepcrc")};

  std::vector<std::string> arguments{" ",
    "-eps_monitor", "-eps_converged_reason", "-eps_type", "gd", 
    "-st_type", "precond", "-st_ksp_type", "preonly",
    "-st_pc_type", "hypre", "-st_pc_hypre_type", "boomeramg",
    // the following are required for gpu
    "-st_pc_hypre_boomeramg_coarsen_type", "PMIS",
    "-st_pc_hypre_boomeramg_interp_type", "ext+i",
    "-st_pc_hypre_boomeramg_relax_type_all", "l1scaled-Jacobi",
    // try to reduce the memory burden (size of search space after restarting)
    "-eps_gd_minv", "2"};

  arguments.emplace_back("-st_pc_hypre_boomeramg_numfunctions");
  arguments.emplace_back(std::to_string(num_equations));

  if (inp.verbose)
  {
    arguments.emplace_back("-eps_view");
    arguments.emplace_back("-st_pc_hypre_boomeramg_print_statistics");
  }

  int argc_val{static_cast<int>(arguments.size())};
  std::vector<char*> argv_val(argc_val);
  for (int i{0}; i < argc_val; ++i)
    argv_val[i] = const_cast<char*>(arguments[i].c_str());

  char** argv{argv_val.data()};

  /*
  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(&argc_val, &argv, slepcrc_fname.c_str(), nullptr);
  else
    MFEMInitializeSlepc(&argc_val, &argv, nullptr, nullptr);
  */

  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(nullptr, nullptr, slepcrc_fname.c_str(), nullptr);
  else
    MFEMInitializeSlepc(nullptr, nullptr, nullptr, nullptr);

  return;
}

void calc_scalar_flux(int g,
                      const Result& angular_flux, 
                      const Quadrature* const quad,
                      Result& scalar_flux)
{
  scalar_flux(g) = 0.0;
  for (int n{0}; n < quad->nd(); ++n)
  {
    ParGridFunction tmp{angular_flux(g,n)};
    tmp *= quad->weight(n);
    scalar_flux(g) += tmp;
  }
  return;
}

void calc_scalar_flux_vector(int g,
                             const Result& angular_flux,
                             const Quadrature* const quad, 
                             Result& scalar_flux)
{
  if (scalar_flux.FESpace()->GetOrder(0) != 1)
  {
    std::cout << "requires linear finite elements" << std::endl;;
    throw "error";
  }
  scalar_flux(g) = 0.0;
  for (int n{0}; n < quad->nd(); ++n)
  {
    Vector tmp;
    angular_flux(g).GetNodalValues(tmp, n+1);
    tmp *= quad->weight(n);
    scalar_flux(g) += tmp;
  }
  return;
}

void calc_fiss_upscatter(const Result& scalar_flux, 
                         const Material_library& mat_lib, 
                         std::vector<QuadratureFunction>& fiss_quad_func, 
                         std::vector<QuadratureFunction>& upscatter_quad_func,
                         const bool flux_free_chi)
{
  const QuadratureSpace* const qspace{fiss_quad_func[0].GetSpace()};
  const Mesh* const mesh{qspace->GetMesh()};
  const FiniteElementSpace* const fespace{scalar_flux(0,0).FESpace()};

  for (QuadratureFunction& qf : fiss_quad_func)
    qf = 0.0;

  for (QuadratureFunction& qf : upscatter_quad_func)
    qf = 0.0;

  int idx{0};
  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};

    const FiniteElement* const fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {
      const IntegrationPoint ip{ir.IntPoint(i)};
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      std::vector<double> spectrum(mat_lib.ngroup());

      // fission
      if (rec.is_fiss)
      {
        const std::vector<double>& nusf{rec.nusf};
        double fiss_sum{0.0};
        for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
        {
          Vector locvec(dofs.Size());
          scalar_flux(gprime,0).GetSubVector(dofs, locvec);
          spectrum[gprime] = dofval*locvec;
          fiss_sum += spectrum[gprime] * nusf[gprime];
        }

        const std::vector<double> chi{
          (mat_lib.rx_wide_chi) ? mat_lib.chi : 
          ((flux_free_chi) ? rec.chi : mat.calculate_chi(spectrum))};

        for (int g{0}; g < mat_lib.ngroup(); ++g)
          fiss_quad_func[g](idx) += chi[g] * fiss_sum;
      }
      else
      {
        for (double& x : spectrum)
          x = -1.0;
      }

      // upscatter
      const SparseMatrix& scat{rec.sigma_s};
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        Array<int> col_idx;
        Vector scat_data;
        scat.GetRow(g, col_idx, scat_data);

        for (int j{0}; j < col_idx.Size(); ++j)
        {
          const int gprime{col_idx[j]};
          if (gprime > g)
          {
            if (spectrum[gprime] < 0.0) // for VVER, 95% cache hits
            {
              Vector locvec(dofs.Size());
              scalar_flux(gprime,0).GetSubVector(dofs, locvec);
              spectrum[gprime] = dofval*locvec;
            }
            upscatter_quad_func[g](idx) += spectrum[gprime]*scat_data(j);
          }
        }
      }

      ++idx;
    }
  }

  return;
}

void calc_downscatter(const int g, 
                      const Result& scalar_flux,
                      const Material_library& mat_lib,
                      QuadratureFunction& dnscatter_quad_func)
{
  dnscatter_quad_func = 0.0;

  const QuadratureSpace* const qspace{dnscatter_quad_func.GetSpace()};
  const Mesh* const mesh{qspace->GetMesh()};
  const FiniteElementSpace* const fespace{scalar_flux(0,0).FESpace()};

  int idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const SparseMatrix& scat{rec.sigma_s};

    Array<int> col_idx;
    Vector scat_data;
    scat.GetRow(g, col_idx, scat_data);

    const FiniteElement* const fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {
      const IntegrationPoint ip{ir.IntPoint(i)};
      // get the weights/factors for this integration point within the element here
      // (assume that at least one energy group will be downscattering)
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);
      for (int j{0}; j < col_idx.Size(); ++j)
      {
        const int gprime{col_idx[j]};
        if (gprime < g)
        {
          // get the grid function values at the DOF here
          // take the inner product
          // const double val{inner*prod};
          Vector locvec(dofs.Size());
          scalar_flux(gprime,0).GetSubVector(dofs, locvec);
          const double flux{dofval*locvec};
          dnscatter_quad_func(idx) += flux*scat_data(j);
        }
      }
      ++idx;
    }

  }
  return;
}

void calc_spn_next(const Result& flux,
                   const Material_library& mat_lib,
                   SPNWrapper<QuadratureFunction>& spn_next_quad_func)
{

  const QuadratureSpace* const qspace{spn_next_quad_func(0,0).GetSpace()};
  const Mesh* const mesh{qspace->GetMesh()};
  const FiniteElementSpace* const fespace{flux(0,0).FESpace()};
  const int dim{mesh->Dimension()};

  for (int n{0}; n < spn_next_quad_func.nmoment(); ++n)
  {
    for (int g{0}; g < spn_next_quad_func.ngroup(); ++g)
    {
      spn_next_quad_func(g,n) = 0.0;
    }
  }

  int idx{0}, idx_sto{0};
  for (int e{0}; e < qspace->GetNE(); ++e)
  {
    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const DenseMatrix& sigma_r{rec.sigma_r};

    ElementTransformation* const eltr{fespace->GetElementTransformation(e)};

    for (int n{0}; n < spn_next_quad_func.nmoment()-1; ++n)
    {
      const double xmult_next{(2.0*n+1.0)*(2.0*n+2.0) / ((4.0*n+1.0)*(4.0*n+3.0))};
      for (int g{0}; g < spn_next_quad_func.ngroup(); ++g)
      {
        const double xs_val{sigma_r(g,2*n+1)};
        Vector v(dim);
        flux(g,n+1).GetGradient(*eltr, v);
        v *= -xmult_next / xs_val;

        idx = idx_sto;
        for (int i{0}; i < ir.GetNPoints(); ++i)
        {
          for (int d{0}; d < dim; ++d)
          {
            spn_next_quad_func(g,n)(idx) += v(d);
            ++idx;
          }
        }

      }
    }

    idx_sto = idx;

  }

  return;
}

void calc_spn_prev(const int g, const int n,
                   const Result& flux,
                   const Material_library& mat_lib,
                   QuadratureFunction& spn_prev_quad_func)
{
  spn_prev_quad_func = 0.0;
  if (n == 0) return;

  const QuadratureSpace* const qspace{spn_prev_quad_func.GetSpace()};
  const Mesh* const mesh{qspace->GetMesh()};
  const FiniteElementSpace* const fespace{flux(0,0).FESpace()};
  const double xmult_prev{(4.0*n*n - 2.0*n) / (16.0*n*n - 1.0)};
  const int dim{mesh->Dimension()};

  int idx{0};
  for (int e{0}; e < qspace->GetNE(); ++e)
  {
    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const DenseMatrix& sigma_r{rec.sigma_r};
    const double xs_val{sigma_r(g,2*n-1)};

    ElementTransformation* const eltr{fespace->GetElementTransformation(e)};
    Vector v(dim);
    flux(g,n-1).GetGradient(*eltr, v);
    v *= -xmult_prev / xs_val;

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {
      for (int d{0}; d < dim; ++d)
      {
        spn_prev_quad_func(idx) += v(d);
        ++idx;
      }
    }

  }

  return;
}

void calc_fiss_src(const Result& scalar_flux, 
                      const Material_library& mat_lib,
                      QuadratureFunction& fiss_src)
{

  const QuadratureSpace* const qspace{fiss_src.GetSpace()};
  const Mesh* const mesh{qspace->GetMesh()};
  const FiniteElementSpace* const fespace{scalar_flux(0,0).FESpace()};

  fiss_src = 0.0;

  int idx{0};
  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};

    if (!rec.is_fiss) continue;

    const std::vector<double>& nusf{rec.nusf};

    const FiniteElement* const fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint ip{ir.IntPoint(i)};
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        Vector locvec(dofs.Size());
        scalar_flux(gprime,0).GetSubVector(dofs, locvec);
        const double flux{dofval*locvec};
        fiss_src(idx) += nusf[gprime]*flux;
      }

      ++idx;
    }
  }

  return;
}

void calc_scalar_flux_slepc_vector(const Result& angular_flux, 
                                   const Quadrature* const quad,
                                   Result& scalar_flux)
{

  scalar_flux.keff = angular_flux.keff;

  const int ngroup{angular_flux(0).ParFESpace()->GetVDim() / quad->nd()};
  ParGridFunction tmp{scalar_flux(0)};
  for (int g{0}; g < ngroup; ++g)
  {
    scalar_flux(g) = 0.0;
    for (int n{0}; n < quad->nd(); ++n)
    {
      const int idx{n + g*quad->nd()};
      GridFunctionCoefficient gfc{&angular_flux(0), idx+1};
      tmp.ProjectCoefficient(gfc);
      tmp *= quad->weight(n);
      scalar_flux(g) += tmp;
    }
  }

  return;
}

void calc_power(const Material_library& mat_lib,
                const Result& scalar_flux,
                double rxpwr,
                ParGridFunction& power)
{

  const int max_att{scalar_flux(0).ParFESpace()->GetParMesh()->attributes.Max()};

  // accumulate power from each energy group
  ParGridFunction tmp{scalar_flux(0)};
  power = 0.0;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Vector kappasf(max_att);
    for (int i{0}; i < max_att; ++i)
      kappasf(i) = mat_lib.mat[i].rec.nusf[g]; // TODO replace with proper value
    PWConstCoefficient kappasf_coeff{kappasf};

    GridFunctionCoefficient gfc{&scalar_flux(g)};
    ProductCoefficient pc{gfc, kappasf_coeff};
    tmp.ProjectCoefficient(pc);
    power += tmp;
  }

  // calculate power normalization factor
  ConstantCoefficient one{1.0};
  ParLinearForm b{scalar_flux(0).ParFESpace()};
  b.AddDomainIntegrator(new DomainLFIntegrator{one});
  b.Assemble();
  const double factor{rxpwr / b(power)}; // use the inner product in the space

  // renormalize power
  power *= factor; // power has units [W/cc]

  return;
}

double par_norml2(const Vector &x)
{
  const double local_norm{std::pow(x.Norml2(),2)};
  double norm{0.0};
  MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(norm);
}
