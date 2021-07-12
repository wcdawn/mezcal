#include "solver.hpp"

#include "fission_coefficient.hpp"
#include "scattering_coefficient.hpp"
#include "spn_source_coefficient.hpp"
#include "make_unique.hpp"
#include "vector_pwconst_coefficient.hpp"
#include "matrix_pwconst_coefficient.hpp"
#include "incoming_flux_coefficient.hpp"
#include "stability_tau_coefficient.hpp"
#include "vector_product_coefficient.hpp"
#include "nda_boundary_factor_coefficient.hpp"
#include "nda_diffusion_coefficient.hpp"

#include "quadrature.hpp"
#include "level_symmetric_quadrature.hpp"
#include "legendre_chebyschev_quadrature.hpp"

#include "conduction.hpp"

#if 0
#include "upscattering_coefficient.hpp"
#include "downscattering_coefficient.hpp"
#endif

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

  if (fespace->GetMyRank() == 0)
  {
    std::cout << "attributes" << std::endl;
    pmesh->attributes.Print();
    std::cout << "bdr_attributes" << std::endl;
    pmesh->bdr_attributes.Print();
  }

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
  HyprePCG cg(MPI_COMM_WORLD);
  cg.SetTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetPrintLevel(inp.hypre_print_level());
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
        amg.SetOperator(*A);
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

#ifdef AMGX_AVAILABLE

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

#else

  MFEM_ABORT("solver not supported without amgx");

#endif

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
  amg.SetOperator(*A);
  amg.SetPrintLevel(inp.hypre_print_level());

  HyprePCG cg(MPI_COMM_WORLD);
  cg.SetTol(inp.tol.inner_rel);
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

  // get the process id so that only the root process writes to terminal
  const int myid{fespace->GetMyRank()};

  if (myid == 0) std::cout << "__spn_order__ = " << inp.spn_order() << std::endl;

  mat_lib.set_nmoment(inp.spn_order()+1);

  time.start(time.matrix_assembly);

  std::unique_ptr<HypreParMatrix> Ahypre{spn_a_matrix(inp, mat_lib, fespace, ess_bdr)};
  std::unique_ptr<HypreParMatrix> Mhypre{spn_m_matrix(inp, mat_lib, fespace, ess_bdr)};

  time.stop(time.matrix_assembly);

  slepc_initialize(inp);
  slepc_solver_wrapper(Ahypre.release(), Mhypre.release(), inp, time, res, mat_lib.ngroup()*inp.spn_eq());
  slepc_finalize();

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

  // get the process id so that only the root process writes to terminal
  const int myid{fespace->GetMyRank()};

  if (myid == 0) std::cout << "__spn_order__ = " << inp.spn_order() << std::endl;

  mat_lib.set_nmoment(inp.spn_order()+1);

  time.start(time.matrix_assembly);

  HypreParMatrix *Ahypre{spn_a_matrix(inp, mat_lib, fespace, ess_bdr)};
  HypreParMatrix *Mhypre{spn_m_matrix(inp, mat_lib, fespace, ess_bdr)};

  time.stop(time.matrix_assembly);

  power_iteration_amgx(Ahypre, Mhypre, inp, time, res, mat_lib.ngroup()*inp.spn_eq());

  return;
}


void spn_slepc_manual(Timer& time, const Input& inp,
                       Material_library& mat_lib,
                       const Array<int>& ess_bdr, const ParMesh* const pmesh,
                       ParFiniteElementSpace* const fespace, Result& res)
{

  mat_lib.set_nmoment(inp.spn_order()+1);

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
  slepc.SetTol(std::min(inp.tol.phi, inp.tol.k));
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
                          const Array<int>& ess_bdr,
                          ParMesh* const pmesh,
                          ParFiniteElementSpace* const fespace, Result& angular_flux, Result& scalar_flux)
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
  HypreGMRES sol(MPI_COMM_WORLD);
  // TODO
  // sol.SetKDim(100); // need more but makes more slower
  sol.SetTol(inp.tol.inner_rel);
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
        amg.SetOperator(*A);
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
                           const Array<int>& ess_bdr,
                           ParMesh* const pmesh,
                           ParFiniteElementSpace* const fespace,
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
  HypreGMRES gmres(MPI_COMM_WORLD);
  gmres.SetTol(inp.tol.inner_rel);
  gmres.SetAbsTol(inp.tol.inner_abs);
  gmres.SetMaxIter(inp.tol.max_inner);
  gmres.SetPrintLevel(inp.hypre_print_level());
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
        amg.SetOperator(*A);
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

  ConstantCoefficient benchmark_source{1.0};
  SPNWrapper<ScalarVectorProductCoefficient> saaf_benchmark_source{mat_lib.ngroup(), quad->nd()};
  if (inp.ref == Analytic::saaf_benchmark)
  {
    saaf_benchmark_source.reserve();
    for (int n{0}; n < quad->nd(); ++n)
      for (int g{0}; g < mat_lib.ngroup(); ++g)
        saaf_benchmark_source.emplace_back(benchmark_source, inv_tot_omega(g,n));
  }

  SPNWrapper<std::unique_ptr<ParLinearForm>> b{mat_lib.ngroup(), quad->nd()};
  b.resize();

  for (int n{0}; n < quad->nd(); ++n)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      b(g,n) = std::make_unique<ParLinearForm>(fespace);

      if (inp.ref == Analytic::saaf_benchmark)
      {
        b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{benchmark_source});
        b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{saaf_benchmark_source(g,n)});
      }
      else
      {
        b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{fiss_quad_coeff[g], nullptr});
        b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{inv_tot_omega_fiss_operator(g,n)});
      }

      b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{upscatter_quad_coeff[g], nullptr});
      b(g,n)->AddDomainIntegrator(new QuadratureLFIntegrator{dnscatter_quad_coeff[g], nullptr});
      b(g,n)->AddDomainIntegrator(new DomainLFIntegrator{self_scatter_src[g]});
      b(g,n)->AddDomainIntegrator(new DomainLFGradIntegrator{inv_tot_omega_scat_operator(g,n)});

      b(g,n)->AddBdrFaceIntegrator(
        new BoundaryFlowIntegrator{incoming_flux_coeff(g,n), omega[n], -1.0}, 
        reflective_condition);

    }
  }

  // setup preconditioner
  HypreBoomerAMG amg;
  amg.SetPrintLevel(inp.hypre_print_level());

  // setup solver
  HyprePCG sol(MPI_COMM_WORLD);
  sol.SetTol(inp.tol.inner_rel);
  sol.SetAbsTol(inp.tol.inner_abs);
  sol.SetMaxIter(inp.tol.max_inner);
  sol.SetPrintLevel(inp.hypre_print_level());
  sol.SetPreconditioner(amg);

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

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    scalar_flux(g) = 1.0;
    for (int n{0}; n < quad->nd(); ++n)
      angular_flux(n,g) = 1.0;
  }

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

      double scattering_conv{1e6};
      for (int scattering_iter{0};
          (scattering_iter < inp.tol.max_inner) && 
          (scattering_conv > inp.tol.scatter);
          ++scattering_iter)
      {

        for (int n{0}; n < quad->nd(); ++n)
        {
          time.start("bassemble");
          b(g,n)->Assemble();
          time.stop("bassemble");

          OperatorPtr A;
          Vector B, X;

          a(g,n)->FormLinearSystem(ess_tdof_list, angular_flux(g,n), *b(g,n), A, X, B);
          amg.SetOperator(*A);
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
  slepc.SetTol(std::min(inp.tol.phi, inp.tol.k));
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

  const int myid{fespace->GetMyRank()};

  time.start(time.matrix_assembly);

  HypreParMatrix *Ahypre{saaf_a_matrix(inp, mat_lib, quad, fespace)};
  HypreParMatrix *Mhypre{saaf_m_matrix(inp, mat_lib, quad, fespace)};

  slepc_hypre::inp = &inp;
  slepc_hypre::mat_lib = &mat_lib;
  slepc_hypre::quad = quad;

  time.stop(time.matrix_assembly);

  std::unique_ptr<Memory_stats> mem{nullptr};

  slepc_initialize(inp);
  if (myid == 0)
  {
    mem = std::make_unique<Memory_stats>(Memory_stats::megabyte);
    mem->update_and_print();
    mem->init_background(1000);
  }
  slepc_solver_wrapper(Ahypre, Mhypre, inp, time, angular_flux, mat_lib.ngroup()*quad->nd());
  Ahypre = nullptr;
  Mhypre = nullptr;
  if (myid == 0)
  {
    mem->stop_background();
    mem.reset();
  }
  slepc_finalize();

  return;
}

void saaf_true_amgx_vector(const Quadrature* const quad,
                           Timer &time,
                           const Input &inp,
                           const Material_library &mat_lib,
                           Result &angular_flux)
{

#ifdef AMGX_AVAILABLE

  ParFiniteElementSpace* const fespace{angular_flux(0).ParFESpace()};
  ParMesh* const pmesh{fespace->GetParMesh()};

  std::unique_ptr<HypreParMatrix> Ahypre{nullptr}, Mhypre{nullptr};

  if ((mat_lib.ngroup() == 1) && (mat_lib.mat.size() == 1))
  {
    std::cout << "scattering ratio c=" <<
      (mat_lib.mat[0].rec.sigma_s.Elem(0,0) + mat_lib.mat[0].rec.nusf[0]) /
      mat_lib.mat[0].rec.sigma_t[0] << std::endl;
  }

  time.start(time.matrix_assembly);

  Ahypre.reset(saaf_a_matrix(inp, mat_lib, quad, fespace));
  Mhypre.reset(saaf_m_matrix(inp, mat_lib, quad, fespace));

  time.stop(time.matrix_assembly);

  if (inp.ref == Analytic::saaf_benchmark)
  {

    const int block_rank{mat_lib.ngroup()*quad->nd()};

    constexpr double benchmark_source_val{1.0};

    Vector benchmark_source(mat_lib.ngroup()*quad->nd());
    benchmark_source = benchmark_source_val;
    VectorConstantCoefficient benchmark_source_func{benchmark_source};

    std::vector<Tensor<Vector>> saaf_benchmark_source(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      saaf_benchmark_source.reserve(mat_lib.ngroup()*quad->nd());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          Vector tmp{quad->omega(n)};
          tmp *= benchmark_source_val/tot;
          saaf_benchmark_source[i].AddElement(tmp, n + quad->nd()*g);
        }
      }
    }
    VectorTensorPWConstCoefficient saaf_benchmark_source_func{block_rank, saaf_benchmark_source};

    ParLinearForm b{fespace};
    b.AddDomainIntegrator(new VectorDomainLFIntegrator{benchmark_source_func});
    b.AddDomainIntegrator(new VectorDomainLFGradIntegrator{saaf_benchmark_source_func});
    b.Assemble();
    std::unique_ptr<HypreParVector> bhypre{b.ParallelAssemble()};

    HypreParVector x{*Ahypre};
    x = 0.0;

    const MPI_Comm mycomm{angular_flux(0).ParFESpace()->GetComm()};
    AmgXSolver Aamgx;
    Aamgx.DefaultParameters(AmgXSolver::SOLVER, inp.verbose);
    Aamgx.ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);

    if (inp.ndevices > 0)
    {
      // use MPITeams
      Aamgx.InitMPITeams(mycomm, inp.ndevices);
    }
    else
    {
      // use MPI Exclusive
      Aamgx.InitExclusiveGPU(mycomm);
    }

    Aamgx.SetOperator(*Ahypre);
    Aamgx.Mult(*bhypre, x);
    Aamgx.Finalize();
    angular_flux(0).Distribute(x);

  }
  else
  {
    power_iteration_amgx(Ahypre.release(), Mhypre.release(), inp, time, angular_flux, mat_lib.ngroup()*quad->nd());
  }

#else

  MFEM_ABORT("solver not supported without amgx");

#endif

  return;
}

namespace slepc_split
{
  std::unique_ptr<Solver> sol{nullptr};
  std::unique_ptr<HypreBoomerAMG> amg{nullptr};
  std::unique_ptr<HypreParMatrix> Alc_hypre{nullptr}; // A_{loss+collision}

  const Input *inp{nullptr};
  const Material_library *mat_lib{nullptr};
  const Quadrature *quad{nullptr};
  Timer* time{nullptr};
  
  Result *angular_flux{nullptr};
  Result *scalar_flux{nullptr};

  int solve_counter{0};

  // TODO implement coefficients and computation for this
  // TODO this should moved into a management object
  std::unique_ptr<ParLinearForm> scat_src{nullptr};
  std::unique_ptr<QuadratureSpace> qspace{nullptr};
  std::unique_ptr<QuadratureFunction> scat_quad{nullptr}, scat_saaf_quad{nullptr};
  std::unique_ptr<VectorQuadratureFunctionCoefficient> scat_quad_func{nullptr}, scat_saaf_quad_func{nullptr};

  PetscErrorCode setup(PC pc)
  {
    
    amg = std::make_unique<HypreBoomerAMG>(*Alc_hypre);
    amg->SetPrintLevel(inp->hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib->ngroup()*quad->nd());

    std::unique_ptr<HyprePCG> cg{std::make_unique<HyprePCG>(*Alc_hypre)};
    cg->SetPrintLevel(inp->hypre_print_level());
    cg->SetMaxIter(inp->tol.max_inner);
    cg->SetTol(inp->tol.inner_rel);
    cg->SetAbsTol(inp->tol.inner_abs);
    cg->SetPreconditioner(*amg);

    sol = std::move(cg);

    ParFiniteElementSpace *angular_flux_fespace = (*angular_flux)(0).ParFESpace();
    ParMesh *pmesh = angular_flux_fespace->GetParMesh();
    const int integration_order{angular_flux_fespace->GetOrder(0)+1};
    qspace = std::make_unique<QuadratureSpace>(pmesh, integration_order);
    scat_quad = std::make_unique<QuadratureFunction>(qspace.get(), mat_lib->ngroup()*quad->nd());
    scat_quad_func = std::make_unique<VectorQuadratureFunctionCoefficient>(*scat_quad);
    scat_saaf_quad = std::make_unique<QuadratureFunction>(qspace.get(), mat_lib->ngroup()*quad->nd()*quad->dim());
    scat_saaf_quad_func = std::make_unique<VectorQuadratureFunctionCoefficient>(*scat_saaf_quad);

    scat_src = std::make_unique<ParLinearForm>(angular_flux_fespace);
    scat_src->AddDomainIntegrator(new VectorDomainLFIntegrator{*scat_quad_func});
    scat_src->AddDomainIntegrator(new VectorDomainLFGradIntegrator{*scat_saaf_quad_func});

    return 0;
  }

  PetscErrorCode apply(PC pc, Vec xin, Vec xout)
  {
    const PetscParVector vin{xin, true};
    PetscParVector vout{xout, true};

    // TODO implement scattering convergence criteria

    double scattering_conv{inp->tol.scatter*1e2+1.0};
    double init_norm{0.0};
    for (int scattering_iter{0}; 
        (scattering_iter < inp->tol.max_scatter) &&
        (scattering_conv > inp->tol.scatter);
        ++scattering_iter)
    {

      time->start("distribute");
      (*angular_flux)(0).Distribute(vout);
      time->stop("distribute");

      time->start("scalar_flux");
      calc_scalar_flux_slepc_vector(*angular_flux, quad, *mat_lib, *scalar_flux);
      time->stop("scalar_flux");

      time->start("calc_scatter");
      calc_all_scatter(*scalar_flux, *mat_lib, quad, *scat_quad, *scat_saaf_quad);
      time->stop("calc_scatter");

      time->start("scatter_assemble");
      scat_src->Assemble();
      std::unique_ptr<HypreParVector> scat_vec{scat_src->ParallelAssemble()};
      time->stop("scatter_assemble");

      time->start("scatter_sum");
      (*scat_vec) += vin;
      time->stop("scatter_sum");

      time->start("inner_solve");
      PetscParVector old{vout};
      old = vout;
      sol->Mult(*scat_vec, vout);
      old -= vout;

      if (scattering_iter == 0)
        init_norm = par_norml2(old);
      else
        scattering_conv = par_norml2(old) / init_norm;

      ++solve_counter;
      time->stop("inner_solve");

      if ((*angular_flux)(0).ParFESpace()->GetMyRank() == 0) 
        std::cout << "...scat_it=" << scattering_iter 
                  << " scat_conv=" << scattering_conv 
                  << " change_norm=" << old.Normlinf()
                  << std::endl;

    }

    return 0;
  }

  PetscErrorCode destroy(PC pc)
  {

    sol.reset();
    amg.reset();
    Alc_hypre.reset();

    scat_src.reset();
    qspace.reset();
    scat_quad.reset();
    scat_quad_func.reset();
    scat_saaf_quad.reset();
    scat_saaf_quad_func.reset();

    return 0;
  }

} // namespace slepc_split

void saaf_split_slepc_vector(const Quadrature* const quad,
                             Timer &time,
                             const Input &inp,
                             const Material_library &mat_lib,
                             Result &angular_flux,
                             Result &scalar_flux)
{

  ParFiniteElementSpace *angular_flux_fespace = angular_flux(0).ParFESpace();
  const ParMesh *pmesh = angular_flux_fespace->GetParMesh();
  const int myid{angular_flux_fespace->GetMyRank()};

  std::unique_ptr<HypreParMatrix> Ahypre{saaf_a_matrix(inp, mat_lib, quad, angular_flux_fespace)};
  std::unique_ptr<HypreParMatrix> Mhypre{saaf_m_matrix(inp, mat_lib, quad, angular_flux_fespace)};

  {

    std::vector<DenseMatrix> omega_mat(quad->nd());
    for (int n{0}; n < quad->nd(); ++n)
    {
      omega_mat[n].SetSize(quad->dim());
      MultVVt(quad->omega(n), omega_mat[n]);
    }

    Tensor<Vector> omega;
    omega.reserve(quad->nd()*mat_lib.ngroup());
    for (int n{0}; n < quad->nd(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        omega.AddElement(quad->omega(n), g*quad->nd()+n);
      }
    }
    VectorTensorConstantCoefficient omega_func{quad->nd()*mat_lib.ngroup(), omega};

    std::vector<Vector> total(pmesh->attributes.Max());
    std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      total[i].SetSize(mat_lib.ngroup()*quad->nd());
      leakage[i].reserve(mat_lib.ngroup()*quad->nd());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          omega_mat[n] *= 1.0/tot;
          leakage[i].AddElement(omega_mat[n], g*quad->nd()+n);
          omega_mat[n] *= tot;
          total[i](g*quad->nd()+n) = tot;
        }
      }
    }
    VectorPWConstCoefficient total_func{total};
    MatrixTensorPWConstCoefficient leakage_func{mat_lib.ngroup()*quad->nd(), leakage};

    ParBilinearForm loss_collision{angular_flux_fespace};
    loss_collision.AddDomainIntegrator(new VectorMassIntegrator{total_func});
    loss_collision.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
    loss_collision.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});
    loss_collision.Assemble();
    loss_collision.Finalize();
    slepc_split::Alc_hypre.reset(loss_collision.ParallelAssemble());
  }

  slepc_split::inp = &inp;
  slepc_split::mat_lib = &mat_lib;
  slepc_split::quad = quad;
  slepc_split::angular_flux = &angular_flux;
  slepc_split::scalar_flux = &scalar_flux;
  slepc_split::time = &time;

  MFEMInitializeSlepc(nullptr, nullptr, "slepcrc", nullptr);
  constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};
  {

    PetscParMatrix A{Ahypre.get(), petsc_format};
    PetscParMatrix M{Mhypre.get(), petsc_format};

    if ((petsc_format != Operator::PETSC_MATHYPRE) &&
        (petsc_format != Operator::PETSC_MATSHELL))
    {
      Ahypre.reset();
      Mhypre.reset();
    }

    SlepcEigenSolver slepc{MPI_COMM_WORLD};
    slepc.SetNumModes(1);
    slepc.SetTol(std::min(inp.tol.k, inp.tol.phi));
    slepc.SetMaxIter(inp.tol.max_outer);
    slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
    slepc.SetOperators(A, M);

    PetscErrorCode ierr;
    PC preconditioner;
    ST spectral_transform;
    KSP krylov_solver;
    ierr = EPSGetST(slepc, &spectral_transform); PCHKERRQ(slepc, ierr);
    ierr = STGetKSP(spectral_transform, &krylov_solver); PCHKERRQ(slepc, ierr);
    ierr = PCCreate(MPI_COMM_WORLD, &preconditioner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner, PCSHELL); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetSetUp(preconditioner, slepc_split::setup); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetApply(preconditioner, slepc_split::apply); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetDestroy(preconditioner, slepc_split::destroy); PCHKERRQ(slepc, ierr);
    ierr = KSPSetPC(krylov_solver, preconditioner); PCHKERRQ(slepc, ierr);

    time.start(time.solver);
    slepc.Solve();
    time.stop(time.solver);

    ierr = PCDestroy(&preconditioner); PCHKERRQ(slepc, ierr);

    if (slepc.GetNumConverged() > 0)
    {
      time.start("parse_solution");
      slepc.GetEigenvalue(0, angular_flux.keff);
      angular_flux.keff = 1.0 / angular_flux.keff;
      scalar_flux.keff = angular_flux.keff;
      Vector tmp(angular_flux_fespace->GetTrueVSize());
      slepc.GetEigenvector(0, tmp);
      angular_flux(0).Distribute(tmp);
      time.stop("parse_solution");
    }

  }
  MFEMFinalizeSlepc();

  if (myid == 0) std::cout << "solve_counter=" << slepc_split::solve_counter << std::endl;

  return;
}

void saaf_split_power_iteration(const Quadrature* const quad,
                                Timer& time,
                                const Input& inp,
                                const Material_library& mat_lib,
                                Result& angular_flux,
                                Result& scalar_flux)
{

  ParFiniteElementSpace *angular_flux_fespace = angular_flux(0).ParFESpace();
  ParMesh *pmesh = angular_flux_fespace->GetParMesh();

  const int myid{angular_flux_fespace->GetMyRank()};

  std::unique_ptr<HypreParMatrix> Ahypre{saaf_a_matrix(inp, mat_lib, quad, angular_flux_fespace)};
  std::unique_ptr<HypreParMatrix> Mhypre{saaf_m_matrix(inp, mat_lib, quad, angular_flux_fespace)};
  std::unique_ptr<HypreParMatrix> Alc_hypre{nullptr};

  {

    std::vector<DenseMatrix> omega_mat(quad->nd());
    for (int n{0}; n < quad->nd(); ++n)
    {
      omega_mat[n].SetSize(quad->dim());
      MultVVt(quad->omega(n), omega_mat[n]);
    }

    Tensor<Vector> omega;
    omega.reserve(quad->nd()*mat_lib.ngroup());
    for (int n{0}; n < quad->nd(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        omega.AddElement(quad->omega(n), g*quad->nd()+n);
      }
    }
    VectorTensorConstantCoefficient omega_func{quad->nd()*mat_lib.ngroup(), omega};

    std::vector<Vector> total(pmesh->attributes.Max());
    std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      total[i].SetSize(mat_lib.ngroup()*quad->nd());
      leakage[i].reserve(mat_lib.ngroup()*quad->nd());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          omega_mat[n] *= 1.0/tot;
          leakage[i].AddElement(omega_mat[n], g*quad->nd()+n);
          omega_mat[n] *= tot;
          total[i](g*quad->nd()+n) = tot;
        }
      }
    }
    VectorPWConstCoefficient total_func{total};
    MatrixTensorPWConstCoefficient leakage_func{mat_lib.ngroup()*quad->nd(), leakage};

    ParBilinearForm loss_collision{angular_flux_fespace};
    loss_collision.AddDomainIntegrator(new VectorMassIntegrator{total_func});
    loss_collision.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
    loss_collision.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});
    loss_collision.Assemble();
    loss_collision.Finalize();
    Alc_hypre.reset(loss_collision.ParallelAssemble());

  }

  const int integration_order{angular_flux_fespace->GetOrder(0)+1};
  QuadratureSpace qspace{pmesh, integration_order};

  QuadratureFunction scat_quad{&qspace, mat_lib.ngroup()*quad->nd()};
  VectorQuadratureFunctionCoefficient scat_quad_func{scat_quad};

  QuadratureFunction scat_saaf_quad{&qspace, mat_lib.ngroup()*quad->nd()*quad->dim()};
  VectorQuadratureFunctionCoefficient scat_saaf_quad_func{scat_saaf_quad};

  ParLinearForm scat_src{angular_flux_fespace};
  scat_src.AddDomainIntegrator(new VectorDomainLFIntegrator{scat_quad_func});
  scat_src.AddDomainIntegrator(new VectorDomainLFGradIntegrator{scat_saaf_quad_func});

  HypreBoomerAMG amg{*Alc_hypre};
  amg.SetPrintLevel(inp.hypre_print_level());
  HYPRE_BoomerAMGSetNumFunctions(amg, mat_lib.ngroup()*quad->nd());

  HyprePCG cg{*Alc_hypre};
  cg.SetPrintLevel(inp.hypre_print_level());
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);
  cg.SetPreconditioner(amg);

  Solver *sol{&cg};

  HypreParVector x{*Alc_hypre};
  //x.Randomize(730);
  x = 1.0;

  angular_flux(0) = 1.0;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux(g) = 1.0;

  double k_conv{inp.tol.k*1e2+1.0};
  double x_conv{inp.tol.phi*1e2+1.0};

  double k{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer) &&
      ((k_conv > inp.tol.k) || (x_conv > inp.tol.phi)); ++iter)
  {

    const double k_old{k};
    HypreParVector x_old{x};
    x_old = x;

    HypreParVector b{x};
    Mhypre->Mult(x, b, 1.0, 0.0);
    b /= par_norml2(x);

    HypreParVector bsto{b};
    bsto = b;

    double scattering_conv{1e6};
    for (int scattering_iter{0};
        (scattering_iter < inp.tol.max_scatter) &&
        (scattering_conv > inp.tol.scatter);
        ++scattering_iter)
    {

      b = bsto;

      if (sol == &cg)
      {
        angular_flux(0).Distribute(x);
        flux_sign_fixup(angular_flux(0));
        calc_scalar_flux_slepc_vector(angular_flux, quad, mat_lib, scalar_flux);
        calc_all_scatter(scalar_flux, mat_lib, quad, scat_quad, scat_saaf_quad);
        scat_src.Assemble();
        std::unique_ptr<HypreParVector> scat_vec{scat_src.ParallelAssemble()};
        b += *scat_vec;
      }

      HypreParVector old{x};
      old = x;
      sol->Mult(b, x);
      old -= x;

      /*
      if (scattering_iter == 0)
        init_norm = par_norml2(old);
      else
        scattering_conv = par_norml2(old) / init_norm;
      */

      const double local_scattering_conv{old.Normlinf() / x.Normlinf()};
      MPI_Allreduce(&local_scattering_conv, &scattering_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if (myid == 0) 
        std::cout << "...scat_iter=" << scattering_iter
                  << " scat_conv=" << scattering_conv
                  << std::endl;

    }

    const double local_k{(x_old*x)/par_norml2(x_old)};
    MPI_Allreduce(&local_k, &k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    k_conv = std::abs(k - k_old);
    x_old -= x;
    x_conv = par_norml2(x_old);

    if (myid == 0) std::cout << "iter=" << iter
                             << " k=" << k 
                             << " dk=" << k_conv 
                             << " dx=" << x_conv 
                             << std::endl;


  }

  return;
}

void saaf_self_power_iteration(const Quadrature* const quad,
                               Timer& time,
                               const Input& inp,
                               const Material_library& mat_lib,
                               Result& angular_flux,
                               Result& scalar_flux)
{

  ParFiniteElementSpace *angular_flux_fespace = angular_flux(0).ParFESpace();
  ParMesh *pmesh = angular_flux_fespace->GetParMesh();

  const int myid{angular_flux_fespace->GetMyRank()};
  const int vdim{mat_lib.ngroup()*quad->nd()};
  constexpr int skip_zeros{2};
  constexpr bool compute_residual{true};

  std::unique_ptr<HypreParMatrix> Ahypre{saaf_a_matrix(inp, mat_lib, quad, angular_flux_fespace)};
  std::unique_ptr<HypreParMatrix> Mhypre{saaf_m_matrix(inp, mat_lib, quad, angular_flux_fespace)};

  std::unique_ptr<HypreParMatrix> A_lc_self{nullptr};

  { // A_lc_self

    std::vector<DenseMatrix> omega_mat(quad->nd());
    for (int n{0}; n < quad->nd(); ++n)
    {
      omega_mat[n].SetSize(quad->dim());
      MultVVt(quad->omega(n), omega_mat[n]);
    }

    Tensor<Vector> omega;
    omega.reserve(quad->nd()*mat_lib.ngroup());
    for (int n{0}; n < quad->nd(); ++n)
    {
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        omega.AddElement(quad->omega(n), g*quad->nd()+n);
      }
    }
    VectorTensorConstantCoefficient omega_func{vdim, omega};

    std::vector<Vector> total(pmesh->attributes.Max());
    std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
    std::vector<DenseMatrix> scat(pmesh->attributes.Max());
    std::vector<Tensor<Vector>> scat_saaf(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      total[i].SetSize(vdim);
      leakage[i].reserve(vdim);
      scat[i].SetSize(vdim);
      scat_saaf[i].reserve(mat_lib.ngroup()*quad->nd()*quad->nd());
      total[i] = 0.0;
      scat[i] = 0.0;
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        const double self_scat{mat_lib.mat[i].rec.sigma_s.Elem(g,g)};
        for (int n{0}; n < quad->nd(); ++n)
        {

          const int idx{g*quad->nd()+n};

          omega_mat[n] *= 1.0/tot;
          leakage[i].AddElement(omega_mat[n], idx);
          omega_mat[n] *= tot;

          total[i](idx) = tot;

          const int row{g*quad->nd()+n};
          for (int nprime{0}; nprime < quad->nd(); ++nprime)
          {
            const int col{g*quad->nd()+nprime};
            const double val{-self_scat*quad->weight(nprime)};

            scat[i](row,col) = val;

            Vector v{quad->omega(n)};
            v *= val/tot;
            scat_saaf[i].AddElement(v, row, col);
          }

        }
      }
      scat_saaf[i].Transpose(); // related to the need for the transpose integrator
    }
    VectorPWConstCoefficient total_func{total};
    MatrixTensorPWConstCoefficient leakage_func{vdim, leakage};
    MatrixPWConstCoefficient scat_func{scat};
    VectorTensorPWConstCoefficient scat_saaf_func{vdim, scat_saaf};

    ParBilinearForm a{angular_flux_fespace};
    a.AddDomainIntegrator(new VectorMassIntegrator{total_func});
    a.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
    a.AddDomainIntegrator(new VectorMassIntegrator{scat_func});
    a.AddDomainIntegrator(new TransposeIntegrator{
        new VectorConvectionIntegrator{scat_saaf_func}});
    a.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});
    a.Assemble(skip_zeros);
    a.Finalize(skip_zeros);
    A_lc_self.reset(a.ParallelAssemble());

  } // A_lc_self

  const int integration_order{angular_flux_fespace->GetOrder(0)+1};
  QuadratureSpace qspace{pmesh, integration_order};

  QuadratureFunction scat_quad{&qspace, vdim};
  VectorQuadratureFunctionCoefficient scat_quad_func{scat_quad};

  QuadratureFunction scat_saaf_quad{&qspace, vdim*quad->dim()};
  VectorQuadratureFunctionCoefficient scat_saaf_quad_func{scat_saaf_quad};

  ParLinearForm scat_src{angular_flux_fespace};
  scat_src.AddDomainIntegrator(new VectorDomainLFIntegrator{scat_quad_func});
  scat_src.AddDomainIntegrator(new VectorDomainLFGradIntegrator{scat_saaf_quad_func});

  constexpr bool use_self_matrix{true};

  HypreBoomerAMG amg;

  if (use_self_matrix)
    amg.SetOperator(*A_lc_self);
  else
    amg.SetOperator(*Ahypre);

  amg.SetPrintLevel(inp.hypre_print_level());
  HYPRE_BoomerAMGSetNumFunctions(amg, vdim);

  HypreGMRES gmres{MPI_COMM_WORLD};
  //gmres.SetPrintLevel(9);
  gmres.SetKDim(50); // Default. TODO from user input.

  if (use_self_matrix)
    gmres.SetOperator(*A_lc_self);
  else
    gmres.SetOperator(*Ahypre);

  gmres.SetMaxIter(inp.tol.max_inner);
  gmres.SetTol(inp.tol.inner_rel);
  gmres.SetAbsTol(inp.tol.inner_abs);
  gmres.SetPreconditioner(amg);

  HypreParVector x{*A_lc_self};
  x = 1.0;

  angular_flux(0) = 1.0;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux(g) = 1.0;

  double k_conv{inp.tol.k*1e2+1.0};
  double x_conv{inp.tol.phi*1e2+1.0};
  double k{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer) &&
      ((k_conv > inp.tol.k) || (x_conv > inp.tol.phi)); ++iter)
  {

    HypreParVector b{x};
    Mhypre->Mult(x, b, 1.0, 0.0);
    b /= k;

    // currently Jacobi in energy treatment
    calc_all_scatter(scalar_flux, mat_lib, quad, scat_quad, scat_saaf_quad, false);
    scat_src.Assemble();
    std::unique_ptr<HypreParVector> scat_vec{scat_src.ParallelAssemble()};
    //b += *scat_vec;

    HypreParVector x_old{x};
    x_old = x;
    gmres.Mult(b, x);

    angular_flux(0).Distribute(x);
    flux_sign_fixup(angular_flux(0));
    calc_scalar_flux_slepc_vector(angular_flux, quad, mat_lib, scalar_flux);

    const double k_old{k};
    k *= par_norml2(x)/par_norml2(x_old);

    // convergence criteria
    k_conv = std::abs(k - k_old);
    x_old -= x;
    x_conv = par_norml2(x_old);

    if (compute_residual)
    {
      // r = A*x - M*x/k
      HypreParVector r0{x};
      Ahypre->Mult(x, r0);

      HypreParVector r1{x};
      Mhypre->Mult(x, r1);
      r1 /= k;

      r0 -= r1;
      std::cout << "residual=" << par_norml2(r0) << std::endl;
    }

    if (myid == 0) std::cout << "iter=" << iter
                             << " k=" << k 
                             << " dk=" << k_conv 
                             << " dx=" << x_conv 
                             << std::endl;

  }
  if (myid == 0) std::cout << "ITERATIONS_FINISHED!!!" << std::endl;

  angular_flux.keff = k;
  scalar_flux.keff = k;
  angular_flux(0).Distribute(x);
  flux_sign_fixup(angular_flux(0));

  return;
}

namespace slepc_diffusion
{

  std::unique_ptr<Solver> sol{nullptr};
  std::unique_ptr<HypreParMatrix> Alo{nullptr};

  std::unique_ptr<HypreBoomerAMG> amg{nullptr};

  const Input *inp{nullptr};
  const Material_library *mat_lib{nullptr};
  const Quadrature *quad{nullptr};
  ParGridFunction *angular_flux{nullptr};
  ParGridFunction *scalar_flux{nullptr};

  PetscErrorCode setup(PC pc)
  {

    MFEM_VERIFY(scalar_flux->ParFESpace()->GetOrdering() == Ordering::byVDIM,
      "scalar flux restriction/prolongation requires orderging byVDIM");
    MFEM_VERIFY(angular_flux->ParFESpace()->GetOrdering() == Ordering::byVDIM,
      "angular flux restriction/prolongation requires orderging byVDIM");

    amg = std::make_unique<HypreBoomerAMG>(*Alo);
    amg->SetPrintLevel(inp->hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib->ngroup());


    std::unique_ptr<HypreGMRES> gmres{std::make_unique<HypreGMRES>(MPI_COMM_WORLD)};
    gmres->SetPrintLevel(inp->hypre_print_level());
    gmres->SetOperator(*Alo);
    gmres->SetMaxIter(inp->tol.max_inner);
    gmres->SetTol(inp->tol.inner_rel);
    gmres->SetAbsTol(inp->tol.inner_abs);
    gmres->SetKDim(50); // Default. Consider setting from user input.
    gmres->SetPreconditioner(*amg);

    sol = std::move(gmres);
    return PetscErrorCode{0};
  }

  PetscErrorCode apply(PC pc, Vec xin, Vec xout)
  {
    const PetscParVector vin{xin, true};
    PetscParVector vout{xout, true};

    // perform restriction (calculate scalar flux)
    HypreParVector scalar_in{scalar_flux->ParFESpace()};
    scalar_in = 0.0;
    const int size{scalar_in.Size() / mat_lib->ngroup()};
    for (int i{0}; i < size; ++i)
    {
      for (int g{0}; g < mat_lib->ngroup(); ++g)
      {
        for (int n{0}; n < quad->nd(); ++n)
        {
          scalar_in(i*mat_lib->ngroup() + g) +=
            vin(i*mat_lib->ngroup()*quad->nd() + g*quad->nd() + n) * quad->weight(n);
        }
      }
    }

    // TODO set boundary conditions appropriately

    HypreParVector scalar_out{scalar_in};
    scalar_out = scalar_in;
    sol->Mult(scalar_in, scalar_out);

    // perform prolongation
    for (int i{0}; i < size; ++i)
    {
      for (int g{0}; g < mat_lib->ngroup(); ++g)
      {

        Vector weight(quad->nd());
        for (int n{0}; n < quad->nd(); ++n)
          weight(n) = vin(i*mat_lib->ngroup()*quad->nd() + g*quad->nd() + n);
        weight /= weight.Sum();
        weight *= scalar_out(i*mat_lib->ngroup()+g);
        for (int n{0}; n < quad->nd(); ++n)
          vout(i*mat_lib->ngroup()*quad->nd() + g*quad->nd() + n) = weight(n);

        /*
        for (int n{0}; n < quad->nd(); ++n)
          vout(i*mat_lib->ngroup()*quad->nd() + g*quad->nd() + n) = 
            scalar_out(i*mat_lib->ngroup() + g) / quad->weight(n);
        */

      }
    }

    scalar_flux->Distribute(scalar_out);
    angular_flux->Distribute(vout);

    return PetscErrorCode{0};
  }

  PetscErrorCode destroy(PC pc)
  {
    amg.reset();
    sol.reset();
    Alo.reset();
    return PetscErrorCode{0};
  }

} // namespace slepc_diffusion

void saaf_true_slepc_diffusion_vector(const Quadrature* const quad,
                                      Timer& time,
                                      const Input& inp,
                                      Material_library& mat_lib,
                                      Result& angular_flux,
                                      Result& scalar_flux)
{

  ParFiniteElementSpace *angular_flux_fespace{angular_flux(0).ParFESpace()};
  ParFiniteElementSpace *scalar_flux_fespace{scalar_flux(0).ParFESpace()};

  ParMesh *pmesh{angular_flux_fespace->GetParMesh()};

  time.start(time.matrix_assembly);
  std::unique_ptr<HypreParMatrix> Ahypre{saaf_a_matrix(inp, mat_lib, quad, angular_flux_fespace)};
  std::unique_ptr<HypreParMatrix> Mhypre{saaf_m_matrix(inp, mat_lib, quad, angular_flux_fespace)};
  
  MFEM_VERIFY(inp.spn_order() == 1, "only works with diffusion (not  SPN) for now");

  {
    // all boundaries will be essential (Derichlet) to match angular flux
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 1;
    mat_lib.set_nmoment(inp.spn_order()+1);
    slepc_diffusion::Alo.reset(spn_a_matrix(inp, mat_lib, scalar_flux_fespace, ess_bdr));
  }

  time.stop(time.matrix_assembly);

  slepc_diffusion::inp = &inp;
  slepc_diffusion::mat_lib = &mat_lib;
  slepc_diffusion::quad = quad;
  slepc_diffusion::angular_flux = &angular_flux(0);
  slepc_diffusion::scalar_flux = &scalar_flux(0);

  MFEMInitializeSlepc(nullptr, nullptr, "slepcrc", nullptr); // also calls PetscInitialize

  //constexpr Operator::Type petsc_format{Operator::PETSC_MATHYPRE};
  constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};

  {

    PetscParMatrix A{Ahypre.get(), petsc_format};
    PetscParMatrix M{Mhypre.get(), petsc_format};

    // no need to call MatSetBlockSize as we are not using multigrid with the
    // PETSc matrices

    if ((petsc_format != Operator::PETSC_MATHYPRE) &&
        (petsc_format != Operator::PETSC_MATSHELL))
    {
      Ahypre.reset();
      Mhypre.reset();
    }

    SlepcEigenSolver slepc{MPI_COMM_WORLD};
    slepc.SetNumModes(1);
    slepc.SetTol(std::min(inp.tol.k, inp.tol.phi));
    slepc.SetMaxIter(inp.tol.max_outer);
    slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
    slepc.SetOperators(A, M);

    PetscErrorCode ierr;
    PC preconditioner;
    ST spectral_transform;
    KSP krylov_solver;
    ierr = EPSGetST(slepc, &spectral_transform); PCHKERRQ(slepc, ierr);
    ierr = STGetKSP(spectral_transform, &krylov_solver); PCHKERRQ(slepc, ierr);
    ierr = PCCreate(MPI_COMM_WORLD, &preconditioner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner, PCSHELL); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetSetUp(preconditioner, slepc_diffusion::setup); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetApply(preconditioner, slepc_diffusion::apply); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetDestroy(preconditioner, slepc_diffusion::destroy); PCHKERRQ(slepc, ierr);
    ierr = KSPSetPC(krylov_solver, preconditioner); PCHKERRQ(slepc, ierr);

    time.start(time.solver);
    slepc.Solve();
    time.stop(time.solver);

    ierr = PCDestroy(&preconditioner); PCHKERRQ(slepc, ierr);

    if (slepc.GetNumConverged() > 0)
    {
      time.start("parse_solution");
      slepc.GetEigenvalue(0, angular_flux.keff);
      angular_flux.keff = 1.0 / angular_flux.keff;
      scalar_flux.keff = angular_flux.keff;

      Vector tmp(angular_flux_fespace->GetTrueVSize());
      slepc.GetEigenvector(0, tmp);
      angular_flux(0).Distribute(tmp);
      time.stop("parse_solution");
    }

  }

  MFEMFinalizeSlepc();

  // TODO calculate scalar flux

  return;
}

namespace slepc_transport
{
  std::unique_ptr<mfem::Solver> sol{nullptr};
  std::unique_ptr<mfem::HypreBoomerAMG> amg{nullptr};
  std::unique_ptr<mfem::HypreParMatrix> A_lo{nullptr};

  const Input *inp{nullptr};
  const Material_library *mat_lib{nullptr};
  const Quadrature *q_lo{nullptr};
  const Quadrature *q_hi{nullptr};
  ParFiniteElementSpace *fespace_lo{nullptr};

  int fine_per_coarse{0};

  PetscErrorCode setup(PC pc)
  {

    amg = std::make_unique<HypreBoomerAMG>(*A_lo);
    amg->SetPrintLevel(inp->hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib->ngroup()*q_lo->nd());

    std::unique_ptr<mfem::HypreGMRES> gmres{std::make_unique<HypreGMRES>(MPI_COMM_WORLD)};
    gmres->SetPrintLevel(inp->hypre_print_level());
    gmres->SetOperator(*A_lo);
    gmres->SetMaxIter(inp->tol.max_inner);
    gmres->SetTol(inp->tol.inner_rel);
    gmres->SetAbsTol(inp->tol.inner_abs);
    gmres->SetKDim(500); // Default. Consider setting from user input.
    gmres->SetPreconditioner(*amg);

    fine_per_coarse = q_hi->nd() / q_lo->nd();

    sol = std::move(gmres);
    return 0;
  }

  PetscErrorCode apply(PC pc, Vec xin, Vec xout)
  {
    const PetscParVector vin_hi{xin, true};
    PetscParVector vout_hi{xout, true};

    // restriction
    HypreParVector vin_lo{fespace_lo};
    const int size{vin_lo.Size() / (mat_lib->ngroup()*q_lo->nd())};
    for (int i{0}; i < size; ++i)
    {
      for (int g{0}; g < mat_lib->ngroup(); ++g)
      {
        for (int n{0}; n < q_lo->nd(); ++n)
        {
          const int idx{i*mat_lib->ngroup()*q_lo->nd() + g*q_lo->nd() + n};
          vin_lo(idx) = 0.0;
          for (int ntilde{0}; ntilde < fine_per_coarse; ++ntilde)
          {
            vin_lo(idx) += 
              vin_hi(i*mat_lib->ngroup()*q_hi->nd() + g*q_hi->nd() + n*fine_per_coarse + ntilde);
          }
        }
      }
    }

    HypreParVector vout_lo{vin_lo};
    sol->Mult(vin_lo, vout_lo);

    // projection
    for (int i{0}; i < size; ++i)
    {
      for (int g{0}; g < mat_lib->ngroup(); ++g)
      {
        for (int n{0}; n < q_lo->nd(); ++n)
        {

          Vector weight(fine_per_coarse);
          for (int ntilde{0}; ntilde < fine_per_coarse; ++ntilde)
            weight(ntilde) = vin_hi(i*mat_lib->ngroup()*q_hi->nd() + g*q_hi->nd() + n*fine_per_coarse + ntilde);
          weight /= weight.Sum();
          weight *= vout_lo(i*mat_lib->ngroup()*q_lo->nd() + g*q_lo->nd() + n);

          for (int ntilde{0}; ntilde < fine_per_coarse; ++ntilde)
            vout_hi(i*mat_lib->ngroup()*q_hi->nd() + g*q_hi->nd() + n*fine_per_coarse + ntilde) = weight(ntilde);

        }
      }
    }

    return 0;
  }

  PetscErrorCode destroy(PC pc)
  {
    sol.reset();
    amg.reset();
    A_lo.reset();
    return 0;
  }

} // namespace slepc_transport

void saaf_true_slepc_transport_vector(const Quadrature* const quad_hi,
                                      Timer& time,
                                      const Input& inp,
                                      const Material_library& mat_lib,
                                      Result& angular_flux)
{
  ParFiniteElementSpace* fespace_hi{angular_flux(0).ParFESpace()};
  ParMesh* pmesh{fespace_hi->GetParMesh()};

  constexpr int quad_order_lo{2};
  std::unique_ptr<Quadrature> quad_lo{
    std::make_unique<Level_symmetric_quadrature>
    (quad_order_lo, pmesh->SpaceDimension())};
  std::unique_ptr<ParFiniteElementSpace> fespace_lo{
    std::make_unique<ParFiniteElementSpace>
    (pmesh, fespace_hi->FEColl(), quad_lo->nd()*mat_lib.ngroup(), Ordering::byVDIM)};

#if 0
  ParGridFunction angular_flux_lo{fespace_lo.get()};
  angular_flux_lo = 69.;

  std::unique_ptr<const HypreParVector> vin_hi{angular_flux(0).ParallelAssemble()};

  // restriction
  // TODO this only works on one processor for now :(
  HypreParVector vin_lo{fespace_lo.get()};
  const int fine_per_coarse {quad_hi->nd() / quad_lo->nd()};
  const int size{vin_lo.Size() / (mat_lib.ngroup()*quad_lo->nd())};
  for (int i{0}; i < size; ++i)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int n{0}; n < quad_lo->nd(); ++n)
      {
        const int idx{i*mat_lib.ngroup()*quad_lo->nd() + g*quad_lo->nd() + n};
        vin_lo(idx) = 0.0;
        for (int ntilde{0}; ntilde < fine_per_coarse; ++ntilde)
        {
          vin_lo(idx) += 
            (*vin_hi)(i*mat_lib.ngroup()*quad_hi->nd() + g*quad_hi->nd() + n*fine_per_coarse + ntilde);
        }
      }
    }
  }

  angular_flux_lo.Distribute(vin_lo);

  {
    ParaViewDataCollection output("debug_output", pmesh);
    output.RegisterField("psi_hi", &angular_flux(0));
    output.RegisterField("psi_lo", &angular_flux_lo);
    //output.SetDataFormat(VTKFormat::ASCII);
    output.Save();
  }

  return;
#endif

  time.start(time.matrix_assembly);
  std::unique_ptr<HypreParMatrix> A_hi{saaf_a_matrix(inp, mat_lib, quad_hi, fespace_hi)};
  std::unique_ptr<HypreParMatrix> M_hi{saaf_m_matrix(inp, mat_lib, quad_hi, fespace_hi)};
  slepc_transport::A_lo.reset(saaf_a_matrix(inp, mat_lib, quad_lo.get(), fespace_lo.get()));
  time.stop(time.matrix_assembly);

  slepc_transport::inp = &inp;
  slepc_transport::mat_lib = &mat_lib;
  slepc_transport::q_hi = quad_hi;
  slepc_transport::q_lo = quad_lo.get();
  slepc_transport::fespace_lo = fespace_lo.get();

  MFEMInitializeSlepc(nullptr, nullptr, "slepcrc", nullptr);
  constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};
  {
    
    PetscParMatrix A{A_hi.get(), petsc_format};
    PetscParMatrix M{M_hi.get(), petsc_format};

    if ((petsc_format != Operator::PETSC_MATHYPRE) &&
        (petsc_format != Operator::PETSC_MATSHELL))
    {
      A_hi.reset();
      M_hi.reset();
    }

    SlepcEigenSolver slepc{MPI_COMM_WORLD};
    slepc.SetNumModes(1);
    slepc.SetTol(std::min(inp.tol.phi, inp.tol.k));
    slepc.SetMaxIter(inp.tol.max_outer);
    slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
    slepc.SetOperators(A, M);

    PetscErrorCode ierr;
    PC preconditioner;
    ST spectral_transform;
    KSP krylov_solver;
    ierr = EPSGetST(slepc, &spectral_transform); PCHKERRQ(slepc, ierr);
    ierr = STGetKSP(spectral_transform, &krylov_solver); PCHKERRQ(slepc, ierr);
    ierr = PCCreate(MPI_COMM_WORLD, &preconditioner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner, PCSHELL); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetSetUp(preconditioner, slepc_transport::setup); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetApply(preconditioner, slepc_transport::apply); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetDestroy(preconditioner, slepc_transport::destroy); PCHKERRQ(slepc, ierr);
    ierr = KSPSetPC(krylov_solver, preconditioner); PCHKERRQ(slepc, ierr);

    time.start(time.solver);
    slepc.Solve();
    time.stop(time.solver);

    ierr = PCDestroy(&preconditioner); PCHKERRQ(slepc, ierr);

    if (slepc.GetNumConverged() > 0)
    {
      time.start("parse_solution");
      slepc.GetEigenvalue(0, angular_flux.keff);
      angular_flux.keff = 1.0 / angular_flux.keff;

      Vector tmp(fespace_hi->GetTrueVSize());
      slepc.GetEigenvector(0, tmp);
      angular_flux(0).Distribute(tmp);
      time.stop("parse_solution");
    }
  }
  MFEMFinalizeSlepc();

  return;
}

void saaf_angle_power_iteration(const Quadrature* const quad,
                                Timer& time,
                                const Input& inp,
                                const Material_library& mat_lib,
                                Result& angular_flux,
                                Result& scalar_flux,
                                ParGridFunction &power,
                                ParGridFunction &temperature)
{

  constexpr bool use_gpu{false};
  constexpr bool use_dsa{true};

  ParFiniteElementSpace *angular_flux_fespace = angular_flux(0).ParFESpace();
  ParFiniteElementSpace *scalar_flux_fespace = scalar_flux(0).ParFESpace();
  ParMesh *pmesh = angular_flux_fespace->GetParMesh();

  constexpr int skip_zeros{2};
  const int myid{angular_flux_fespace->GetMyRank()};

  // transport streaming+collision operator
  std::vector<std::unique_ptr<HypreParMatrix>> Amat(mat_lib.ngroup());
  std::vector<std::unique_ptr<HypreBoomerAMG>> Aprec(mat_lib.ngroup()); // could be HypreSolver
  std::vector<std::unique_ptr<Solver>> Asol(mat_lib.ngroup());

  // dsa streaming+removal operator
  std::vector<std::unique_ptr<HypreParMatrix>> Dmat(mat_lib.ngroup());
  std::vector<std::unique_ptr<HypreBoomerAMG>> Dprec(mat_lib.ngroup()); // could be HypreSolver
  std::vector<std::unique_ptr<Solver>> Dsol(mat_lib.ngroup());

  time.start(time.matrix_assembly);

  { // streaming+collision

    std::vector<DenseMatrix> omega_mat(quad->nd());
    for (int n{0}; n < quad->nd(); ++n)
    {
      omega_mat[n].SetSize(quad->dim());
      MultVVt(quad->omega(n), omega_mat[n]);
    }

    Tensor<Vector> omega;
    omega.reserve(quad->nd());
    for(int n{0}; n < quad->nd(); ++n)
    {
      omega.AddElement(quad->omega(n), n);
    }
    VectorTensorConstantCoefficient omega_func{quad->nd(), omega};
    
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      ParBilinearForm a{angular_flux_fespace};

      std::vector<Vector> total(pmesh->attributes.Max());
      std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};

        leakage[i].reserve(quad->nd());
        for (int n{0}; n < quad->nd(); ++n)
        {
          const double val{1.0/tot};
          omega_mat[n] *= val;
          leakage[i].AddElement(omega_mat[n], n);
          omega_mat[n] *= 1.0/val;
        }
        
        total[i].SetSize(quad->nd());
        total[i] = tot;

      }

      VectorPWConstCoefficient total_func{total};
      MatrixTensorPWConstCoefficient leakage_func{quad->nd(), leakage};
      a.AddDomainIntegrator(new VectorMassIntegrator{total_func});
      a.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
      a.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});

      a.Assemble(skip_zeros); // note: no essential bcs in saaf
      a.Finalize(skip_zeros);
      Amat[g].reset(a.ParallelAssemble());

      if (use_gpu)
      {

#ifdef AMGX_AVAILABLE

        std::unique_ptr<AmgXSolver> this_sol{std::make_unique<AmgXSolver>()};
        this_sol->DefaultParameters(AmgXSolver::SOLVER, inp.verbose);
        this_sol->ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);

        if (inp.ndevices > 0)
          this_sol->InitMPITeams(MPI_COMM_WORLD, inp.ndevices); // use MPITeams
        else
          this_sol->InitExclusiveGPU(MPI_COMM_WORLD); // use MPI Exclusive

        this_sol->SetOperator(*Amat[g]);
        Asol[g] = std::move(this_sol);

        // Amat no longer needs to be stored on cpu
        Amat[g].reset();

#else

        MFEM_ABORT("solver not supported without amgx");

#endif

      }
      else
      {

        std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*Amat[g])};
        amg->SetPrintLevel(inp.hypre_print_level());
        HYPRE_BoomerAMGSetNumFunctions(*amg, quad->nd());
        Aprec[g] = std::move(amg);

        std::unique_ptr<HyprePCG> this_sol{std::make_unique<HyprePCG>(MPI_COMM_WORLD)};
        this_sol->SetPrintLevel(2);
        this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
        this_sol->SetOperator(*Amat[g]);
        this_sol->SetTol(inp.tol.inner_rel);
        this_sol->SetAbsTol(inp.tol.inner_abs);
        this_sol->SetMaxIter(inp.tol.max_inner);
        Asol[g] = std::move(this_sol);

      }

    } // g = 1,mat_lib.ngroup()

  } // streaming+collision

  { // DSA operator

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      Vector diffusive(pmesh->attributes.Max());
      Vector removal(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
      {
        const XSRecord &rec{mat_lib.mat[i].rec};
        diffusive(i) = 1.0 / (3.0*rec.sigma_t[g]);
        removal(i)   = rec.sigma_t[g] - rec.sigma_s.Elem(g,g);
      }
      PWConstCoefficient diffusive_func{diffusive};
      PWConstCoefficient removal_func{removal};

      ConstantCoefficient albedo{0.5};

      ParBilinearForm d{scalar_flux_fespace};
      d.AddDomainIntegrator(new DiffusionIntegrator{diffusive_func});
      d.AddDomainIntegrator(new MassIntegrator{removal_func});
      d.AddBoundaryIntegrator(new MassIntegrator{albedo});

      d.Assemble(skip_zeros);
      d.Finalize(skip_zeros);
      Dmat[g].reset(d.ParallelAssemble());

      if (use_gpu)
      {

#ifdef AMGX_AVAILABLE

        std::unique_ptr<AmgXSolver> this_sol{std::make_unique<AmgXSolver>()};
        this_sol->DefaultParameters(AmgXSolver::SOLVER, inp.verbose);
        this_sol->ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);

        if (inp.ndevices > 0)
          this_sol->InitMPITeams(MPI_COMM_WORLD, inp.ndevices);
        else
          this_sol->InitExclusiveGPU(MPI_COMM_WORLD);
  

        this_sol->SetOperator(*Dmat[g]);
        Dsol[g] = std::move(this_sol);

        // Dmat no longer needs to be stored on cpu
        Dmat[g].reset();

#else

        MFEM_ABORT("solver not supported without amgx");

#endif

      }
      else
      {

        std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*Dmat[g])};
        amg->SetPrintLevel(inp.hypre_print_level());
        Dprec[g] = std::move(amg);

        std::unique_ptr<HyprePCG> this_sol{std::make_unique<HyprePCG>(MPI_COMM_WORLD)};
        this_sol->SetPreconditioner(*Dprec[g]);
        this_sol->SetOperator(*Dmat[g]);
        this_sol->SetPrintLevel(inp.hypre_print_level());
        this_sol->SetTol(inp.tol.inner_rel);
        this_sol->SetAbsTol(inp.tol.inner_abs);
        this_sol->SetMaxIter(inp.tol.max_inner);
        Dsol[g] = std::move(this_sol);

      }

    } // g=1,mat_lib.ngroup()

  } // DSA operator

  time.stop(time.matrix_assembly);

  const int integration_order{scalar_flux_fespace->GetOrder(0)+1};
  QuadratureSpace qspace{scalar_flux_fespace->GetParMesh(), integration_order};

  std::vector<QuadratureFunction> fiss_upscatter_quad, saaf_fiss_upscatter_quad;
  fiss_upscatter_quad.reserve(mat_lib.ngroup());
  saaf_fiss_upscatter_quad.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_upscatter_quad.emplace_back(&qspace, quad->nd());
    saaf_fiss_upscatter_quad.emplace_back(&qspace, quad->dim()*quad->nd());
  }

  QuadratureFunction dnscatter_quad{&qspace, quad->nd()};
  QuadratureFunction saaf_dnscatter_quad{&qspace, quad->dim()*quad->nd()};

  QuadratureFunction selfscatter_quad{&qspace, quad->nd()};
  QuadratureFunction saaf_selfscatter_quad{&qspace, quad->dim()*quad->nd()};

  std::vector<VectorQuadratureFunctionCoefficient> src_quad_func, saaf_src_quad_func;
  src_quad_func.reserve(mat_lib.ngroup());
  saaf_src_quad_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    src_quad_func.emplace_back(selfscatter_quad);
    saaf_src_quad_func.emplace_back(saaf_selfscatter_quad);
  }

  std::vector<std::unique_ptr<ParLinearForm>> lf_self_scatter(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    lf_self_scatter[g] = std::make_unique<ParLinearForm>(angular_flux_fespace);
    lf_self_scatter[g]->AddDomainIntegrator(new VectorDomainLFIntegrator{src_quad_func[g]});
    lf_self_scatter[g]->AddDomainIntegrator(new VectorDomainLFGradIntegrator{saaf_src_quad_func[g]});
  }

////////////////////////////////////////////////////////////////////////////////
  // DSA rhs

  std::vector<GridFunctionCoefficient> scalar_flux_func; // (k+1/2)
  scalar_flux_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux_func.emplace_back(&scalar_flux(g));

  ParGridFunction dsa_phi_old{scalar_flux_fespace};    // (k)
  GridFunctionCoefficient dsa_phi_old_func{&dsa_phi_old};

  std::vector<SumCoefficient> dsa_flux_func;
  dsa_flux_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    dsa_flux_func.emplace_back(scalar_flux_func[g], dsa_phi_old_func, 1.0, -1.0);

  std::vector<PWConstCoefficient> self_scatter_func;
  self_scatter_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector self_scatter(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      self_scatter(i) = mat_lib.mat[i].rec.sigma_s.Elem(g,g);
    self_scatter_func.emplace_back(self_scatter);
  }

  std::vector<ProductCoefficient> dsa_src;
  dsa_src.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    dsa_src.emplace_back(self_scatter_func[g], dsa_flux_func[g]);

  std::vector<std::unique_ptr<ParLinearForm>> lf_dsa(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    lf_dsa[g] = std::make_unique<ParLinearForm>(scalar_flux_fespace);
    lf_dsa[g]->AddDomainIntegrator(new DomainLFIntegrator{dsa_src[g]});
  }

////////////////////////////////////////////////////////////////////////////////

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    angular_flux(g) = 1.0;
    scalar_flux(g) = 1.0;
  }
  angular_flux.keff = 1.0;
  scalar_flux.keff = 1.0;

  QuadratureFunction fiss_src_quad_old{&qspace};
  fiss_src_quad_old = 1.0;
  double k_conv{inp.tol.k * 1e1 + 1e-6};
  double phi_conv{inp.tol.phi * 1e1 + 1e-6};
  double keff_old{angular_flux.keff};
  double fiss_sum_old{1.0};
  int solve_counter{0};

  time.start(time.solver);
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    time.start("power");
    calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
    time.stop("power");

    time.start("temperature");
    temperature_heat_conduction(inp, mat_lib, power, temperature);
    time.stop("temperature");

    time.start("calc_fiss_upscatter_vector");
    calc_fiss_upscatter_vector(scalar_flux, mat_lib, quad,
      fiss_upscatter_quad, saaf_fiss_upscatter_quad,
      angular_flux.keff, inp.flux_free_chi);
    time.stop("calc_fiss_upscatter_vector");

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      time.start("calc_downscatter_vector");
      calc_downscatter_vector(g, scalar_flux, mat_lib, quad,
        dnscatter_quad, saaf_dnscatter_quad);
      dnscatter_quad += fiss_upscatter_quad[g];
      saaf_dnscatter_quad += saaf_fiss_upscatter_quad[g];
      time.stop("calc_downscatter_vector");

      double scattering_conv{inp.tol.scatter * 1e1 + 1e-6};
      for (int scattering_iter{0}; (scattering_iter < inp.tol.max_scatter) &&
        (scattering_conv > inp.tol.scatter); ++scattering_iter)
      {

        dsa_phi_old = scalar_flux(g); // (k)

        time.start("selfscatter_and_assemble");
        calc_selfscatter_vector(g, scalar_flux(g), mat_lib, quad,
          selfscatter_quad, saaf_selfscatter_quad);
        selfscatter_quad += dnscatter_quad;
        saaf_selfscatter_quad += saaf_dnscatter_quad;
        lf_self_scatter[g]->Assemble();
        std::unique_ptr<HypreParVector> b{lf_self_scatter[g]->ParallelAssemble()};
        time.stop("selfscatter_and_assemble");


        // Maybe try FormLinearSystem?

        time.start("Amult");
        std::unique_ptr<HypreParVector> x{angular_flux(g).ParallelAssemble()};
        Asol[g]->Mult(*b, *x);
        angular_flux(g).Distribute(*x);
        time.stop("Amult");

        time.start("scalar_flux");
        calc_scalar_flux_vector(g, angular_flux, quad, scalar_flux); // (k+1/2)
        time.start("scalar_flux");

        if (use_dsa)
        {

          time.start("Dmult");

          ParGridFunction pgf_dphi{scalar_flux(g)};

          lf_dsa[g]->Assemble();
          std::unique_ptr<HypreParVector> f{lf_dsa[g]->ParallelAssemble()};
          std::unique_ptr<HypreParVector> dphi{pgf_dphi.ParallelAssemble()};
          Dsol[g]->Mult(*f, *dphi);
          pgf_dphi.Distribute(*dphi);

          scalar_flux(g) += pgf_dphi;

          time.stop("Dmult");

        }

        ++solve_counter;

        dsa_phi_old -= scalar_flux(g);
        const double local_scattering_conv{dsa_phi_old.Normlinf() / scalar_flux(g).Normlinf()};
        MPI_Allreduce(&local_scattering_conv, &scattering_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (myid == 0)
          std::cout << "..."
                    << scattering_iter
                    << " g=" << g
                    << " conv=" << scattering_conv
                    << '\n';

      } // scattering_iter
    } // ngroup

    time.start("fisssum");
    QuadratureFunction fiss_src_quad{&qspace};
    calc_fiss_src(scalar_flux, mat_lib, fiss_src_quad);
    const double local_fiss_sum{fiss_src_quad.Sum()};
    double fiss_sum{0.0};
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time.stop("fisssum");

    time.start("phiconv");
    fiss_src_quad_old -= fiss_src_quad;
    
    // note: some nodes may contain zero fissile elements depending on decomposition
    const double delta{fiss_src_quad_old.Normlinf()};
    const double norm{fiss_src_quad.Normlinf()};
    const double local_phi_conv{(norm > 0.0) ? delta / norm : 0.0};

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

    if ((angular_flux.keff < 0.0) ||
        (angular_flux.keff > 1e50) ||
        std::isnan(angular_flux.keff))
    {
      if (myid == 0) std::cout << "INVALID VALUE FOR KEFF" << std::endl;
      return;
    }

    scalar_flux.keff = angular_flux.keff;
    // store old values
    keff_old = angular_flux.keff;
    fiss_sum_old = fiss_sum;
    fiss_src_quad_old = fiss_src_quad;

  }
  time.stop(time.solver);

  if (myid == 0)
  {
    std::cout << "ITERATIONS FINISHED!!!" << std::endl;
    std::cout << "solve_counter= " << solve_counter << std::endl;
  }

  return;
}

void saaf_angle_jacobi_power_iteration(const Quadrature* const quad,
                                       Timer& time,
                                       const Input& inp,
                                       const Material_library& mat_lib,
                                       Result& angular_flux,
                                       Result& scalar_flux)
{

  ParFiniteElementSpace *angular_flux_fespace = angular_flux(0).ParFESpace();
  ParFiniteElementSpace *scalar_flux_fespace = scalar_flux(0).ParFESpace();
  ParMesh *pmesh = angular_flux_fespace->GetParMesh();

  const int myid{angular_flux_fespace->GetMyRank()};

  constexpr int skip_zeros{2};
  constexpr bool use_dsa{true};

  std::unique_ptr<HypreParMatrix> A;
  std::unique_ptr<Solver> Asol;
  std::unique_ptr<HypreSolver> Aprec;
  
  std::unique_ptr<HypreParMatrix> D;
  std::unique_ptr<Solver> Dsol;
  std::unique_ptr<HypreSolver> Dprec;

  { // A operator

    std::vector<DenseMatrix> omega_mat(quad->nd());
    for (int n{0}; n < quad->nd(); ++n)
    {
      omega_mat[n].SetSize(quad->dim());
      MultVVt(quad->omega(n), omega_mat[n]);
    }

    Tensor<Vector> omega;
    omega.reserve(quad->nd()*mat_lib.ngroup());
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      for (int n{0}; n < quad->nd(); ++n)
        omega.AddElement(quad->omega(n), n+g*quad->nd());
    VectorTensorConstantCoefficient omega_func{quad->nd()*mat_lib.ngroup(), omega};

    std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      leakage[i].reserve(quad->nd()*mat_lib.ngroup());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int idx{n+g*quad->nd()};
          omega_mat[n] *= (1.0/tot);
          leakage[i].AddElement(omega_mat[n], idx);
          omega_mat[n] *= tot;
        }
      }
    }
    MatrixTensorPWConstCoefficient leakage_func{quad->nd()*mat_lib.ngroup(), leakage};

    std::vector<Vector> total(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      total[i].SetSize(quad->nd()*mat_lib.ngroup());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int idx{n+g*quad->nd()};
          total[i](idx) = tot;
        }
      }
    }
    VectorPWConstCoefficient total_func{total};

    ParBilinearForm a{angular_flux_fespace};
    a.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
    a.AddDomainIntegrator(new VectorMassIntegrator{total_func});
    a.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});

    a.Assemble(skip_zeros);
    a.Finalize(skip_zeros);
    A.reset(a.ParallelAssemble());


    std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*A)};
    amg->SetPrintLevel(inp.hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib.ngroup()*quad->nd());
    Aprec = std::move(amg);

    std::unique_ptr<HyprePCG> cg{std::make_unique<HyprePCG>(*A)};
    cg->SetPrintLevel(inp.hypre_print_level());
    cg->SetTol(inp.tol.inner_rel);
    cg->SetAbsTol(inp.tol.inner_abs);
    cg->SetMaxIter(inp.tol.max_inner);
    cg->SetPreconditioner(*Aprec);

    Asol = std::move(cg);

  } // A operator

  { // DSA operator

    std::vector<Vector> diffusive(pmesh->attributes.Max());
    std::vector<Vector> removal(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      diffusive[i].SetSize(mat_lib.ngroup());
      removal[i].SetSize(mat_lib.ngroup());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        diffusive[i](g) = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_t[g]);
        removal[i](g) = mat_lib.mat[i].rec.sigma_t[g] - mat_lib.mat[i].rec.sigma_s.Elem(g,g);
      }
    }
    VectorPWConstCoefficient diffusive_func{diffusive};
    VectorPWConstCoefficient removal_func{removal};

    ParBilinearForm d{scalar_flux_fespace};
    d.AddDomainIntegrator(new VectorDiffusionIntegrator{diffusive_func});
    d.AddDomainIntegrator(new VectorMassIntegrator{removal_func});

    constexpr int diffusion_skip_zeros{1};
    d.Assemble(diffusion_skip_zeros);
    // all bcs essential to match the scalar flux
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 1;
    d.EliminateEssentialBCDiag(ess_bdr, 1.0);

    d.Finalize(diffusion_skip_zeros);

    D.reset(d.ParallelAssemble());

    std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*D)};
    amg->SetPrintLevel(inp.hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib.ngroup());
    Dprec = std::move(amg);

    std::unique_ptr<HyprePCG> cg{std::make_unique<HyprePCG>(*D)};
    cg->SetPrintLevel(inp.hypre_print_level());
    cg->SetTol(inp.tol.inner_rel);
    cg->SetAbsTol(inp.tol.inner_abs);
    cg->SetMaxIter(inp.tol.max_inner);
    cg->SetPreconditioner(*Dprec);

    Dsol = std::move(cg);


  } // DSA operator

  const int integration_order{angular_flux_fespace->GetOrder(0)+1};
  QuadratureSpace qspace{pmesh, integration_order};

  QuadratureFunction src_quad{&qspace, quad->nd()*mat_lib.ngroup()};
  QuadratureFunction saaf_src_quad{&qspace, quad->dim()*quad->nd()*mat_lib.ngroup()};

  QuadratureFunction self_scatter_quad{&qspace, quad->nd()*mat_lib.ngroup()};
  VectorQuadratureFunctionCoefficient quad_func{self_scatter_quad};

  QuadratureFunction saaf_self_scatter_quad{&qspace, quad->dim()*quad->nd()*mat_lib.ngroup()};
  VectorQuadratureFunctionCoefficient saaf_quad_func{saaf_self_scatter_quad};

  ParLinearForm b{angular_flux_fespace};
  b.AddDomainIntegrator(new VectorDomainLFIntegrator{quad_func});
  b.AddDomainIntegrator(new VectorDomainLFGradIntegrator{saaf_quad_func});

  std::vector<Vector> self_scatter_vec(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    self_scatter_vec[i].SetSize(mat_lib.ngroup());
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      self_scatter_vec[i](g) = mat_lib.mat[i].rec.sigma_s.Elem(g,g);
  }
  VectorPWConstCoefficient self_scatter_func{self_scatter_vec};

  VectorGridFunctionCoefficient scalar_flux_func{&scalar_flux(0)};

  ParGridFunction dsa_phi_old{scalar_flux_fespace};
  VectorGridFunctionCoefficient dsa_phi_old_func{&dsa_phi_old};

  VectorSumCoefficient dsa_flux_func{scalar_flux_func, dsa_phi_old_func, 1.0, -1.0};
  VectorProductCoefficient dsa_src_func{self_scatter_func, dsa_flux_func};

  ParLinearForm dsa_rhs{scalar_flux_fespace};
  dsa_rhs.AddDomainIntegrator(new VectorDomainLFIntegrator{dsa_src_func});

  QuadratureFunction fiss_src_quad{&qspace};

  scalar_flux(0) = 1.0;
  angular_flux(0) = 1.0;
  scalar_flux.keff = 1.0;
  angular_flux.keff = 1.0;

  int solve_counter{0};

  double keff_old{1.0};
  double fiss_sum_old{1.0};

  double k_conv{inp.tol.k*1e2+1.0};
  double phi_conv{inp.tol.phi*1e2+1.0};
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    calc_src(mat_lib, quad, scalar_flux(0), angular_flux.keff, src_quad, saaf_src_quad);


    double scattering_conv{inp.tol.scatter * 1e1 + 1.0};
    for (int scattering_iter{0}; (scattering_iter < inp.tol.max_scatter) &&
      (scattering_conv > inp.tol.scatter); ++scattering_iter)
    {

      dsa_phi_old = scalar_flux(0);

      calc_self_scatter(mat_lib, quad, scalar_flux(0), self_scatter_quad, saaf_self_scatter_quad);
      self_scatter_quad += src_quad;
      saaf_self_scatter_quad += saaf_src_quad;
      b.Assemble();
      std::unique_ptr<HypreParVector> bvec{b.ParallelAssemble()};

      std::unique_ptr<HypreParVector> xvec{angular_flux(0).ParallelAssemble()};
      Asol->Mult(*bvec, *xvec);
      angular_flux(0).Distribute(*xvec);

      calc_scalar_flux_vector_vector(angular_flux(0), mat_lib, quad, scalar_flux(0));

      if (use_dsa)
      {

        dsa_rhs.Assemble();
        std::unique_ptr<HypreParVector> dsa_vec{dsa_rhs.ParallelAssemble()};
        std::unique_ptr<HypreParVector> dphi{scalar_flux(0).ParallelAssemble()};
        Dsol->Mult(*dsa_vec, *dphi);

        ParGridFunction pgf_dphi{scalar_flux_fespace};
        pgf_dphi.Distribute(*dphi);

        scalar_flux(0) += pgf_dphi;

      }

      ++solve_counter;

      dsa_phi_old -= scalar_flux(0);
      const double local_scattering_conv{dsa_phi_old.Normlinf() / scalar_flux(0).Normlinf()};
      MPI_Allreduce(&local_scattering_conv, &scattering_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if (myid == 0)
        std::cout << "..."
                  << scattering_iter
                  << " conv=" << scattering_conv
                  << '\n';

    }

    QuadratureFunction fiss_src_quad_old{fiss_src_quad};
    calc_fiss_src_vector(scalar_flux(0), mat_lib, fiss_src_quad);
    const double local_fiss_sum{fiss_src_quad.Sum()};
    double fiss_sum{0.0};
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    fiss_src_quad_old -= fiss_src_quad;

    // note: some nodes may contain zero fissile elements depending on decomposition
    const double delta{fiss_src_quad_old.Normlinf()};
    const double norm{fiss_src_quad.Normlinf()};
    const double local_phi_conv{(norm > 0.0) ? delta / norm : 0.0};

    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (iter > 0) angular_flux.keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(angular_flux.keff - keff_old);

    if (myid == 0)
      std::cout << iter 
        << " keff=" << angular_flux.keff
        << " dk=" << k_conv 
        << " dphi=" << phi_conv 
        << std::endl;

    scalar_flux.keff = angular_flux.keff;
    keff_old = angular_flux.keff;
    fiss_sum_old = fiss_sum;

  }

  if (myid == 0)
  {
    std::cout << "ITERATIONS FINISHED!!!" << std::endl;
    std::cout << "solve_counter= " << solve_counter << std::endl;
  }

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

  ParFiniteElementSpace* fespace{angular_flux(g).ParFESpace()};

  MFEM_VERIFY(fespace->GetOrdering() == Ordering::byVDIM,
    "scalar flux calculation relies on angular flux being ordered by vdim");

  const int ndofs{fespace->GetNDofs()};
  const int vdim{quad->nd()};

  scalar_flux(g) = 0.0;

  for (int i{0}; i < ndofs; ++i)
    for (int n{0}; n < quad->nd(); ++n)
      scalar_flux(g)(i) += angular_flux(g)(n+i*vdim) * quad->weight(n);

  return;
}

void calc_fiss_upscatter(const Result& scalar_flux, 
                         const Material_library& mat_lib, 
                         std::vector<QuadratureFunction>& fiss_quad_func, 
                         std::vector<QuadratureFunction>& upscatter_quad_func,
                         const bool flux_free_chi)
{
  const QuadratureSpace* const qspace{fiss_quad_func[0].GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

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
        if (rec.is_upscatter[g])
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
        } // is_upscatter
      } // for mat_lib.ngroup()

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
  const ParFiniteElementSpace* const fespace{scalar_flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

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


void calc_selfscatter(const int g, 
                      const ParGridFunction &scalar_flux,
                      const Material_library& mat_lib,
                      QuadratureFunction& selfscatter_quad_func)
{
  selfscatter_quad_func = 0.0;

  const QuadratureSpace* const qspace{selfscatter_quad_func.GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux.ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  int idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const SparseMatrix& scat{rec.sigma_s};

    const double scatxs{scat.Elem(g,g)};

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {
      const IntegrationPoint ip{ir.IntPoint(i)};
      const double flux{scalar_flux.GetValue(e, ip)};
      selfscatter_quad_func(idx) += flux*scatxs;
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
  const ParFiniteElementSpace* const fespace{flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};
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
  const ParFiniteElementSpace* const fespace{flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};
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
  const ParFiniteElementSpace* const fespace{scalar_flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  fiss_src = 0.0;

  int idx{0};
  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};

    if (!rec.is_fiss)
    {
      idx += ir.GetNPoints();
      continue;
    }

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
                                   const Material_library& mat_lib,
                                   Result& scalar_flux)
{

  ParFiniteElementSpace *angular_flux_fespace = angular_flux(0).ParFESpace();

  // spatial dofs (not including vector DOFS)
  const int ndofs{angular_flux_fespace->GetNDofs()};
  const int ngroup{mat_lib.ngroup()};

  scalar_flux.keff = angular_flux.keff;

  for (int g{0}; g < ngroup; ++g)
    scalar_flux(g) = 0.0;

  for (int i{0}; i < ndofs; ++i)
  {
    for (int g{0}; g < ngroup; ++g)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        scalar_flux(g)(i) += 
          angular_flux(0)(n+g*quad->nd()+i*quad->nd()*ngroup) * quad->weight(n);
      }
    }
  }

  /*
  const int ngroup{angular_flux(0).ParFESpace()->GetVDim() / quad->nd()};
  ParGridFunction tmp{scalar_flux(0)};
  for (int g{0}; g < ngroup; ++g)
  {
    scalar_flux(g) = 0.0;
    for (int n{0}; n < quad->nd(); ++n)
    {
      const int idx{n + g*quad->nd()};
      GridFunctionCoefficient gfc{&angular_flux(0), idx+1};
      ConstantCoefficient weight{quad->weight(n)};
      ProductCoefficient coeff{weight, gfc};
      tmp.ProjectCoefficient(coeff);
      scalar_flux(g) += tmp;
    }
  }
  */

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

void power_iteration_amgx(HypreParMatrix* A, HypreParMatrix* M, const Input& inp, Timer& time, Result& res, int numfunctions)
{

  const int myid{res(0).ParFESpace()->GetMyRank()};

  constexpr bool use_amgx{false};
  constexpr bool compute_residual{false};

  std::unique_ptr<Solver> sol, prec;

  if (use_amgx)
  {

#if AMGX_AVAILABLE

    if (myid == 0) std::cout << "AMGX_SETUP_BEGIN" << std::endl;
    time.start("amgx_setup");
    std::unique_ptr<AmgXSolver> Aamgx{std::make_unique<AmgXSolver>()};
    Aamgx->DefaultParameters(AmgXSolver::SOLVER, inp.verbose);
    Aamgx->ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);

    if (myid == 0) std::cout << "AMGX_MPI_SETTING_BEGIN" << std::endl;
    if (inp.ndevices > 0)
      Aamgx->InitMPITeams(MPI_COMM_WORLD, inp.ndevices); // use MPITeams
    else
      Aamgx->InitExclusiveGPU(MPI_COMM_WORLD); // use MPI exclusive
    if (myid == 0) std::cout << "AMGX_MPI_SETTING_END" << std::endl;

    if (myid == 0) std::cout << "AMGX_SET_OPERATOR_BEGIN" << std::endl;
    Aamgx->SetOperator(*A);
    if (!compute_residual)
    {
      delete A;
      A = nullptr;
    }
    if (myid == 0) std::cout << "AMGX_SET_OPERATOR_END" << std::endl;

    time.stop("amgx_setup");
    if (myid == 0) std::cout << "AMGX_SETUP_END" << std::endl;

    sol = std::move(Aamgx);

#else

    MFEM_ABORT("solver not supported without amgx");

#endif

  }
  else
  {

    std::unique_ptr<HypreGMRES> gmres{std::make_unique<HypreGMRES>(*A)};
    gmres->SetPrintLevel((inp.verbose) ? 9 : 0);
    gmres->SetMaxIter(inp.tol.max_inner);
    gmres->SetTol(inp.tol.inner_rel);
    gmres->SetAbsTol(inp.tol.inner_abs);
    gmres->SetKDim(50); // default, should come from user input

    std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*A)};
    amg->SetPrintLevel(inp.hypre_print_level());
    if (numfunctions > 1)
      HYPRE_BoomerAMGSetNumFunctions(*amg, numfunctions);
    gmres->SetPreconditioner(*amg);

    sol = std::move(gmres);
    prec = std::move(amg);

  }

  if (myid == 0) std::cout << "RANDOMIZE_BEGIN" << std::endl;
  HypreParVector x{*M};
  x.UseDevice(true);
  x.Randomize(730); // if default seed is used, the seed is actually the time and we lose reproducability
  if (myid == 0) std::cout << "RANDOMIZE_END" << std::endl;

  double k_conv{inp.tol.k*1e2+1.0};
  double x_conv{inp.tol.phi*1e2+1.0};

  if (myid == 0) std::cout << "ITERATION_BEGIN" << std::endl;
  double k{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer) && 
      ((k_conv > inp.tol.k) || (x_conv > inp.tol.phi)); ++iter)
  {

    if (myid == 0) std::cout << "COPY_BEGIN" << std::endl;
    time.start("old_copy");
    const double k_old{k};
    HypreParVector x_old{x}; // constructor doesn't perform deep copy
    x_old.UseDevice(true);
    x_old = x;
    time.stop("old_copy");
    if (myid == 0) std::cout << "COPY_END" << std::endl;

    if (myid == 0) std::cout << "SCALE_BEGIN" << std::endl;
    time.start("calc_v");
    // v = x / \| x \|
    HypreParVector v{x}; // constructor doesn't perform deep copy
    v.UseDevice(true);
    v = x;
    v /= par_norml2(x);
    time.stop("calc_v");
    if (myid == 0) std::cout << "SCALE_END" << std::endl;

    if (myid == 0) std::cout << "MATVEC_BEGIN" << std::endl;
    time.start("matvec");
    // b = M*v
    HypreParVector b{v}; // constructor doesn't perform deep copy
    b.UseDevice(true);
    //Mhypre->Mult(v, b); // this is performed on CPU
    M->Mult(v, b, 1.0, 0.0); // make sure we use the proper function signature
    time.stop("matvec");
    if (myid == 0) std::cout << "MATVEC_END" << std::endl;

    if (myid == 0) std::cout << "AMGX_SOLVE_BEGIN" << std::endl;
    time.start("amgx_solve");
    // x = A^-1 * b
    sol->Mult(b, x);
    time.stop("amgx_solve");
    if (myid == 0) std::cout << "AMGX_SOLVE_END" << std::endl;

    if (myid == 0) std::cout << "DOT_BEGIN" << std::endl;
    time.start("dot_product");
    // k = v^T * x
    const double local_k{v*x};
    MPI_Allreduce(&local_k, &k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time.stop("dot_product");
    if (myid == 0)  std::cout << "DOT_END" << std::endl;

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
      A->Mult(x, r0);

      HypreParVector r1{x};
      r1.UseDevice(true);
      M->Mult(x, r1);
      r1 /= k;

      r0 -= r1;
      std::cout << "residual=" << par_norml2(r0) << std::endl;
      time.stop("residual");
    }

    if (myid == 0) std::cout << "iter=" << iter
                             << " k=" << k 
                             << " dk=" << k_conv 
                             << " dx=" << x_conv 
                             << std::endl;

  }
  if (myid == 0) std::cout << "ITERATIONS_FINISHED!!!" << std::endl;

  // cleanup
  delete A;
  delete M;

  // store solution
  res.keff = k;
  time.start("distribute");
  res(0).Distribute(x);
  time.stop("distribute");
  flux_sign_fixup(res(0));

  return;
}

void flux_sign_fixup(Vector& pgf)
{
  const double local_max{pgf.Max()};
  double max;
  MPI_Allreduce(&local_max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // note that normlinf is not on the GPU
  const double local_normlinf{pgf.Normlinf()};
  double normlinf;
  MPI_Allreduce(&local_normlinf, &normlinf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if ((max < 0.0) || (max < local_normlinf))
    pgf *= -1.0;

  return;
}

HypreParMatrix* spn_a_matrix(const Input &inp,
                             const Material_library &mat_lib,
                             ParFiniteElementSpace* fespace,
                             const Array<int>& ess_bdr)
{

  const ParMesh* pmesh{fespace->GetParMesh()};

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

  bool all_essential{true};
  for (int i{0}; (i < ess_bdr.Size()) && (all_essential); ++i)
    all_essential = (ess_bdr[i] == 0);
  const int skip_zeros{(all_essential) ? 2 : 1};
  diffusion_bilinear.Assemble(skip_zeros);
  diffusion_bilinear.EliminateEssentialBCDiag(ess_bdr, 1.0);
  diffusion_bilinear.Finalize(skip_zeros);

  return diffusion_bilinear.ParallelAssemble();
}

mfem::HypreParMatrix* spn_m_matrix(const Input &inp, 
                                   const Material_library &mat_lib,
                                   ParFiniteElementSpace* fespace,
                                   const Array<int>& ess_bdr)
{

  const ParMesh* pmesh{fespace->GetParMesh()};

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

  bool all_essential{true};
  for (int i{0}; (i < ess_bdr.Size()) && (all_essential); ++i)
    all_essential = (ess_bdr[i] == 0);
  const int skip_zeros{(all_essential) ? 2 : 1};
  fission_bilinear.Assemble(skip_zeros);
  fission_bilinear.EliminateEssentialBCDiag(ess_bdr, std::numeric_limits<double>::min());
  fission_bilinear.Finalize(skip_zeros);

  return fission_bilinear.ParallelAssemble();

}

namespace slepc_amgx
{

  std::unique_ptr<Solver> sol{nullptr};
  HypreParMatrix *Ahypre{nullptr};

  bool verbose{true};
  int ndevices{0};

  PetscErrorCode setup(PC pc)
  {

    std::cout << "PC_SHELL_SETUP" << std::endl;

#ifdef AMGX_AVAILABLE

    std::unique_ptr<AmgXSolver> amgx{std::make_unique<AmgXSolver>()};
    amgx->DefaultParameters(AmgXSolver::PRECONDITIONER, verbose);
    amgx->ReadParameters("amgxrc.json", AmgXSolver::EXTERNAL);
    MFEM_VERIFY(ndevices == 0, "petsc+slepc+amgx link only works for MPIExclusive mode.");
    amgx->InitExclusiveGPU(MPI_COMM_WORLD);
    amgx->SetOperator(*Ahypre);
    sol = std::move(amgx);

    delete Ahypre;
    Ahypre = nullptr;

#else

    MFEM_ABORT("solver not supported without amgx");

#endif
    
    return PetscErrorCode{0};
  }

  PetscErrorCode apply(PC pc, Vec xin, Vec xout)
  {
    const PetscParVector vin{xin, true};
    PetscParVector vout{xout, true};
    sol->Mult(vin, vout);
    return PetscErrorCode{0};
  }


  PetscErrorCode destroy(PC pc)
  {
    std::cout << "PC_SHELL_DESTROY" << std::endl;
    sol.reset();
    return PetscErrorCode{0};
  }

} // namespace slepc_amgx

namespace slepc_hypre
{

  std::unique_ptr<Solver> sol{nullptr};
  std::unique_ptr<HypreSolver> prec{nullptr};
  const HypreParMatrix *Ahypre{nullptr};

  const Input *inp{nullptr};
  const Material_library *mat_lib{nullptr};
  const Quadrature *quad{nullptr};

  PetscErrorCode setup(PC pc)
  {

    std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*Ahypre)};
    amg->SetPrintLevel(inp->hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib->ngroup()*quad->nd());
    HYPRE_BoomerAMGSetAggNumLevels(*amg, 2);
    prec = std::move(amg);

    /*
    std::unique_ptr<HypreEuclid> euclid{std::make_unique<HypreEuclid>(*Ahypre)};
    HYPRE_EuclidSetLevel(*euclid, 4);
    prec = std::move(euclid);
    */

    std::unique_ptr<HypreGMRES> gmres{std::make_unique<HypreGMRES>(*Ahypre)};
    gmres->SetPreconditioner(*prec);
    gmres->SetPrintLevel(9);
    gmres->SetMaxIter(inp->tol.max_inner);
    gmres->SetTol(inp->tol.inner_rel);
    gmres->SetAbsTol(inp->tol.inner_abs);
    gmres->SetKDim(50); // Default...
    sol = std::move(gmres);

    return 0;
  }

  PetscErrorCode apply(PC pc, Vec xin, Vec xout)
  {
    const PetscParVector vin{xin, true};
    PetscParVector vout{xout, true};
    sol->Mult(vin, vout);
    return 0;
  }

  PetscErrorCode destroy(PC pc)
  {
    sol.reset();
    prec.reset();
    delete Ahypre;
    Ahypre = nullptr;
    return 0;
  }

} // namespace slepc_hypre

void slepc_solver_wrapper(HypreParMatrix* Ahypre,
                          HypreParMatrix* Mhypre,
                          const Input& inp,
                          Timer& time,
                          Result& res,
                          int numfunctions)
{

  constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};

  const PetscParMatrix A{Ahypre, petsc_format};
  const PetscParMatrix M{Mhypre, petsc_format};

  /* This seems incompatible with the GMRES setup...
  if (numfunctions > 1)
  {
    PetscErrorCode ierr;
    ierr = MatSetBlockSize(A, numfunctions); PCHKERRQ(A, ierr);
    ierr = MatSetBlockSize(M, numfunctions); PCHKERRQ(M, ierr);
  }
  */

  constexpr bool slepc_amgx{false};
  constexpr bool slepc_ksp{false};
  constexpr bool slepc_hypre{false};

  if ((petsc_format != Operator::PETSC_MATHYPRE) && 
      (petsc_format != Operator::PETSC_MATSHELL))
  {
    if ((!slepc_amgx) && (!slepc_hypre))
    {
      delete Ahypre;
      Ahypre = nullptr;
    }
    delete Mhypre;
    Mhypre = nullptr;
  }

  SlepcEigenSolver slepc{MPI_COMM_WORLD};
  slepc.SetNumModes(1); // only fundamental mode
  slepc.SetTol(std::min(inp.tol.phi, inp.tol.k));
  slepc.SetMaxIter(inp.tol.max_outer);
  slepc.SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);
  slepc.SetOperators(A, M);
  
  PetscErrorCode ierr;
  PC preconditioner;
  if (slepc_amgx)
  {

    ST spectral_transform;
    ierr = EPSGetST(slepc, &spectral_transform); PCHKERRQ(slepc, ierr);

    KSP petsc_solver;
    ierr = STGetKSP(spectral_transform, &petsc_solver); PCHKERRQ(slepc, ierr);

    ierr = PCCreate(MPI_COMM_WORLD, &preconditioner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner, PCSHELL); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetSetUp(preconditioner, slepc_amgx::setup); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetApply(preconditioner, slepc_amgx::apply); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetDestroy(preconditioner, slepc_amgx::destroy); PCHKERRQ(slepc, ierr);
    slepc_amgx::Ahypre = Ahypre;
    slepc_amgx::verbose = inp.verbose;
    slepc_amgx::ndevices = inp.ndevices;
    Ahypre = nullptr;

    ierr = KSPSetPC(petsc_solver, preconditioner); PCHKERRQ(slepc, ierr);
  }
  else if (slepc_ksp)
  {

    ST spectral_transform;
    ierr = EPSGetST(slepc, &spectral_transform); PCHKERRQ(slepc, ierr);

    KSP petsc_solver;
    ierr = STGetKSP(spectral_transform, &petsc_solver); PCHKERRQ(slepc, ierr);

    ierr = PCCreate(MPI_COMM_WORLD, &preconditioner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner, PCKSP); PCHKERRQ(slepc, ierr);

    KSP petsc_solver_inner;
    ierr = PCKSPGetKSP(preconditioner, &petsc_solver_inner); PCHKERRQ(slepc, ierr);
    ierr = KSPSetType(petsc_solver_inner, KSPGMRES); PCHKERRQ(slepc, ierr);
    ierr = KSPGMRESSetRestart(petsc_solver_inner, inp.tol.max_inner); PCHKERRQ(slepc, ierr); // (no restarting rn)
    ierr = KSPSetTolerances(petsc_solver_inner, inp.tol.inner_rel, inp.tol.inner_abs, 1e6, inp.tol.max_inner); PCHKERRQ(slepc, ierr);

    PC preconditioner_inner;
    ierr = KSPGetPC(petsc_solver_inner, &preconditioner_inner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner_inner, PCGAMG); PCHKERRQ(slepc, ierr);
    ierr = PCGAMGSetSymGraph(preconditioner_inner, PETSC_TRUE); PCHKERRQ(slepc, ierr);

    ierr = KSPSetPC(petsc_solver, preconditioner); PCHKERRQ(slepc, ierr);
  }
  else if (slepc_hypre)
  {

    ST spectral_transform;
    ierr = EPSGetST(slepc, &spectral_transform); PCHKERRQ(slepc, ierr);

    KSP petsc_solver;
    ierr = STGetKSP(spectral_transform, &petsc_solver); PCHKERRQ(slepc, ierr);

    ierr = PCCreate(MPI_COMM_WORLD, &preconditioner); PCHKERRQ(slepc, ierr);
    ierr = PCSetType(preconditioner, PCSHELL); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetSetUp(preconditioner, slepc_hypre::setup); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetApply(preconditioner, slepc_hypre::apply); PCHKERRQ(slepc, ierr);
    ierr = PCShellSetDestroy(preconditioner, slepc_hypre::destroy); PCHKERRQ(slepc, ierr);

    slepc_hypre::Ahypre = Ahypre;
    Ahypre = nullptr;

    ierr = KSPSetPC(petsc_solver, preconditioner); PCHKERRQ(slepc, ierr);

  }

  /* RESTART
  // set based on the previous state
  Vector true_dof;
  res(0).GetTrueDofs(true_dof);
  const PetscParVector tv{MPI_COMM_WORLD, true_dof, true};
  const std::unique_ptr<Vec[]> array{std::make_unique<Vec[]>(1)};
  array[0] = tv;
  EPSSetInitialSpace(slepc, 1, array.get());
  */

  time.start(time.solver);
  slepc.Solve();
  time.stop(time.solver);

  if ((slepc_amgx) || (slepc_hypre))
  {
    ierr = PCDestroy(&preconditioner); PCHKERRQ(slepc, ierr);
  }

  delete Ahypre;
  delete Mhypre;

  if (slepc.GetNumConverged() >= 1)
  {
    slepc.GetEigenvalue(0, res.keff);
    res.keff = 1.0 / res.keff;
    Vector tmp(res(0).ParFESpace()->GetTrueVSize());
    slepc.GetEigenvector(0, tmp);
    res(0).Distribute(tmp);
  }
  else
  {
    // if the solution failed to converge, do something so that the
    // power/temperature solver doesn't completely bomb
    res(0) = 1.0;
  }

  flux_sign_fixup(res(0));

  return;
}

void slepc_initialize(const Input& inp)
{

  const std::string slepcrc_fname{inp.concatenate_path("slepcrc")};

#if 0
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

  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(&argc_val, &argv, slepcrc_fname.c_str(), nullptr);
  else
    MFEMInitializeSlepc(&argc_val, &argv, nullptr, nullptr);
#endif

  if (exists(slepcrc_fname))
    MFEMInitializeSlepc(nullptr, nullptr, slepcrc_fname.c_str(), nullptr);
  else
    MFEMInitializeSlepc(nullptr, nullptr, nullptr, nullptr);

  return;
}

void slepc_finalize()
{
  MFEMFinalizeSlepc();
  return;
}

mfem::HypreParMatrix* saaf_a_matrix(const Input &inp, 
                                   const Material_library &mat_lib,
                                   const Quadrature* const quad,
                                   ParFiniteElementSpace* fespace)
{

  const ParMesh* pmesh{fespace->GetParMesh()};
  const int skip_zeros{2};
  const int block_rank{mat_lib.ngroup()*quad->nd()};

  std::vector<DenseMatrix> omega_operator(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    omega_operator[n].SetSize(quad->dim());
    MultVVt(quad->omega(n), omega_operator[n]);
  }

  std::vector<Tensor<DenseMatrix>> streaming(pmesh->attributes.Max());
  std::vector<Vector> collision(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    const XSRecord& rec{mat_lib.mat[i].rec};
    streaming[i].reserve(block_rank);
    collision[i].SetSize(block_rank);
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      const double tot{rec.sigma_t[g]};
      for (int n{0}; n < quad->nd(); ++n)
      {
        const int dia{n + g*quad->nd()};

        omega_operator[n] *= 1.0/tot;
        streaming[i].AddElement(omega_operator[n], dia);
        omega_operator[n] *= tot;

        collision[i](dia) = tot;
      }
    }
  }
  ParBilinearForm sink_bilinear{fespace};
  MatrixTensorPWConstCoefficient streaming_func{block_rank, streaming};
  VectorPWConstCoefficient collision_func{collision};
  sink_bilinear.AddDomainIntegrator(new TensorDiffusionIntegrator{streaming_func});
  sink_bilinear.AddDomainIntegrator(new VectorMassIntegrator{collision_func});

  Tensor<Vector> boundary;
  boundary.reserve(block_rank);
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int n{0}; n < quad->nd(); ++n)
      boundary.AddElement(quad->omega(n), n+g*quad->nd());
  VectorTensorConstantCoefficient boundary_func{block_rank, boundary};
  sink_bilinear.AddBdrFaceIntegrator(
    new VectorDGTraceIntegrator{boundary_func, 1.0});

  std::vector<DenseMatrix> scattering(pmesh->attributes.Max());
  std::vector<Tensor<Vector>> scattering_saaf(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    scattering[i].SetSize(block_rank);
    const XSRecord& record{mat_lib.mat[i].rec};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      Array<int> col_idx;
      Vector scat_dat;
      record.sigma_s.GetRow(g, col_idx, scat_dat);

      const double tot{record.sigma_t[g]};

      for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
      {
        const int gprime{col_idx[gidx]};
        const double scat{scat_dat(gidx)};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int row{n + g*quad->nd()};
          for (int nprime{0}; nprime < quad->nd(); ++nprime)
          {
            const int col{nprime + gprime*quad->nd()};
            const double val{-scat*quad->weight(nprime)};
            scattering[i](row,col) = val;

            Vector v{quad->omega(n)};
            v *= val/tot;
            scattering_saaf[i].AddElement(v, row, col);
          } // nprime < quad->nd()
        } // n < quad->nd()
      } // gidx < col_idx.Size()

    } // g < mat_lib.ngroup()
    scattering_saaf[i].Transpose(); // related to the need for the transpose integrator
  } // i < pmesh->attributes.Max()
  MatrixPWConstCoefficient scattering_func{scattering};
  VectorTensorPWConstCoefficient scattering_saaf_func{block_rank, scattering_saaf};
  sink_bilinear.AddDomainIntegrator(new VectorMassIntegrator{scattering_func});
  sink_bilinear.AddDomainIntegrator(
    new TransposeIntegrator{
    new VectorConvectionIntegrator{scattering_saaf_func}});

  sink_bilinear.Assemble(skip_zeros);
  sink_bilinear.Finalize(skip_zeros); // no essential boundary conditions (maybe)

  return sink_bilinear.ParallelAssemble();

}

HypreParMatrix* saaf_m_matrix(const Input &inp, 
                              const Material_library &mat_lib,
                              const Quadrature* const quad,
                              ParFiniteElementSpace* fespace)
{

  const ParMesh* pmesh{fespace->GetParMesh()};
  const int skip_zeros{2};
  const int block_rank{mat_lib.ngroup()*quad->nd()};

  std::vector<DenseMatrix> fiss(pmesh->attributes.Max());
  std::vector<Tensor<Vector>> fiss_saaf(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    fiss[i].SetSize(block_rank);
    fiss_saaf[i].reserve(block_rank*block_rank);
    const Material_record& mat{mat_lib.mat[i]};

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      const double tot{mat.rec.sigma_t[g]};
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
            v /= tot;
            v *= val * quad->weight(nprime);
            fiss_saaf[i].AddElement(v, row, col);
          } // nprime < quad->nd()
        } // n < quad->nd()
      } // gprime < mat_lib.ngroup()
    } // g < mat_lib.ngroup()
    fiss_saaf[i].Transpose(); // related to the need for the transpose integrator
  } // i < pmesh->attributes.Max()
  MatrixPWConstCoefficient fiss_func{fiss};
  VectorTensorPWConstCoefficient fiss_saaf_func{block_rank, fiss_saaf};
  ParBilinearForm fiss_bilinear{fespace};
  fiss_bilinear.AddDomainIntegrator(new VectorMassIntegrator{fiss_func});
  fiss_bilinear.AddDomainIntegrator(
    new TransposeIntegrator{
    new VectorConvectionIntegrator{fiss_saaf_func}});

  fiss_bilinear.Assemble(skip_zeros);
  fiss_bilinear.Finalize(skip_zeros);

  return fiss_bilinear.ParallelAssemble();

}

void calc_all_scatter(const Result& scalar_flux,
                      const Material_library& mat_lib,
                      const Quadrature* quad,
                      QuadratureFunction& scat_quad,
                      QuadratureFunction& scat_saaf_quad,
                      bool include_self_scatter)
{

  scat_quad = 0.0;
  scat_saaf_quad = 0.0;

  const QuadratureSpace* qspace{scat_quad.GetSpace()};
  const ParFiniteElementSpace* fespace{scalar_flux(0).ParFESpace()};
  const ParMesh* pmesh{fespace->GetParMesh()};

  int idx{0};
  int saaf_idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{pmesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const SparseMatrix& scat{rec.sigma_s};

    const FiniteElement *fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint ip{ir.IntPoint(i)};
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {

        Array<int> col_idx;
        Vector scat_dat;
        scat.GetRow(g, col_idx, scat_dat);

        const double tot{rec.sigma_t[g]};

        double val{0.0};
        for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
        {
          const int gprime{col_idx[gidx]};
          if ((gprime != g) || (include_self_scatter))
          {
            Vector locvec(dofs.Size());
            scalar_flux(gprime).GetSubVector(dofs, locvec);
            const double flux{dofval*locvec};
            val += scat_dat(gidx)*flux;
          }
        }

        for (int n{0}; n < quad->nd(); ++n)
        {
          scat_quad(idx) = val;
          ++idx;
        }

        val /= tot;
        for (int n{0}; n < quad->nd(); ++n)
        {
          Vector dir{quad->omega(n)};
          for (int d{0}; d < dir.Size(); ++d)
          {
            scat_saaf_quad(saaf_idx) = val*dir(d);
            ++saaf_idx;
          }
        }

      }


    }

  }

  return;
}

#if 0
void calc_fission_source(const Material_library& mat_lib, 
                         const Result& scalar_flux,
                         ParGridFunction& fission_source)
{

  ParFiniteElementSpace* fespace = fission_source.ParFESpace();
  ParMesh* pmesh = fespace->GetParMesh();

  fission_source = 0.0;

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Vector nusf(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      nusf(i) = mat_lib.mat[i].rec.nusf[g];

    PWConstCoefficient nusf_func{nusf};
    GridFunctionCoefficient scalar_flux_func{&scalar_flux(g)};
    ProductCoefficient src{nusf_func, scalar_flux_func};

    ParGridFunction tmp{fespace};
    tmp.ProjectCoefficient(src);

    fission_source += tmp;

  }

  return;
}

void calc_chi(const Material_library& mat_lib,
              const Result& scalar_flux,
              std::vector<ParGridFunction>& chi,
              bool flux_free_chi)
{

  ParFiniteElementSpace* fespace = chi[0].ParFESpace();
  ParMesh* pmesh = fespace->GetParMesh();

  if (mat_lib.rx_wide_chi)
  {

    for (int g{0}; g < mat_lib.ngroup(); ++g)
      chi[g] = mat_lib.chi[g];

  }
  else if (flux_free_chi)
  {

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      Vector chi_vec(pmesh->attributes.Max());
      for (int i{0}; i < pmesh->attributes.Max(); ++i)
        chi_vec(i) = mat_lib.mat[i].rec.chi[g];
      PWConstCoefficient chi_func{chi_vec};
      chi[g].ProjectCoefficient(chi_func);
    }

  }
  else // isotopic chi
  {

    QuadratureSpace qspace{pmesh, fespace->GetOrder(0)+1};
    std::vector<QuadratureFunction> chi_quad;
    chi_quad.reserve(mat_lib.ngroup());
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      chi_quad.emplace_back(&qspace);
      chi_quad[g] = 0.0;
    }

    int idx{0};

    for (int e{0}; e < qspace.GetNE(); ++e)
    {

      const IntegrationRule& ir{qspace.GetElementIntRule(e)};
      const int attribute{pmesh->GetAttribute(e)-1};
      const Material_record& mat{mat_lib.mat[attribute]};

      const FiniteElement* const fe{fespace->GetFE(e)};
      Array<int> dofs;
      fespace->GetElementDofs(e, dofs);

      for (int i{0}; i < ir.GetNPoints(); ++i)
      {

        const IntegrationPoint& ip{ir.IntPoint(i)};
        Vector dofval(dofs.Size());
        fe->CalcShape(ip, dofval);

        std::vector<double> spectrum(mat_lib.ngroup());
        for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
        {
          Vector locvec(dofs.Size());
          scalar_flux(gprime).GetSubVector(dofs, locvec);
          spectrum[gprime] = dofval*locvec;
        }

        const std::vector<double> this_chi{mat.calculate_chi(spectrum)};
        for (int g{0}; g < mat_lib.ngroup(); ++g)
          chi_quad[g](idx) = this_chi[g];

        ++idx;

      }

    }

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      QuadratureFunctionCoefficient chi_quad_func{chi_quad[g]};
      chi[g].ProjectCoefficient(chi_quad_func);
    }

  }

  return;
}

#endif

void dg_spn_power_iteration(Input &inp,
                            const Material_library &mat_lib,
                            ParMesh *pmesh)
{      

  MFEM_VERIFY(mat_lib.ngroup() == 1, "DG diffusion only written for one group diffusion for now");
  MFEM_VERIFY(inp.spn_order() == 1, "DG diffusion only written for one group diffusion for now");

  constexpr int vdim{1}; // TODO this will surely have to be changed :(
  constexpr int skip_zeros{2};

  DG_FECollection fec{inp.order, pmesh->Dimension()};
  ParFiniteElementSpace fespace{pmesh, &fec, vdim, Ordering::byVDIM};
  const int myid{fespace.GetMyRank()};

  // parameters for the DGDiffusionIntegrator
  // results in Symmetric Interior Penalty (IP or SIPG) method
  const double sigma{-1.0};
  const double kappa{static_cast<double>(std::pow(fec.GetOrder()+1,2))};

  ParGridFunction scalar_flux{&fespace};
  scalar_flux = 1.0;
  
  Vector diffusion_vec(pmesh->attributes.Max());
  Vector removal_vec(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    diffusion_vec(i) = 1.0/(3.0*mat_lib.mat[i].rec.sigma_tr[0]);
    removal_vec(i)   = mat_lib.mat[i].rec.sigma_r(0,0);
  }
  PWConstCoefficient diffusion{diffusion_vec};
  PWConstCoefficient removal{removal_vec};

  ParBilinearForm a{&fespace};
  a.AddDomainIntegrator(new DiffusionIntegrator{diffusion});
  a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator{diffusion, sigma, kappa});
  a.AddBdrFaceIntegrator(new DGDiffusionIntegrator{diffusion, sigma, kappa});
  a.AddDomainIntegrator(new MassIntegrator{removal});
  a.Assemble(skip_zeros);
  a.Finalize(skip_zeros);
  std::unique_ptr<HypreParMatrix> Ahypre{a.ParallelAssemble()};

  ConstantCoefficient zero{0.0}; // zero flux condition at boundary

  GridFunctionCoefficient scalar_flux_func{&scalar_flux};

  Vector nusf_vec(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
    nusf_vec(i) = mat_lib.mat[i].rec.nusf[0];
  PWConstCoefficient nusf_func{nusf_vec};
   
  ProductCoefficient fiss_src_func{scalar_flux_func, nusf_func};

  Vector keff_vec(1);
  keff_vec = 1.0/1.0;
  PWConstCoefficient keff_vec_func{keff_vec};
  ProductCoefficient scaled_fiss_src_func{keff_vec_func, fiss_src_func};

  ParLinearForm b{&fespace};
  b.AddDomainIntegrator(new DomainLFIntegrator{scaled_fiss_src_func});
  b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator{zero, sigma, kappa});

  ParLinearForm fiss_src{&fespace};
  fiss_src.AddDomainIntegrator(new DomainLFIntegrator{nusf_func});
  fiss_src.Assemble();

  HyprePCG cg{*Ahypre};
  cg.SetPrintLevel(inp.hypre_print_level());
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetTol(inp.tol.inner_rel);
  HYPRE_PCGSetAbsoluteTol(cg, inp.tol.inner_abs);

  HypreBoomerAMG amg{*Ahypre};
  amg.SetPrintLevel(inp.hypre_print_level());
  cg.SetPreconditioner(amg);

  HypreParVector x{&fespace};
  x = 1.0;

  double keff{1.0};
  double k_conv{inp.tol.k*1e2+1.0};
  double phi_conv{inp.tol.phi*1e2+1.0};
  double fiss_sum_old{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    keff_vec = 1.0/keff; // cannot scale entire vector as it contains boundary terms as well
    keff_vec_func.UpdateConstants(keff_vec);
    b.Assemble();
    std::unique_ptr<HypreParVector> bhypre{b.ParallelAssemble()};

    HypreParVector x_old{x};
    x_old = x;
    cg.Mult(b, x);
    scalar_flux.Distribute(x);

    const double keff_old{keff};
    const double fiss_sum{fiss_src(scalar_flux)};
    if (iter > 0) keff *= fiss_sum / fiss_sum_old;

    k_conv = std::abs(keff - keff_old);
    x_old -= x;
    phi_conv = par_norml2(x_old);
    fiss_sum_old = fiss_sum;

    if (myid == 0)
      std::cout << "iter=" << iter 
                << " keff=" << keff 
                << " dk=" << k_conv
                << " dphi=" << phi_conv
                << std::endl;

  }

  ParaViewDataCollection output{inp.input_file_stub() + "_dg_spn", pmesh};
  output.RegisterField("scalar_flux", &scalar_flux);
  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  return;
}

void dg_transport_power_iteration(Input &inp,
                                  const Material_library &mat_lib,
                                  const Quadrature *quad,
                                  ParMesh *pmesh)
{

  MFEM_VERIFY(mat_lib.ngroup() == 1, "DG transport only written for one group for now");

  constexpr int vdim{1}; // TODO this will surely have to be changed :(
  constexpr int skip_zeros{2};

  DG_FECollection fec{inp.order, pmesh->Dimension()};
  ParFiniteElementSpace fespace{pmesh, &fec, vdim, Ordering::byVDIM};
  const int myid{fespace.GetMyRank()};

  std::vector<std::unique_ptr<ParGridFunction>> angular_flux(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    angular_flux[n] = std::make_unique<ParGridFunction>(&fespace);
    *angular_flux[n] = 1.0;
  }

  Result scalar_flux{&fespace, mat_lib.ngroup()};
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux(g) = 1.0;

  constexpr double alpha{1.0}; // necessary for DG convection terms
  std::vector<VectorConstantCoefficient> dir;
  dir.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    dir.emplace_back(quad->omega(n));

  std::vector<std::unique_ptr<HypreParMatrix>> Ahypre(quad->nd());

  {

    Vector tot(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      tot(i) = 5*mat_lib.mat[i].rec.sigma_t[0];
    PWConstCoefficient tot_func{tot};

    for (int n{0}; n < quad->nd(); ++n)
    {
      ParBilinearForm a{&fespace};

      /* method from SFENT paper */
      a.AddDomainIntegrator(new ConservativeConvectionIntegrator{dir[n], alpha});
      a.AddInteriorFaceIntegrator(new DGTraceIntegrator{dir[n], alpha});
      a.AddBdrFaceIntegrator(new DGTraceIntegrator{dir[n], alpha});

      /* method from MFEM ex9
      a.AddDomainIntegrator(new ConvectionIntegrator{dir[n], -1.0});
      a.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator{dir[n], -1.0});
      a.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator{dir[n], -1.0});
      */

      a.AddDomainIntegrator(new MassIntegrator{tot_func});
      a.Assemble(skip_zeros);
      a.Finalize(skip_zeros);
      Ahypre[n].reset(a.ParallelAssemble());
      const std::string name{"ahypre_n"+std::to_string(n)};
      Ahypre[n]->Print(name.c_str());
    }

  }

  const int integration_order{fec.GetOrder()+1};
  QuadratureSpace qspace{pmesh, integration_order};

  std::vector<QuadratureFunction> fiss_quad, upscatter_quad;
  fiss_quad.reserve(mat_lib.ngroup());
  upscatter_quad.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_quad.emplace_back(&qspace);
    upscatter_quad.emplace_back(&qspace);
  }

  QuadratureFunction dnscatter_quad{&qspace};

  QuadratureFunction selfscatter_quad{&qspace};
  QuadratureFunctionCoefficient quad_func{selfscatter_quad};

  ConstantCoefficient inflow{0.0}; // this would be the inflow BC (maybe for reflective bc)
  ConstantCoefficient source{10.0};

  std::vector<std::unique_ptr<ParLinearForm>> b(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
  {
    b[n] = std::make_unique<ParLinearForm>(&fespace);
    b[n]->AddBdrFaceIntegrator(new BoundaryFlowIntegrator{inflow, dir[n], -1*alpha});
    //b[n]->AddDomainIntegrator(new DomainLFIntegrator{quad_func});
    b[n]->AddDomainIntegrator(new DomainLFIntegrator{source});
  }

  HypreGMRES gmres{MPI_COMM_WORLD};
  gmres.SetPrintLevel((inp.verbose) ? 9 : 0);
  gmres.SetMaxIter(inp.tol.max_inner);
  gmres.SetTol(inp.tol.inner_rel);
  gmres.SetAbsTol(inp.tol.inner_abs);

  QuadratureFunction fiss_src{&qspace};
  double fiss_sum_old{0.0};

  double keff{1.0};
  double k_conv{inp.tol.k*1e2+1.0};
  double phi_conv{inp.tol.phi*1e2+1.0};
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    calc_fiss_upscatter(scalar_flux, mat_lib, fiss_quad, upscatter_quad, inp.flux_free_chi);
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      fiss_quad[g] /= keff;
      upscatter_quad[g] += fiss_quad[g];
    }

    constexpr int g{0}; // TODO
    calc_downscatter(g, scalar_flux, mat_lib, dnscatter_quad);
    dnscatter_quad += upscatter_quad[g];

    double scattering_conv{inp.tol.scatter*1e2+1.0};
    for (int scattering_iter{0}; 
        (scattering_iter < inp.tol.max_scatter) &&
        (scattering_conv > inp.tol.scatter);
        ++scattering_iter)
    {

      calc_selfscatter(g, scalar_flux(g), mat_lib, selfscatter_quad);
      selfscatter_quad += dnscatter_quad;

      for (int n{0}; n < quad->nd(); ++n)
      {

        b[n]->Assemble();
        std::unique_ptr<HypreParVector> bhypre{b[n]->ParallelAssemble()};

        HypreBoomerAMG amg{*(Ahypre[n])};
        amg.SetPrintLevel(inp.hypre_print_level());
        gmres.SetPreconditioner(amg);

        gmres.SetOperator(*(Ahypre[n]));

        std::unique_ptr<HypreParVector> x{angular_flux[n]->ParallelAssemble()};
        gmres.Mult(*bhypre, *x);
        angular_flux[n]->Distribute(*x);

        if (myid == 0)
          std::cout << "......n=" << n << std::endl;

      }

      ParGridFunction old{scalar_flux(g)};
      dg_calc_scalar_flux(angular_flux, quad, scalar_flux(g));
      old -= scalar_flux(g);
      const double local_scattering_conv{old.Normlinf() / scalar_flux(g).Normlinf()};
      MPI_Allreduce(&local_scattering_conv, &scattering_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if (myid == 0)
        std::cout << "..."
                  << scattering_iter
                  << " g=" << g
                  << " conv=" << scattering_conv
                  << std::endl;

    }


    QuadratureFunction fiss_src_old{fiss_src};
    calc_fiss_src(scalar_flux, mat_lib, fiss_src);

    const double local_fiss_sum{fiss_src.Sum()};
    double fiss_sum{0.0};
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double keff_old{keff};
    if (iter > 0) keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(keff_old - keff);

    // note: some nodes may contain zero fissile elements depending on decomposition
    fiss_src_old -= fiss_src;
    const double delta{fiss_src_old.Normlinf()};
    const double norm{fiss_src.Normlinf()};
    const double local_phi_conv{(norm > 0.0) ? delta / norm : 0.0};
    MPI_Allreduce(&local_phi_conv, &phi_conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    fiss_sum_old = fiss_sum;

    if (myid == 0)
      std::cout << "iter=" << iter 
                << " keff=" << keff 
                << " dk=" << k_conv
                << " dphi=" << phi_conv
                << std::endl;

  }

  dg_calc_scalar_flux(angular_flux, quad, scalar_flux(0));

  const size_t angle_width{std::to_string(quad->nd()-1).length()};
  const size_t group_width{std::to_string(mat_lib.ngroup()-1).length()};

  ParaViewDataCollection output{inp.input_file_stub() + "_dg_transport", pmesh};
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    std::ostringstream scalar_flux_record;
    scalar_flux_record << "scalar_flux_g"
                       << std::setfill('0')
                       << std::setw(group_width)
                       << g;
    output.RegisterField("scalar_flux", &scalar_flux(g));
  }
  for (int n{0}; n < quad->nd(); ++n)
  {
    std::ostringstream angular_flux_record;
    angular_flux_record << "psi_n"
                        << std::setfill('0')
                        << std::setw(angle_width)
                        << n;
    output.RegisterField(angular_flux_record.str(), angular_flux[n].get());
  }
  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  return;
}

void dg_calc_scalar_flux(std::vector<std::unique_ptr<ParGridFunction>> &angular_flux,
                         const Quadrature *quad,
                         ParGridFunction& scalar_flux)
{

  scalar_flux = 0.0;
  for (int n{0}; n < quad->nd(); ++n)
  {
    *(angular_flux[n]) *= quad->weight(n);
    scalar_flux += *(angular_flux[n]);
    *(angular_flux[n]) /= quad->weight(n);
  }

  return;
}

void spn_partial_assembly_power_iteration(const Input &inp,
                                          const Material_library &mat_lib,
                                          mfem::Array<int> &ess_bdr,
                                          Result &scalar_flux)
{

  ParFiniteElementSpace *fespace = scalar_flux(0).ParFESpace();
  ParMesh *pmesh = fespace->GetParMesh();
  MFEM_VERIFY(UsesTensorBasis(*fespace),
      "AssemblyLevel::PARTIAL requries tensor basis finite elements"
      " (e.g. quads in 2d)");

  const int myid{fespace->GetMyRank()};

  std::vector<PWConstCoefficient> diffusion_func;
  diffusion_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector diffusion_vec(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      diffusion_vec(i) = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_tr[g]);
    diffusion_func.emplace_back(diffusion_vec);
  }

  std::vector<PWConstCoefficient> removal_func;
  removal_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector removal_vec(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      removal_vec(i) = mat_lib.mat[i].rec.sigma_r(g,0);
    removal_func.emplace_back(removal_vec);
  }

  std::vector<std::unique_ptr<ParBilinearForm>> a(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    a[g] = std::make_unique<ParBilinearForm>(fespace);
    a[g]->SetAssemblyLevel(AssemblyLevel::PARTIAL); // TODO consider ELEMENT
    a[g]->AddDomainIntegrator(new DiffusionIntegrator{diffusion_func[g]});
    a[g]->AddDomainIntegrator(new MassIntegrator{removal_func[g]});
    a[g]->Assemble();
  }

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  const int integration_order{fespace->GetOrder(0)+1};
  QuadratureSpace qspace{pmesh, integration_order};

  std::vector<QuadratureFunction> fiss_quad, upscatter_quad;
  fiss_quad.reserve(mat_lib.ngroup());
  upscatter_quad.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_quad.emplace_back(&qspace);
    upscatter_quad.emplace_back(&qspace);
  }

  QuadratureFunction dnscatter_quad{&qspace};
  QuadratureFunctionCoefficient dnscatter_quad_func{dnscatter_quad};

  std::vector<std::unique_ptr<ParLinearForm>> b(mat_lib.ngroup());
  ConstantCoefficient one{1.0};
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    b[g] = std::make_unique<ParLinearForm>(fespace);
    b[g]->AddDomainIntegrator(new DomainLFIntegrator{dnscatter_quad_func});
  }

  QuadratureFunction fiss_src_quad{&qspace};

  ConstantCoefficient zero{0.0};
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    scalar_flux(g) = 1.0;
    scalar_flux(g).ProjectBdrCoefficient(zero, ess_bdr);
  }

  CGSolver cg{MPI_COMM_WORLD};
  cg.SetPrintLevel(inp.mfem_print_level());
  cg.SetMaxIter(inp.tol.max_inner);
  cg.SetRelTol(inp.tol.inner_rel);
  cg.SetAbsTol(inp.tol.inner_abs);

  scalar_flux.keff = 1.0;
  double k_conv{inp.tol.k*1e2+1.0};
  double phi_conv{inp.tol.phi*1e2+1.0};
  double fiss_src{1.0};
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    const double fiss_src_old{fiss_src};
    ParGridFunction flux_old{scalar_flux(0)};

    calc_fiss_upscatter(scalar_flux, mat_lib, fiss_quad, upscatter_quad, inp.flux_free_chi);
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      fiss_quad[g] /= scalar_flux.keff;
      upscatter_quad[g] += fiss_quad[g];
    }

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      calc_downscatter(g, scalar_flux, mat_lib, dnscatter_quad);
      dnscatter_quad += upscatter_quad[g];

      b[g]->Assemble();

      OperatorPtr A;
      Vector B, X;


      a[g]->FormLinearSystem(ess_tdof_list, scalar_flux(g), *(b[g]), A, X, B);

      OperatorJacobiSmoother prec{*(a[g]), ess_tdof_list};
      cg.SetPreconditioner(prec);

      cg.SetOperator(*A);
      cg.Mult(B, X);
      a[g]->RecoverFEMSolution(X, *(b[g]), scalar_flux(g));

    }

    const double kold{scalar_flux.keff};
    calc_fiss_src(scalar_flux, mat_lib, fiss_src_quad);
    fiss_src = par_norml2(fiss_src_quad);
    scalar_flux.keff *= fiss_src/fiss_src_old;

    k_conv = std::abs(scalar_flux.keff - kold);
    flux_old -= scalar_flux(0);
    phi_conv = par_norml2(flux_old);

    if (myid == 0)
      std::cout << "iter=" << iter
                << " keff=" << scalar_flux.keff
                << " dk=" << k_conv
                << " dphi=" << phi_conv
                << std::endl;

  }

  return;
}

void calc_src(const Material_library &mat_lib,
              const Quadrature* const quad,
              const ParGridFunction &scalar_flux,
              const double keff,
              QuadratureFunction &src_quad,
              QuadratureFunction &saaf_src_quad,
              bool flux_free_chi,
              bool include_self_scatter)
{

  src_quad = 0.0;
  saaf_src_quad = 0.0;

  const QuadratureSpace *qspace{src_quad.GetSpace()};
  const ParFiniteElementSpace *fespace{scalar_flux.ParFESpace()};
  const ParMesh *pmesh{fespace->GetParMesh()};

  int idx{0};
  int saaf_idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{pmesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const SparseMatrix& scat{rec.sigma_s};

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint ip{ir.IntPoint(i)};

      Vector val(quad->nd()*mat_lib.ngroup());
      val = 0.0;

      std::vector<double> spectrum_vec(mat_lib.ngroup());
      Vector spectrum{spectrum_vec.data(), static_cast<int>(spectrum_vec.size())};
      scalar_flux.GetVectorValue(e, ip, spectrum);

      if (rec.is_fiss)
      {
        const Vector nusf{const_cast<double*>(rec.nusf.data()), 
          static_cast<int>(rec.nusf.size())};
        const double fiss_sum{spectrum*nusf};
        const std::vector<double> chi{
          (mat_lib.rx_wide_chi) ? mat_lib.chi :
          ((flux_free_chi) ? rec.chi : mat.calculate_chi(spectrum_vec))};
        for (int g{0}; g < mat_lib.ngroup(); ++g)
          for (int n{0}; n < quad->nd(); ++n)
            val(n+g*quad->nd()) += chi[g]*fiss_sum/keff;
      }

      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {

        Array<int> col_idx;
        Vector scat_data;
        scat.GetRow(g, col_idx, scat_data);

        for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
        {

          const int gprime{col_idx[gidx]};
          if ((include_self_scatter) || (gprime != g)) // do not include self scatter
          {
            for (int n{0}; n < quad->nd(); ++n)
              val(n+g*quad->nd()) += spectrum(gprime)*scat_data(gidx);
          }

        }

      }

      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          src_quad(idx) = val(n+g*quad->nd());
          ++idx;

          Vector dir{quad->omega(n)};
          for (int d{0}; d < quad->dim(); ++d)
          {
            saaf_src_quad(saaf_idx) = val(n+g*quad->nd())/tot*dir(d);
            ++saaf_idx;
          }
        }
      }


    }

  }

  return;
}

void calc_self_scatter(const Material_library &mat_lib,
                       const Quadrature* const quad,
                       const ParGridFunction &scalar_flux,
                       QuadratureFunction &self_scatter_quad,
                       QuadratureFunction &saaf_self_scatter_quad)
{

  self_scatter_quad = 0.0;
  saaf_self_scatter_quad = 0.0;

  const QuadratureSpace *qspace{self_scatter_quad.GetSpace()};
  const ParFiniteElementSpace *fespace{scalar_flux.ParFESpace()};
  const ParMesh *pmesh{fespace->GetParMesh()};

  int idx{0};
  int saaf_idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{pmesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};
    const SparseMatrix& scat{rec.sigma_s};

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint ip{ir.IntPoint(i)};

      Vector flux(mat_lib.ngroup());
      scalar_flux.GetVectorValue(e, ip, flux);

      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{rec.sigma_t[g]};
        const double val{flux(g)*scat.Elem(g,g)};
        for (int n{0}; n < quad->nd(); ++n)
        {
          self_scatter_quad(idx) = val;
          ++idx;

          Vector dir{quad->omega(n)};
          for (int d{0}; d < quad->dim(); ++d)
          {
            saaf_self_scatter_quad(saaf_idx) = val/tot*dir(d);
            ++saaf_idx;
          }
        }
      }

    }
  }

  return;
}

void calc_scalar_flux_vector_vector(const ParGridFunction &angular_flux,
                                    const Material_library &mat_lib,
                                    const Quadrature* const quad,
                                    ParGridFunction &scalar_flux)
{

  scalar_flux = 0.0;

  const int size{scalar_flux.Size() / mat_lib.ngroup()}; // integer division
  for (int i{0}; i < size; ++i)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        scalar_flux(g + i*mat_lib.ngroup()) +=
          quad->weight(n) *
          angular_flux(n + g*quad->nd() + i*quad->nd()*mat_lib.ngroup());
      }
    }
  }

  return;
}

void calc_fiss_src_vector(const ParGridFunction &scalar_flux,
                          const Material_library &mat_lib,
                          QuadratureFunction &fiss_src_quad)
{

  fiss_src_quad = 0.0;

  const QuadratureSpace *qspace{fiss_src_quad.GetSpace()};
  const ParFiniteElementSpace *fespace{scalar_flux.ParFESpace()};
  const ParMesh *pmesh{fespace->GetParMesh()};

  int idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{pmesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};

    const Vector nusf{const_cast<double*>(rec.nusf.data()), 
      static_cast<int>(rec.nusf.size())};

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint ip{ir.IntPoint(i)};

      Vector flux(mat_lib.ngroup());
      scalar_flux.GetVectorValue(e, ip, flux);

      fiss_src_quad(idx) = flux*nusf;
      ++idx;

    }

  }


  return;
}

void saaf_angle_nda_slepc(const Quadrature* const quad,
                          Timer &time,
                          const Input &inp,
                          const Material_library &mat_lib,
                          Result &angular_flux_res,
                          Result &scalar_flux_res)
{

  ParGridFunction &angular_flux{angular_flux_res(0)};
  ParGridFunction &scalar_flux{scalar_flux_res(0)};

  ParFiniteElementSpace *angular_flux_fespace = angular_flux.ParFESpace();
  ParFiniteElementSpace *scalar_flux_fespace = scalar_flux.ParFESpace();
  ParMesh *pmesh = angular_flux_fespace->GetParMesh();

  const int myid{angular_flux_fespace->GetMyRank()};

  std::unique_ptr<HypreParMatrix> A;
  std::unique_ptr<Solver> Asol;
  std::unique_ptr<HypreSolver> Aprec;

  std::unique_ptr<HypreParMatrix> D;
  std::unique_ptr<HypreParMatrix> D_fission;

  { // A operator

    std::vector<DenseMatrix> omega_mat(quad->nd());
    for (int n{0}; n < quad->nd(); ++n)
    {
      omega_mat[n].SetSize(quad->dim());
      MultVVt(quad->omega(n), omega_mat[n]);
    }

    Tensor<Vector> omega;
    omega.reserve(quad->nd()*mat_lib.ngroup());
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      for (int n{0}; n < quad->nd(); ++n)
        omega.AddElement(quad->omega(n), n+g*quad->nd());
    VectorTensorConstantCoefficient omega_func{quad->nd()*mat_lib.ngroup(), omega};

    std::vector<Tensor<DenseMatrix>> leakage(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      leakage[i].reserve(quad->nd()*mat_lib.ngroup());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int idx{n+g*quad->nd()};
          omega_mat[n] *= 1.0/tot;
          leakage[i].AddElement(omega_mat[n], idx);
          omega_mat[n] *= tot;
        }
      }
    }
    MatrixTensorPWConstCoefficient leakage_func{quad->nd()*mat_lib.ngroup(), leakage};

    std::vector<Vector> total(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      total[i].SetSize(quad->nd()*mat_lib.ngroup());
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        const double tot{mat_lib.mat[i].rec.sigma_t[g]};
        for (int n{0}; n < quad->nd(); ++n)
        {
          const int idx{n+g*quad->nd()};
          total[i](idx) = tot;
        }
      }
    }
    VectorPWConstCoefficient total_func{total};

    ParBilinearForm a{angular_flux_fespace};
    a.AddDomainIntegrator(new TensorDiffusionIntegrator{leakage_func});
    a.AddDomainIntegrator(new VectorMassIntegrator{total_func});
    a.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});

    a.Assemble(2);
    a.Finalize(2);
    A.reset(a.ParallelAssemble());

    std::unique_ptr<HypreBoomerAMG> amg{std::make_unique<HypreBoomerAMG>(*A)};
    amg->SetPrintLevel(inp.hypre_print_level());
    HYPRE_BoomerAMGSetNumFunctions(*amg, mat_lib.ngroup()*quad->nd());
    Aprec = std::move(amg);

    std::unique_ptr<HyprePCG> cg{std::make_unique<HyprePCG>(*A)};
    cg->SetPrintLevel(2);
    cg->SetTol(inp.tol.inner_rel);
    cg->SetAbsTol(inp.tol.inner_abs);
    cg->SetMaxIter(inp.tol.max_inner);
    cg->SetPreconditioner(*Aprec);
    Asol = std::move(cg);

  } // A operator

  const int integration_order{10};
  QuadratureSpace qspace{pmesh, integration_order};

  QuadratureFunction src_quad{&qspace, quad->nd()*mat_lib.ngroup()};
  QuadratureFunction saaf_src_quad{&qspace, quad->dim()*quad->nd()*mat_lib.ngroup()};
  VectorQuadratureFunctionCoefficient src_quad_func{src_quad};
  VectorQuadratureFunctionCoefficient saaf_src_quad_func{saaf_src_quad};

  ParLinearForm b{angular_flux_fespace};
  b.AddDomainIntegrator(new VectorDomainLFIntegrator{src_quad_func});
  b.AddDomainIntegrator(new VectorDomainLFGradIntegrator{saaf_src_quad_func});

  // NDA operator

  std::vector<Vector> diffusion(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    diffusion[i].SetSize(mat_lib.ngroup());
    for (int g{0}; g < mat_lib.ngroup(); ++g)
      diffusion[i](g) = 1.0 / (3.0 * mat_lib.mat[i].rec.sigma_t[g]);
  }
  VectorPWConstCoefficient diffusion_func{diffusion};

  std::vector<DenseMatrix> removal(pmesh->attributes.Max());
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    const XSRecord &rec{mat_lib.mat[i].rec};

    removal[i].SetSize(mat_lib.ngroup());
    rec.sigma_s.ToDenseMatrix(removal[i]);
    removal[i].Neg();

    for (int g{0}; g < mat_lib.ngroup(); ++g)
      removal[i](g,g) += rec.sigma_t[g];

  }
  MatrixPWConstCoefficient removal_func{removal};

  std::unique_ptr<ParFiniteElementSpace> current_fespace{
    std::make_unique<ParFiniteElementSpace>(pmesh,
    angular_flux_fespace->FEColl(), pmesh->Dimension()*mat_lib.ngroup(),
    Ordering::byVDIM)};
  ParGridFunction current{current_fespace.get()};

  // the diffusion coefficient is also called the drift velocity
  NDADiffusionCoefficient nda_diffusion_func{scalar_flux, current, angular_flux, quad, mat_lib};
  NDABoundaryFactorCoefficient nda_boundary_factor_func{scalar_flux, current, angular_flux, quad, mat_lib};

  ParBilinearForm d{scalar_flux_fespace};
  d.AddDomainIntegrator(new VectorDiffusionIntegrator{diffusion_func});
  d.AddDomainIntegrator(new VectorMassIntegrator{removal_func});
  d.AddDomainIntegrator(new TransposeIntegrator{
    new VectorConvectionIntegrator{nda_diffusion_func, -1.0}});
  d.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{nda_boundary_factor_func, 2.0, 0.0});

  { // NDA fission terms

    std::vector<DenseMatrix> fission(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
    {
      fission[i].SetSize(mat_lib.ngroup());
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
    MatrixPWConstCoefficient fission_func{fission};

    ParBilinearForm d_fission{scalar_flux_fespace};
    d_fission.AddDomainIntegrator(new VectorMassIntegrator{fission_func});
    d_fission.Assemble(2);
    d_fission.Finalize(2);
    D_fission.reset(d_fission.ParallelAssemble()); // this will only be assembled once

  } // NDA fission terms

  constexpr Operator::Type petsc_format{Operator::PETSC_MATHYPRE};

  MFEMInitializeSlepc(nullptr, nullptr, "slepcrc", nullptr);

  std::unique_ptr<const PetscParMatrix> D_fission_petsc{
    std::make_unique<PetscParMatrix>(D_fission.get(), petsc_format)};

  std::unique_ptr<SlepcEigenSolver> slepc{std::make_unique<SlepcEigenSolver>(MPI_COMM_WORLD)};
  slepc->SetNumModes(1); // only fundamental mode
  //slepc->SetTol(std::min(inp.tol.k, inp.tol.phi)); // needs to be tight
  slepc->SetTol(1e-12); // needs to be tight
  slepc->SetMaxIter(100);
  slepc->SetWhichEigenpairs(SlepcEigenSolver::SMALLEST_MAGNITUDE);

  angular_flux = 1.0;
  scalar_flux = 1.0;

  double keff{1.0};
  double k_conv{inp.tol.k*1e2+1.0};
  double phi_conv{inp.tol.phi*1e2+1.0};
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    const double kold{keff};

    // transport sweep
    calc_src(mat_lib, quad, scalar_flux, keff, src_quad, saaf_src_quad, inp.flux_free_chi, true); // include self scatter
    b.Assemble();
    std::unique_ptr<HypreParVector> bvec{b.ParallelAssemble()};
    std::unique_ptr<HypreParVector> xvec{angular_flux.ParallelAssemble()};
    Asol->Mult(*bvec, *xvec);
    angular_flux.Distribute(*xvec);

    calc_scalar_flux_vector_vector(angular_flux, mat_lib, quad, scalar_flux);
    calc_current_vector_vector(angular_flux, mat_lib, quad, current);

    // reconstruct NDA operator
    d.Update();
    d.Assemble(2);
    d.Finalize(2);
    D.reset(d.ParallelAssemble());

    /*
    d_fission.Update();
    d_fission.Assemble(2);
    d_fission.Finalize(2);
    D_fission.reset(d_fission.ParallelAssemble());
    power_iteration_amgx(D.release(), D_fission.release(), inp, time, scalar_flux_res, mat_lib.ngroup());
    */

    const std::unique_ptr<const PetscParMatrix> D_petsc{
      std::make_unique<PetscParMatrix>(D.get(), petsc_format)};
    slepc->SetOperators(*D_petsc, *D_fission_petsc);
    slepc->Solve();

    MFEM_VERIFY(slepc->GetNumConverged() >= 1,
      " inner slepc nda failed to converge. cannot continue.");

    slepc->GetEigenvalue(0, keff);
    keff = 1.0/keff;
    Vector tmp(scalar_flux_fespace->GetTrueVSize());
    slepc->GetEigenvector(0, tmp);
    scalar_flux.Distribute(tmp);
    flux_sign_fixup(scalar_flux);

    k_conv = std::abs(keff - kold);
    phi_conv = 0.0; // TODO implement

    if (myid == 0)
      std::cout << iter 
        << " keff=" << keff
        << " dk=" << k_conv 
        << " dphi=" << phi_conv 
        << std::endl;

  }

  slepc.reset();
  D_fission_petsc.reset();

  MFEMFinalizeSlepc();

  {

    ParGridFunction drift_velocity{current};
    drift_velocity.ProjectCoefficient(nda_diffusion_func);

    ParGridFunction boundary_factor{current};
    boundary_factor = 0.0;
    boundary_factor.ProjectBdrCoefficient(nda_boundary_factor_func, pmesh->bdr_attributes);

    ParaViewDataCollection output("debug_output", pmesh);

    output.RegisterField("current", &current);
    output.RegisterField("scalar_flux", &scalar_flux);
    output.RegisterField("drift_velocity", &drift_velocity);
    output.RegisterField("boundary_factor", &boundary_factor);

    ParFiniteElementSpace tmp_fespace{pmesh, scalar_flux_fespace->FEColl()};
    ParFiniteElementSpace der_fespace{pmesh, scalar_flux_fespace->FEColl(), pmesh->Dimension(), Ordering::byVDIM};
    std::vector<ParGridFunction> grad_scalar_flux;
    grad_scalar_flux.reserve(mat_lib.ngroup());
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      GridFunctionCoefficient gfc{&scalar_flux, g+1};
      ParGridFunction tmp{&tmp_fespace};
      tmp.ProjectCoefficient(gfc);

      GradientGridFunctionCoefficient ggfc{&tmp};
      grad_scalar_flux.emplace_back(&der_fespace);
      grad_scalar_flux.back().ProjectCoefficient(ggfc);

      const std::string field_name{"grad_scalar_flux_g" + std::to_string(g)};
      output.RegisterField(field_name, &grad_scalar_flux.back());
    }

    output.Save();

  }

  return;
}

void calc_current_vector_vector(const ParGridFunction &angular_flux,
                                const Material_library &mat_lib,
                                const Quadrature * const quad,
                                ParGridFunction &current)
{

  current = 0.0;
  const int size{current.Size() / (quad->dim()*mat_lib.ngroup())}; // integer division
  for (int i{0}; i < size; ++i)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir{quad->omega(n)};
        for (int d{0}; d < quad->dim(); ++d)
        {
          current(d + g*quad->dim() + i*quad->dim()*mat_lib.ngroup()) +=
            dir(d) * quad->weight(n) *
            angular_flux(n + g*quad->nd() + i*quad->nd()*mat_lib.ngroup());
        }
      }
    }
  }

  return;
}

void calc_fiss_upscatter_vector(const Result &scalar_flux,
                                const Material_library &mat_lib,
                                const Quadrature * const quad,
                                std::vector<QuadratureFunction> &fiss_upscatter_quad,
                                std::vector<QuadratureFunction> &saaf_fiss_upscatter_quad,
                                const double keff,
                                const bool flux_free_chi)
{

  const QuadratureSpace* const qspace{fiss_upscatter_quad[0].GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    fiss_upscatter_quad[g] = 0.0;
    saaf_fiss_upscatter_quad[g] = 0.0;
  }

  int idx{0};
  int saaf_idx{0};

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

      // get all xs at this point
      // this will eventually be temperautre dependent
      const std::vector<double> &nusf{rec.nusf};
      const std::vector<double> &tot{rec.sigma_t};
      const SparseMatrix &scat{rec.sigma_s};
      std::vector<double> chi;
      if (rec.is_fiss)
      {

        // accumulate flux spectrum
        for (int g{0}; g < mat_lib.ngroup(); ++g)
        {
          Vector locvec(dofs.Size());
          scalar_flux(g).GetSubVector(dofs, locvec);
          spectrum[g] = dofval*locvec;
        }

        // set chi appropriately
        chi.resize(mat_lib.ngroup());
        if (flux_free_chi)
          chi = rec.chi;
        else if (mat_lib.rx_wide_chi)
          chi = mat_lib.chi;
        else
          chi = mat.calculate_chi(spectrum);

      }
      else
      {
        // assign to dummy value to work with cacheing later
        for (double &x : spectrum)
          x = -1.0;
      }

      Vector val(mat_lib.ngroup());
      val = 0.0;

      // fission
      if (rec.is_fiss)
      {

        double fiss_sum{0.0};
        for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
          fiss_sum += nusf[gprime]*spectrum[gprime];

        for (int g{0}; g < mat_lib.ngroup(); ++g)
          val(g) += chi[g]*fiss_sum;

        val /= keff;

      }

      // upscatter
      for (int g{0}; g < mat_lib.ngroup(); ++g)
      {
        if (rec.is_upscatter[g])
        {

          Array<int> col_idx;
          Vector scat_dat;
          scat.GetRow(g, col_idx, scat_dat);

          for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
          {
            const int gprime{col_idx[gidx]};
            if (gprime > g)
            {
              if (spectrum[gprime] < 0.0)
              {
                Vector locvec(dofs.Size());
                scalar_flux(gprime).GetSubVector(dofs, locvec);
                spectrum[gprime] = dofval*locvec;
              }
              val(g) += scat_dat(gidx)*spectrum[gprime];
            }
          }
        }
      }

      // val array is calculated
      // need to incorporate into quadrature functions
      for (int n{0}; n < quad->nd(); ++n)
      {
        for (int g{0}; g < mat_lib.ngroup(); ++g)
          fiss_upscatter_quad[g](idx) = val(g);
        ++idx;
      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir(quad->omega(n));
        for (int d{0}; d < quad->dim(); ++d)
        {
          for (int g{0}; g < mat_lib.ngroup(); ++g)
            saaf_fiss_upscatter_quad[g](saaf_idx) = dir(d)/tot[g] * val(g);
          ++saaf_idx;
        }
      }

    }

  }

  return;
}

void calc_downscatter_vector(const int g,
                             const Result &scalar_flux,
                             const Material_library &mat_lib,
                             const Quadrature * const quad,
                             QuadratureFunction &dnscatter_quad,
                             QuadratureFunction &saaf_dnscatter_quad)
{

  const QuadratureSpace* const qspace{dnscatter_quad.GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux(0).ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  dnscatter_quad = 0.0;
  saaf_dnscatter_quad = 0.0;

  int idx{0};
  int saaf_idx{0};

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

      // get all xs at this point
      // this will eventually be temperature dependent
      const double &tot{rec.sigma_t[g]};
      const SparseMatrix &scat{rec.sigma_s};

      Array<int> col_idx;
      Vector scat_dat;
      scat.GetRow(g, col_idx, scat_dat);

      double val{0.0};

      // assume column indices are sorted
      for (int gidx{0}; gidx < col_idx.Size(); ++gidx)
      {

        const int gprime{col_idx[gidx]};
        if (gprime >= g) break;

        Vector locvec(dofs.Size());
        scalar_flux(gprime).GetSubVector(dofs, locvec);
        const double flux{dofval*locvec};

        val += scat_dat(gidx)*flux;

      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        dnscatter_quad(idx) = val;
        ++idx;
      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir{quad->omega(n)};
        for (int d{0}; d < quad->dim(); ++d)
        {
          saaf_dnscatter_quad(saaf_idx) = dir(d)/tot * val;
          ++saaf_idx;
        }
      }

    }

  }

  return;
}

void calc_selfscatter_vector(const int g,
                             const mfem::ParGridFunction &scalar_flux,
                             const Material_library &mat_lib,
                             const Quadrature * const quad,
                             QuadratureFunction &selfscatter_quad,
                             QuadratureFunction &saaf_selfscatter_quad)
{

  const QuadratureSpace* const qspace{selfscatter_quad.GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux.ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  selfscatter_quad = 0.0;
  saaf_selfscatter_quad = 0.0;

  int idx{0};
  int saaf_idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record& mat{mat_lib.mat[attribute]};
    const XSRecord& rec{mat.rec};

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint ip{ir.IntPoint(i)};

      // get all xs at this point
      // this will eventually be temperautre dependent
      const double scat{rec.sigma_s.Elem(g,g)};
      const double tot{rec.sigma_t[g]};

      const double val{scat*scalar_flux.GetValue(e, ip)};

      for (int n{0}; n < quad->nd(); ++n)
      {
        selfscatter_quad(idx) = val;
        ++idx;
      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir{quad->omega(n)};
        for (int d{0}; d < quad->dim(); ++d)
        {
          saaf_selfscatter_quad(saaf_idx) = dir(d)/tot * val;
          ++saaf_idx;
        }
      }

    }
  }

  return;
}
