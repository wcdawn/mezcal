#include "solver_saaf_angle_power_iteration.hpp"

#include "make_unique.hpp"

#include "../addon/mfem_addon.hpp"
#include "../addon/tensor.hpp"

#include "vector_pwconst_coefficient.hpp"
#include "vector_incoming_flux_coefficient.hpp"

#include <cmath>
#include <vector>

using namespace mfem;

namespace 
{
  double par_norml2(const Vector &x)
  {
    const double local_norm{std::pow(x.Norml2(),2)};
    double norm{0.0};
    MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(norm);
  }
}

template<class T> char * as_bytes(T &i)
{
  void *addr = &i;
  return static_cast<char*>(addr);
}

template<class T> const char * as_bytes(const T &i)
{
  const void *addr = &i;
  return static_cast<const char*>(addr);
}

void saaf_angle_power_iteration(const Quadrature* const quad,
                                Timer& time,
                                Input& inp,
                                const Material_library& mat_lib,
                                std::vector<std::unique_ptr<ParGridFunction>> &angular_flux,
                                std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux,
                                double &keff,
                                ParGridFunction &power,
                                ParGridFunction &temperature)
{

  ParFiniteElementSpace *scalar_flux_fespace = scalar_flux[0]->ParFESpace();
  const int myid{scalar_flux_fespace->GetMyRank()};
  ParMesh *pmesh = scalar_flux_fespace->GetParMesh();

  if (inp.restart.flux)
  {

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      std::unique_ptr<char[]> fname{std::make_unique<char[]>(inp.input_file_stub().size()+26)};
      sprintf(fname.get(), "%s_restart_g%04d_%06d.sol", inp.input_file_stub().c_str(), g, myid);

      std::ifstream ifs{fname.get(), std::ios_base::binary};
      int af_size;
      ifs.read(as_bytes(af_size), sizeof(int));
      MFEM_VERIFY(af_size == angular_flux[g]->Size(), "inconsistent restart size");
      double *af = angular_flux[g]->HostWrite();
      for (int i{0}; i < angular_flux[g]->Size(); ++i)
        ifs.read(as_bytes(af[i]), sizeof(double));

      calc_scalar_flux_vector(*angular_flux[g], quad, *scalar_flux[g]);
    }

    const std::string keff_fname{inp.input_file_stub() + "_restart_keff.out"};
    std::ifstream keff_ifs{keff_fname};
    while (keff_ifs.good())
      keff_ifs >> keff;

  }
  else
  {

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      *angular_flux[g] = 1.0;
      *scalar_flux[g] = 1.0;
    }

    keff = 1.0;

  }

  ParFiniteElementSpace *angular_flux_fespace = angular_flux[0]->ParFESpace();

  constexpr int skip_zeros{2};

  enum class LINEAR_SOLVER
  {
    invalid=-1,
    amgx,
    hypre_pcg,
    hypre_gmres,
    mfem_cg,
    mfem_gmres,
    mfem_minres,
    mfem_bicgstab
  };
  // Choose which solver will be used for the SAAF system.
  // Except for AmgX, HypreBoomerAMG will always be used for preconditioner.
  // DSA system will always be solved with HypreBoomerAMG+HyprePCG.
  LINEAR_SOLVER linear_solver{LINEAR_SOLVER::invalid};
  const std::string inp_linear_solver{inp.get_linear_solver()};
  if (inp_linear_solver == "amgx")
    linear_solver = LINEAR_SOLVER::amgx;
  else if (inp_linear_solver == "hypre_pcg")
    linear_solver = LINEAR_SOLVER::hypre_pcg;
  else if (inp_linear_solver == "hypre_gmres")
    linear_solver = LINEAR_SOLVER::hypre_gmres;
  else if (inp_linear_solver == "mfem_cg")
    linear_solver = LINEAR_SOLVER::mfem_cg;
  else if (inp_linear_solver == "mfem_gmres")
    linear_solver = LINEAR_SOLVER::mfem_gmres;
  else if (inp_linear_solver == "mfem_minres")
    linear_solver = LINEAR_SOLVER::mfem_minres;
  else if (inp_linear_solver == "mfem_bicgstab")
    linear_solver = LINEAR_SOLVER::mfem_bicgstab;
  else
  {
    if (myid == 0) std::cout << "WARNING: unknown (or no) linear solver specified = *" << inp_linear_solver << "*" << std::endl;
    if (myid == 0) std::cout << "Using default (hypre_pcg) linear solver." << std::endl;
    linear_solver = LINEAR_SOLVER::hypre_pcg;
  }

  enum class LINEAR_PRECONDITIONER
  {
    invalid=-1,
    amg,
    jacobi
  };
  LINEAR_PRECONDITIONER linear_preconditioner{LINEAR_PRECONDITIONER::invalid};
  const std::string inp_linear_preconditioner{inp.get_linear_preconditioner()};
  if (inp_linear_preconditioner == "amg")
    linear_preconditioner = LINEAR_PRECONDITIONER::amg;
  else if (inp_linear_preconditioner == "jacobi")
    linear_preconditioner = LINEAR_PRECONDITIONER::jacobi;
  else
  {
    if (myid == 0) std::cout << "WARNING: unknown (or no) linear preconditioner specified = *" << inp_linear_preconditioner << "*" << std::endl;
    if (myid == 0) std::cout << "Using default (amg) linear preconditioner." << std::endl;
    linear_preconditioner = LINEAR_PRECONDITIONER::amg;
  }

  // transport streaming+collision operator
  std::vector<std::unique_ptr<HypreParMatrix>> Amat(mat_lib.ngroup());
  std::vector<std::unique_ptr<HypreSolver>> Aprec(mat_lib.ngroup());
  std::vector<std::unique_ptr<Solver>> Asol(mat_lib.ngroup());

  // dsa streaming+removal operator
  std::vector<std::unique_ptr<HypreParMatrix>> Dmat(mat_lib.ngroup());
  std::vector<std::unique_ptr<HypreSolver>> Dprec(mat_lib.ngroup());
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

      // precconditioner
      // everything uses a hypre preconditioenr except for amgx
      if (linear_solver != LINEAR_SOLVER::amgx)
      {
        switch (linear_preconditioner)
        {
          case (LINEAR_PRECONDITIONER::amg):
          {
            std::unique_ptr<HypreBoomerAMG> prec{std::make_unique<HypreBoomerAMG>(*Amat[g])};
            if (inp.verbose)
              prec->SetPrintLevel(1);
            else
              prec->SetPrintLevel(0);
            HYPRE_BoomerAMGSetNumFunctions(*prec, quad->nd());
            Aprec[g] = std::move(prec);
            break;
          }
          case (LINEAR_PRECONDITIONER::jacobi):
          {
            std::unique_ptr<HypreDiagScale> prec{std::make_unique<HypreDiagScale>(*Amat[g])};
            Aprec[g] = std::move(prec);
            break;
          }
          default:
            if (myid == 0) std::cout << "invalid linear preconditioner" << std::endl;
            return;
        }
      }

      // tolver
      switch (linear_solver)
      {
        case (LINEAR_SOLVER::amgx):
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

          break;
        }
        case (LINEAR_SOLVER::hypre_pcg):
        {
          std::unique_ptr<HyprePCG> this_sol{std::make_unique<HyprePCG>(MPI_COMM_WORLD)};
          if (inp.verbose)
            this_sol->SetPrintLevel(2);
          else
            this_sol->SetPrintLevel(0);
          this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
          this_sol->SetOperator(*Amat[g]);
          this_sol->SetTol(inp.tol.inner_rel);
          this_sol->SetAbsTol(inp.tol.inner_abs);
          this_sol->SetMaxIter(inp.tol.max_inner);
          Asol[g] = std::move(this_sol);
          break;
        }
        case (LINEAR_SOLVER::hypre_gmres):
        {
          std::unique_ptr<HypreGMRES> this_sol{std::make_unique<HypreGMRES>(MPI_COMM_WORLD)};
          this_sol->SetPrintLevel(2);
          this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
          this_sol->SetOperator(*Amat[g]);
          this_sol->SetTol(inp.tol.inner_rel);
          this_sol->SetAbsTol(inp.tol.inner_abs);
          this_sol->SetMaxIter(inp.tol.max_inner);
          this_sol->SetKDim(5); // Very small. Shouldn't be necessary for SPD systems.
          Asol[g] = std::move(this_sol);
          break;
        }
        case (LINEAR_SOLVER::mfem_cg):
        {
          std::unique_ptr<CGSolver> this_sol{std::make_unique<CGSolver>(MPI_COMM_WORLD)};
          this_sol->SetPrintLevel(1);
          this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
          this_sol->SetOperator(*Amat[g]);
          this_sol->SetRelTol(inp.tol.inner_rel);
          this_sol->SetAbsTol(inp.tol.inner_abs);
          this_sol->SetMaxIter(inp.tol.max_inner);
          Asol[g] = std::move(this_sol);
          break;
        }
        case (LINEAR_SOLVER::mfem_gmres):
        {
          std::unique_ptr<GMRESSolver> this_sol{std::make_unique<GMRESSolver>(MPI_COMM_WORLD)};
          this_sol->SetPrintLevel(1);
          this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
          this_sol->SetOperator(*Amat[g]);
          this_sol->SetRelTol(inp.tol.inner_rel);
          this_sol->SetAbsTol(inp.tol.inner_abs);
          this_sol->SetMaxIter(inp.tol.max_inner);
          this_sol->SetKDim(5); // Very small. Shouldn't be necessary for SPD systems.
          Asol[g] = std::move(this_sol);
          break;
        }
        case (LINEAR_SOLVER::mfem_minres):
        {
          std::unique_ptr<MINRESSolver> this_sol{std::make_unique<MINRESSolver>(MPI_COMM_WORLD)};
          this_sol->SetPrintLevel(1);
          this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
          this_sol->SetOperator(*Amat[g]);
          this_sol->SetRelTol(inp.tol.inner_rel);
          this_sol->SetAbsTol(inp.tol.inner_abs);
          this_sol->SetMaxIter(inp.tol.max_inner);
          Asol[g] = std::move(this_sol);
          break;
        }
        case (LINEAR_SOLVER::mfem_bicgstab):
        {
          std::unique_ptr<BiCGSTABSolver> this_sol{std::make_unique<BiCGSTABSolver>(MPI_COMM_WORLD)};
          this_sol->SetPrintLevel(1);
          this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
          this_sol->SetOperator(*Amat[g]);
          this_sol->SetRelTol(inp.tol.inner_rel);
          this_sol->SetAbsTol(inp.tol.inner_abs);
          this_sol->SetMaxIter(inp.tol.max_inner);
          Asol[g] = std::move(this_sol);
          break;
        }
        default:
          if (myid==0) std::cout << "unsupported LINEAR_SOLVER" << std::endl;
          return;
      }

    } // g = 1,mat_lib.ngroup()

  } // streaming+collision

  if (inp.tol.use_dsa)
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

      if (linear_solver == LINEAR_SOLVER::amgx)
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

        switch (linear_preconditioner)
        {
          case (LINEAR_PRECONDITIONER::amg):
          {
            std::unique_ptr<HypreBoomerAMG> prec{std::make_unique<HypreBoomerAMG>(*Dmat[g])};
            prec->SetPrintLevel(0);
            Dprec[g] = std::move(prec);
            break;
          }
          case (LINEAR_PRECONDITIONER::jacobi):
          {
            std::unique_ptr<HypreDiagScale> prec{std::make_unique<HypreDiagScale>(*Dmat[g])};
            Dprec[g] = std::move(prec);
            break;
          }
          default:
            if (myid == 0) std::cout << "invalid linear preconditioner" << std::endl;
            return;
        }

        std::unique_ptr<HyprePCG> this_sol{std::make_unique<HyprePCG>(MPI_COMM_WORLD)};
        this_sol->SetPreconditioner(*Dprec[g]);
        this_sol->SetOperator(*Dmat[g]);
        this_sol->SetTol(inp.tol.inner_rel);
        this_sol->SetAbsTol(inp.tol.inner_abs);
        this_sol->SetMaxIter(inp.tol.max_inner);
        Dsol[g] = std::move(this_sol);

      }

    } // g=1,mat_lib.ngroup()

  } // DSA operator

  time.stop(time.matrix_assembly);

  const int integration_order{scalar_flux_fespace->GetOrder(0)+1};
  QuadratureSpace qspace{pmesh, integration_order};

  QuadratureFunction power_quad{&qspace};

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

  VectorQuadratureFunctionCoefficient src_quad_func{selfscatter_quad}, saaf_src_quad_func{saaf_selfscatter_quad};

  std::vector<VectorIncomingFluxCoefficient> vector_incoming_flux_coeff;
  vector_incoming_flux_coeff.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    vector_incoming_flux_coeff.emplace_back(*angular_flux[g], *quad);

  Tensor<Vector> tensor_omega;
  tensor_omega.reserve(quad->nd());
  for (int n{0}; n < quad->nd(); ++n)
    tensor_omega.AddElement(quad->omega(n), n);
  VectorTensorConstantCoefficient omega_term{quad->nd(), tensor_omega};

  Array<int> reflective_condition(pmesh->bdr_attributes.Max());
  reflective_condition = 0;
  for (int i{0}; (i < inp.boundary_conditions.Size()) && (i < pmesh->bdr_attributes.Max()); ++i)
  {
    if (inp.boundary_conditions[i] == 0) // zero represents reflective
      reflective_condition[i] = 1;
  }

  std::vector<std::unique_ptr<ParLinearForm>> lf_self_scatter(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    lf_self_scatter[g] = std::make_unique<ParLinearForm>(angular_flux_fespace);
    lf_self_scatter[g]->AddDomainIntegrator(new VectorDomainLFIntegrator{src_quad_func});
    lf_self_scatter[g]->AddDomainIntegrator(new VectorDomainLFGradIntegrator{saaf_src_quad_func});
    if (inp.use_full_assembly)
    {
      MFEM_VERIFY(reflective_condition.Max() == 0,
        "VectorBoundaryFlowIntegrator (required for reflective bcs) is "
        "incompatible with LinearAssemblyLevel::FULL.");
      lf_self_scatter[g]->SetAssemblyLevel(LinearAssemblyLevel::FULL);
    }
    else
    {
      lf_self_scatter[g]->AddBdrFaceIntegrator(
        new VectorBoundaryFlowIntegrator{vector_incoming_flux_coeff[g], omega_term, -1.0},
        reflective_condition);
    }
  }

////////////////////////////////////////////////////////////////////////////////
  // DSA rhs

  std::vector<GridFunctionCoefficient> scalar_flux_func; // (k+1/2)
  scalar_flux_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    scalar_flux_func.emplace_back(scalar_flux[g].get());

  ParGridFunction dsa_phi_old{scalar_flux_fespace}; // (k)
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
  if (inp.tol.use_dsa)
  {
    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {
      lf_dsa[g] = std::make_unique<ParLinearForm>(scalar_flux_fespace);
      lf_dsa[g]->AddDomainIntegrator(new DomainLFIntegrator{dsa_src[g]});
      if (inp.use_full_assembly) lf_dsa[g]->SetAssemblyLevel(LinearAssemblyLevel::FULL);
    }
  }

////////////////////////////////////////////////////////////////////////////////

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    angular_flux[g]->UseDevice(true);
    scalar_flux[g]->UseDevice(true);
    lf_self_scatter[g]->UseDevice(true);
    if (inp.tol.use_dsa) lf_dsa[g]->UseDevice(true);

    fiss_upscatter_quad[g].UseDevice(true);
    saaf_fiss_upscatter_quad[g].UseDevice(true);
  }

  dsa_phi_old.UseDevice(true);
  dnscatter_quad.UseDevice(true);
  saaf_dnscatter_quad.UseDevice(true);
  selfscatter_quad.UseDevice(true);
  saaf_selfscatter_quad.UseDevice(true);

////////////////////////////////////////////////////////////////////////////////

  QuadratureFunction fiss_src_quad_old{&qspace};
  fiss_src_quad_old.UseDevice(true);
  fiss_src_quad_old = 1.0;
  double k_conv{inp.tol.k * 1e1 + 1e-6};
  double phi_conv{inp.tol.phi * 1e1 + 1e-6};
  double keff_old{keff};
  double fiss_sum_old{1.0};
  int solve_counter{0};

  std::unique_ptr<std::ofstream> keff_ofs;
  if (inp.restart.out)
  {
    const std::string keff_fname{inp.input_file_stub() + "_restart_keff.out"};
    keff_ofs = std::make_unique<std::ofstream>(keff_fname);
    *keff_ofs << std::setprecision(20);
  }

  time.start(time.solver);
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    time.start("power");
    calc_power_vector(mat_lib, scalar_flux, inp.rxpwr, power_quad);
    time.stop("power");

#if 0
    time.start("temperature");
    temperature_heat_conduction(inp, mat_lib, power, temperature);
    time.stop("temperature");
#endif

    time.start("calc_fiss_upscatter_vector");
    calc_fiss_upscatter_vector(scalar_flux, mat_lib, quad,
      fiss_upscatter_quad, saaf_fiss_upscatter_quad,
      keff, inp.flux_free_chi);
    time.stop("calc_fiss_upscatter_vector");

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      if (myid == 0) std::cout << "BEGIN_GROUP " << g << std::endl;

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

        dsa_phi_old = *scalar_flux[g]; // (k)

        time.start("calc_selfscatter_vector");
        calc_selfscatter_vector(g, *scalar_flux[g], mat_lib, quad,
          selfscatter_quad, saaf_selfscatter_quad);
        time.stop("calc_selfscatter_vector");
        time.start("calc_selfscatter_vector_sum");
        selfscatter_quad += dnscatter_quad;
        saaf_selfscatter_quad += saaf_dnscatter_quad;
        time.stop("calc_selfscatter_vector_sum");

        time.start("b_assemble");
        lf_self_scatter[g]->Assemble();
        std::unique_ptr<HypreParVector> b{lf_self_scatter[g]->ParallelAssemble()};
        b->UseDevice(true);
        time.stop("b_assemble");

        time.start("Amult");
        std::unique_ptr<HypreParVector> x{angular_flux[g]->ParallelAssemble()};
        x->UseDevice(true);
        Asol[g]->Mult(*b, *x);
        angular_flux[g]->Distribute(*x);
        time.stop("Amult");

        time.start("scalar_flux");
        calc_scalar_flux_vector(*angular_flux[g], quad, *scalar_flux[g]); // (k+1/2)
        time.start("scalar_flux");

        if (inp.tol.use_dsa)
        {

          ParGridFunction pgf_dphi{*scalar_flux[g]};
          pgf_dphi.UseDevice(true);

          time.start("b_dsa_assemble");
          lf_dsa[g]->Assemble();
          std::unique_ptr<HypreParVector> f{lf_dsa[g]->ParallelAssemble()};
          std::unique_ptr<HypreParVector> dphi{pgf_dphi.ParallelAssemble()};
          f->UseDevice(true);
          dphi->UseDevice(true);
          time.stop("b_dsa_assemble");

          time.start("Dmult");
          Dsol[g]->Mult(*f, *dphi);
          pgf_dphi.Distribute(*dphi);
          time.stop("Dmult");

          *scalar_flux[g] += pgf_dphi; // (k+1)

        }

        ++solve_counter;

        dsa_phi_old -= *scalar_flux[g];

        scattering_conv = par_norml2(dsa_phi_old)/par_norml2(*scalar_flux[g]);

        if (myid == 0)
          std::cout << "..."
                    << scattering_iter
                    << " g=" << g
                    << " conv=" << scattering_conv
                    << '\n';

      } // scattering_iter

      // check the clock and decide if we should cleanup
      if ((inp.restart.out) && (time.elapsed()/60. > inp.restart.runtime))
      {
        if (myid == 0)
        {
          std::cout << "RESTART CLEANUP\n"
                    << "time_elapsed= " << time.elapsed()/60. << "\n"
                    << "restart_runtime= " << inp.restart.runtime << "\n"
                    << std::endl;
        }
        time.start("restart_out");
        std::unique_ptr<char[]> fname{std::make_unique<char[]>(inp.input_file_stub().size()+26)};
        for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
        {
          sprintf(fname.get(), "%s_restart_g%04d_%06d.sol", inp.input_file_stub().c_str(), gprime, myid);

          const int size = angular_flux[gprime]->Size();
          std::ofstream ofs{fname.get(), std::ios_base::binary};
          ofs.write(as_bytes(size), sizeof(int));
          const double *af = angular_flux[gprime]->HostRead();
          for (int i{0}; i < size; ++i)
            ofs.write(as_bytes(af[i]), sizeof(double));
        }

        time.stop("restart_out");

        QuadratureFunctionCoefficient gfc{power_quad};
        power.ProjectCoefficient(gfc);

        return; // leave here and cleanup
      }

    } // ngroup

    time.start("fisssum");
    QuadratureFunction fiss_src_quad{&qspace};
    fiss_src_quad.UseDevice(true);
    calc_fiss_src(scalar_flux, mat_lib, fiss_src_quad);
    const double local_fiss_sum{fiss_src_quad.Sum()};
    double fiss_sum;
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    time.stop("fisssum");

    time.start("phiconv");
    fiss_src_quad_old -= fiss_src_quad;
    phi_conv = par_norml2(fiss_src_quad_old)/par_norml2(fiss_src_quad);
    time.stop("phiconv");

    if (iter > 0) keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(keff - keff_old);

    if (myid == 0)
    {
      std::cout << iter
                << " keff=" << keff
                << " dk=" << k_conv
                << " dphi=" << phi_conv
                << std::endl;
      if (inp.restart.out) *keff_ofs << keff << std::endl;
    }

    if ((keff < 0.0) || (keff > 20.0) || std::isnan(keff))
    {
      if (myid == 0) std::cout << "INVALID VALUE FOR KEFF" << std::endl;
      return;
    }

    // store old values
    keff_old = keff;
    fiss_sum_old = fiss_sum;
    fiss_src_quad_old = fiss_src_quad;

  }
  time.stop(time.solver);

  if (myid == 0)
  {
    std::cout << "ITERATIONS FINISHED!!!" << std::endl;
    std::cout << "solve_counter= " << solve_counter << std::endl;
  }

  QuadratureFunctionCoefficient gfc{power_quad};
  power.ProjectCoefficient(gfc);


  return;
}

void calc_fiss_upscatter_vector(const std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux,
                                const Material_library &mat_lib,
                                const Quadrature * const quad,
                                std::vector<QuadratureFunction> &fiss_upscatter_quad,
                                std::vector<QuadratureFunction> &saaf_fiss_upscatter_quad,
                                const double keff,
                                const bool flux_free_chi)
{

  const QuadratureSpace* const qspace{fiss_upscatter_quad[0].GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux[0]->ParFESpace()};
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

    const IntegrationRule &ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record &mat{mat_lib.mat[attribute]};
    const XSRecord &rec{mat.rec};

    const FiniteElement* const fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip{ir.IntPoint(i)};
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
          scalar_flux[g]->GetSubVector(dofs, locvec.HostWrite());
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
          val(g) = chi[g]*fiss_sum;

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
                scalar_flux[gprime]->GetSubVector(dofs, locvec.HostWrite());
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
        {
          double *fquad = fiss_upscatter_quad[g].HostWrite();
          const double *vval = val.HostRead();
          fquad[idx] = vval[g];
        }
        ++idx;
      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir(quad->omega(n));
        for (int d{0}; d < quad->dim(); ++d)
        {
          for (int g{0}; g < mat_lib.ngroup(); ++g)
          {
            double *fquad = saaf_fiss_upscatter_quad[g].HostWrite();
            const double *ddir = dir.HostRead();
            const double *vval = val.HostRead();
            fquad[saaf_idx] = ddir[d]/tot[g] * vval[g];
          }
          ++saaf_idx;
        }
      }

    }

  }

  return;
}

void calc_downscatter_vector(const int g,
                             const std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux,
                             const Material_library &mat_lib,
                             const Quadrature * const quad,
                             QuadratureFunction &dnscatter_quad,
                             QuadratureFunction &saaf_dnscatter_quad)
{

  const QuadratureSpace* const qspace{dnscatter_quad.GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux[0]->ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  dnscatter_quad = 0.0;
  saaf_dnscatter_quad = 0.0;

  double *dnquad = dnscatter_quad.HostWrite();
  double *saaf_dnquad = saaf_dnscatter_quad.HostWrite();

  int idx{0};
  int saaf_idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record &mat{mat_lib.mat[attribute]};
    const XSRecord &rec{mat.rec};

    const FiniteElement* const fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip{ir.IntPoint(i)};
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
        scalar_flux[gprime]->GetSubVector(dofs, locvec.HostWrite());
        const double flux{dofval*locvec};

        val += scat_dat(gidx)*flux;

      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        dnquad[idx] = val;
        ++idx;
      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir{quad->omega(n)};
        for (int d{0}; d < quad->dim(); ++d)
        {
          const double *ddir = dir.HostRead();
          saaf_dnquad[saaf_idx] = ddir[d]/tot * val;
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

  double *selfquad = selfscatter_quad.HostWrite();
  double *saaf_selfquad = saaf_selfscatter_quad.HostWrite();

  int idx{0};
  int saaf_idx{0};

  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record &mat{mat_lib.mat[attribute]};
    const XSRecord &rec{mat.rec};

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip{ir.IntPoint(i)};

      // get all xs at this point
      // this will eventually be temperautre dependent
      const double scat{rec.sigma_s.Elem(g,g)};
      const double tot{rec.sigma_t[g]};
      const double flux{scalar_flux.GetValue(e, ip)};
      const double val{scat*flux};

      for (int n{0}; n < quad->nd(); ++n)
      {
        selfquad[idx] = val;
        ++idx;
      }

      for (int n{0}; n < quad->nd(); ++n)
      {
        const Vector &dir{quad->omega(n)};
        for (int d{0}; d < quad->dim(); ++d)
        {
          const double *ddir = dir.HostRead();
          saaf_selfquad[saaf_idx] = ddir[d]/tot * val;
          ++saaf_idx;
        }
      }

    }
  }

  return;
}

void calc_scalar_flux_vector(const ParGridFunction &angular_flux,
                             const Quadrature * const quad, 
                             ParGridFunction &scalar_flux)
{

  ParFiniteElementSpace* fespace{angular_flux.ParFESpace()};

  MFEM_VERIFY(fespace->GetOrdering() == Ordering::byVDIM,
    "scalar flux calculation relies on angular flux being ordered by vdim");

  const int ndofs{fespace->GetNDofs()};
  const int vdim{quad->nd()};

  double *sf = scalar_flux.HostWrite();
  const double *af = angular_flux.HostRead();

  for (int i{0}; i < ndofs; ++i)
  {
    sf[i] = 0.0;
    for (int n{0}; n < quad->nd(); ++n)
      sf[i] += af[n+i*vdim] * quad->weight(n);
  }

  return;
}

/* 
 * WARNING
 * This is *NO LONGER A DUPLICATE* of the function of the same name from in solver.[ch]pp.
 * WARNING
 */
void calc_fiss_src(const std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux, 
                   const Material_library &mat_lib,
                   QuadratureFunction &fiss_src)
{

  const QuadratureSpace* const qspace{fiss_src.GetSpace()};
  const ParFiniteElementSpace* const fespace{scalar_flux[0]->ParFESpace()};
  const ParMesh* const mesh{fespace->GetParMesh()};

  fiss_src = 0.0;
  double *fiss = fiss_src.HostReadWrite();

  int idx{0};
  for (int e{0}; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir{qspace->GetElementIntRule(e)};
    const int attribute{mesh->GetAttribute(e)-1};
    const Material_record &mat{mat_lib.mat[attribute]};
    const XSRecord &rec{mat.rec};

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

      const IntegrationPoint &ip{ir.IntPoint(i)};
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        Vector locvec(dofs.Size());
        scalar_flux[gprime]->GetSubVector(dofs, locvec.HostWrite());
        const double flux{dofval*locvec};
        fiss[idx] += nusf[gprime]*flux;
      }

      ++idx;
    }

  }

  return;
}

void calc_power_vector(const Material_library &mat_lib,
                       const std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux,
                       const double rxpwr,
                       QuadratureFunction &power_quad)
{

  constexpr double kappa_over_nu = 1.0; // TODO this should be group dependent
  // TODO power normalization

  const QuadratureSpace* const qspace = power_quad.GetSpace();
  const ParFiniteElementSpace* const fespace = scalar_flux[0]->ParFESpace();
  const ParMesh* const mesh = fespace->GetParMesh();

  power_quad = 0.0; // [W/cc]
  double *pquad = power_quad.HostReadWrite();

  int idx = 0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = mesh->GetAttribute(e) - 1;
    const XSRecord &rec = mat_lib.mat[attribute].rec;

    if (!rec.is_fiss)
    {
      idx += ir.GetNPoints();
      continue;
    }

    std::vector<double> kappasf = rec.nusf; // copy
    for (int g = 0; g < mat_lib.ngroup(); ++g)
      kappasf[g] *= kappa_over_nu;

    const FiniteElement *fe = fespace->GetFE(e);
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip = ir.IntPoint(i);
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      for (int gprime = 0; gprime < mat_lib.ngroup(); ++gprime)
      {
        Vector locvec(dofs.Size());
        scalar_flux[gprime]->GetSubVector(dofs, locvec.HostWrite());
        const double flux = dofval*locvec;
        pquad[idx] += kappasf[gprime]*flux;
      }

      ++idx;

    }

  }

  return;
}
