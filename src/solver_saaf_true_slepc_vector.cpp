#include "solver_saaf_true_slepc_vector.hpp"

#include "memory.hpp"

#include "../addon/tensor.hpp"
#include "../addon/mfem_addon.hpp"

#include "vector_pwconst_coefficient.hpp"
#include "matrix_pwconst_coefficient.hpp"

#ifdef PETSC_AVAILABLE
#include "../mfem/linalg/petscinternals.hpp" // PCHKERRQ
#include "slepceps.h"
#endif

#include <cmath>
#include <string>
#include <vector>

using namespace mfem;

#if __cplusplus >= 201703L
#include <filesystem>
using std::filesystem::exists; // requires C++17
#else
#include <fstream>
namespace
{
bool exists(const std::string& fname)
{
  const std::ifstream ifs{fname};
  return ifs.good();
}
} // namespace anonymous
#endif

void saaf_true_slepc_vector(const Quadrature* const quad,
                            Timer& time,
                            const Input& inp,
                            const Material_library& mat_lib,
                            Result& angular_flux)
{

#ifdef PETSC_AVAILABLE

  ParFiniteElementSpace* const fespace{angular_flux(0).ParFESpace()};

  const int myid{fespace->GetMyRank()};

  time.start(time.matrix_assembly);

  time.start("Amat");
  std::unique_ptr<HypreParMatrix> Ahypre{saaf_a_matrix(inp, time, mat_lib, quad, fespace)};
  time.stop("Amat");
  time.start("Mmat");
  std::unique_ptr<HypreParMatrix> Mhypre{saaf_m_matrix(inp, time, mat_lib, quad, fespace)};
  time.stop("Mmat");

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
  slepc_solver_wrapper(Ahypre.release(), Mhypre.release(), inp, time, angular_flux, mat_lib.ngroup()*quad->nd());
  if (myid == 0)
  {
    mem->stop_background();
    mem.reset();
  }
  slepc_finalize();

#else
  MFEM_ABORT("MEZCAL must be built with PETSc & SLEPc!");
#endif

  return;
}

namespace
{

namespace slepc_amgx
{

  std::unique_ptr<Solver> sol{nullptr};
  HypreParMatrix *Ahypre{nullptr};

  bool verbose{true};
  int ndevices{0};

#ifdef PETSC_AVAILABLE
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

#endif

} // namespace slepc_amgx

namespace slepc_hypre
{

  std::unique_ptr<Solver> sol{nullptr};
  std::unique_ptr<HypreSolver> prec{nullptr};
  const HypreParMatrix *Ahypre{nullptr};

  const Input *inp{nullptr};
  const Material_library *mat_lib{nullptr};
  const Quadrature *quad{nullptr};

#ifdef PETSC_AVAILABLE
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
#endif

} // namespace slepc_hypre

mfem::HypreParMatrix* saaf_a_matrix(const Input &inp, 
                                    Timer &time,
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
    const XSRecord& rec{mat_lib.mat[i].rec[0]};
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
    const XSRecord& record{mat_lib.mat[i].rec[0]};

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

  time.start("a_assemble");
  sink_bilinear.Assemble(skip_zeros);
  time.stop("a_assemble");

  time.start("a_finalize");
  sink_bilinear.Finalize(skip_zeros); // no essential boundary conditions (maybe)
  time.stop("a_finalize");

  time.start("a_parallel_assemble");
  HypreParMatrix *A = sink_bilinear.ParallelAssemble();
  time.stop("a_parallel_assemble");
  return A;

}

HypreParMatrix* saaf_m_matrix(const Input &inp,
                              Timer &time,
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
      const double tot{mat.rec[0].sigma_t[g]};
      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {

        double val{0.0};
        for (int iso_idx{0}; iso_idx < mat.niso; ++iso_idx)
        {
          const Material_isotope& isotope{mat.iso[iso_idx]};
          const XSRecord& record{*isotope.rec[0]};
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

  time.start("m_assemble");
  fiss_bilinear.Assemble(skip_zeros);
  time.stop("m_assemble");

  time.start("m_finalize");
  fiss_bilinear.Finalize(skip_zeros);
  time.stop("m_finalize");

  time.start("m_parallel_assemble");
  HypreParMatrix *M = fiss_bilinear.ParallelAssemble();
  time.stop("m_parallel_assemble");
  return M;

}

void slepc_initialize(const Input& inp)
{

#ifdef PETSC_AVAILABLE

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

#else
  MFEM_ABORT("MEZCAL must be built with PETSc & SLEPc!");
#endif

  return;
}

void slepc_solver_wrapper(HypreParMatrix* Ahypre,
                          HypreParMatrix* Mhypre,
                          const Input& inp,
                          Timer& time,
                          Result& res,
                          int numfunctions)
{

#ifdef PETSC_AVAILABLE

  //constexpr Operator::Type petsc_format{Operator::PETSC_MATAIJ};
  constexpr Operator::Type petsc_format{Operator::PETSC_MATHYPRE};

  const PetscParMatrix A{Ahypre, petsc_format};
  const PetscParMatrix M{Mhypre, petsc_format};

  /*
  // This seems incompatible with the GMRES setup...
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

#else
  MFEM_ABORT("MEZCAL must be built with PETSc & SLEPc!");
#endif

  return;
}

void slepc_finalize()
{
#ifdef PETSC_AVAILABLE
  MFEMFinalizeSlepc();
  return;
#else
  MFEM_ABORT("MEZCAL must be built with PETSc & SLEPc!");
#endif
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

} // namespace anonymous
