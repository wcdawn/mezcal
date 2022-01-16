#include "solver_saaf_angle_power_iteration.hpp"

#include "make_unique.hpp"

#include "../addon/mfem_addon.hpp"
#include "../addon/tensor.hpp"

#include "vector_pwconst_coefficient.hpp"
#include "vector_incoming_flux_coefficient.hpp"

#include "conduction.hpp"

#include <cmath>
#include <vector>

using namespace mfem;

// note: the function signature may need to change
double xscalc_sigma_t(const int g, const XSRecord &rec, const double T)
{
  return rec.sigma_t[g];
}

double xscalc_selfscatter(const int g, const XSRecord &rec, const double T)
{
  return rec.sigma_s.Elem(g,g);
}

double xscalc_nusf(const int g, const XSRecord &rec, const double T)
{
  return rec.nusf[g];
}

// returns a row of the scattering matrix
void xscalc_sigma_s(const int g, const XSRecord &rec, const double T,
    Array<int> &col_idx, Vector &dat)
{

  Array <int> c; // columns
  Vector d; // data
  rec.sigma_s.GetRow(g, c, d);

  Vector v(rec.ngroup());
  v = 0.0;
  v.SetSubVector(c, d); // add other vectors to v for interpolating
  
  for (int g = 0; g < v.Size(); ++g)
    if (std::abs(v(g)) > 0.0)
      col_idx.Append(g);

  dat.SetSize(col_idx.Size());
  for (int i = 0; i < col_idx.Size(); ++i)
    dat(i) = v(col_idx[i]);

  return;
}

class DSARemovalCoefficient final : public Coefficient
{
public:
  DSARemovalCoefficient(const int g_,
                        const Material_library &mat_lib_,
                        const ParGridFunction &temperature_)
    : g{g_}, mat_lib{mat_lib_}, temperature{temperature_} {}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    const int attribute = T.Attribute-1;
    const XSRecord &rec = mat_lib.mat[attribute].rec;
    const double temp = temperature.GetValue(T, ip);
    const double tot = xscalc_sigma_t(g, rec, temp);
    const double scat = xscalc_selfscatter(g, rec, temp);
    return tot - scat;
  }
private:
  const int g;
  const Material_library &mat_lib;
  const ParGridFunction &temperature;
};

class DSADiffusiveCoefficient final : public Coefficient
{
public:
  DSADiffusiveCoefficient(const int g_,
                          const Material_library &mat_lib_,
                          const ParGridFunction &temperature_)
    : g{g_}, mat_lib{mat_lib_}, temperature{temperature_} {}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    const int attribute = T.Attribute-1;
    const XSRecord &rec = mat_lib.mat[attribute].rec;
    const double temp = temperature.GetValue(T, ip);
    const double tot = xscalc_sigma_t(g, rec, temp);
    return 1.0 / (3.0 * tot);
  }
private:
  const int g;
  const Material_library &mat_lib;
  const ParGridFunction &temperature;
};

class SelfscatterCoefficient final : public Coefficient
{
public:
  SelfscatterCoefficient(const int g_,
                         const Material_library &mat_lib_,
                         const ParGridFunction &temperature_)
    : g{g_}, mat_lib{mat_lib_}, temperature{temperature_} {}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    const int attribute = T.Attribute-1;
    const XSRecord &rec = mat_lib.mat[attribute].rec;
    const double temp = temperature.GetValue(T, ip);
    const double scat = xscalc_selfscatter(g, rec, temp);
    return scat;
  }
private:
  const int g;
  const Material_library &mat_lib;
  const ParGridFunction &temperature;
};

class SAAFTotalCoefficient final : public VectorCoefficient
{
public:
  SAAFTotalCoefficient(const int g_,
                       const Material_library &mat_lib_,
                       const Quadrature * const quad_,
                       const ParGridFunction &temperature_)
    : VectorCoefficient(quad_->nd()), g{g_}, mat_lib{mat_lib_}, quad{quad_}, temperature{temperature_} {}
  virtual void Eval(Vector &v, ElementTransformation &T, const IntegrationPoint &ip) override
  {
    const int attribute = T.Attribute-1;
    const XSRecord &rec = mat_lib.mat[attribute].rec;
    const double temp = temperature.GetValue(T, ip);
    const double tot = xscalc_sigma_t(g, rec, temp);
    v.SetSize(quad->nd());
    v = tot;
  }
private:
  const int g;
  const Material_library &mat_lib;
  const Quadrature * const quad;
  const ParGridFunction &temperature;
};

class SAAFStreamingCoefficient final : public MatrixTensorCoefficient
{
public:
  SAAFStreamingCoefficient(const int g_,
                           const Material_library &mat_lib_,
                           const Quadrature * const quad_,
                           const ParGridFunction &temperature_)
    : MatrixTensorCoefficient(quad_->nd()), g{g_}, mat_lib{mat_lib_}, quad{quad_}, temperature{temperature_} {}
  virtual void Eval(std::vector<DenseMatrix> &data,
                    Array<int> &row_idx,
                    Array<int> &col_idx,
                    ElementTransformation &T,
                    const IntegrationPoint &ip) const override
  {
    const int attribute = T.Attribute-1;
    const XSRecord &rec = mat_lib.mat[attribute].rec;
    const double temp = temperature.GetValue(T, ip);
    const double tot = xscalc_sigma_t(g, rec, temp);

    data.reserve(quad->nd());
    row_idx.SetSize(quad->nd());
    col_idx.SetSize(quad->nd());

    for (int n = 0; n < quad->nd(); ++n)
    {
      DenseMatrix omega_mat(quad->dim());
      MultVVt(quad->omega(n), omega_mat);
      omega_mat *= 1.0/tot;
      data.emplace_back(omega_mat);
      row_idx[n] = n;
      col_idx[n] = n;
    }

    return;
  }
private:
  const int g;
  const Material_library &mat_lib;
  const Quadrature * const quad;
  const ParGridFunction &temperature;
};

HypreParMatrix * build_amat(const int g,
                            const Material_library &mat_lib,
                            const Quadrature * const quad,
                            ParFiniteElementSpace *angular_flux_fespace,
                            const ParGridFunction &temperature)
{

  constexpr int skip_zeros = 2;

  Tensor<Vector> omega;
  omega.reserve(quad->nd());
  for(int n{0}; n < quad->nd(); ++n)
  {
    omega.AddElement(quad->omega(n), n);
  }
  VectorTensorConstantCoefficient omega_func{quad->nd(), omega};

  SAAFStreamingCoefficient streaming_func{g, mat_lib, quad, temperature};
  SAAFTotalCoefficient total_func{g, mat_lib, quad, temperature};

  ParBilinearForm a{angular_flux_fespace};
  a.AddDomainIntegrator(new VectorMassIntegrator{total_func});
  a.AddDomainIntegrator(new TensorDiffusionIntegrator{streaming_func});
  a.AddBdrFaceIntegrator(new VectorDGTraceIntegrator{omega_func, 1.0});

  a.Assemble(skip_zeros);
  a.Finalize(skip_zeros);
  return a.ParallelAssemble();
}


HypreParMatrix * build_dmat(const int g,
                           const Material_library &mat_lib,
                           ParFiniteElementSpace *scalar_flux_fespace,
                           const ParGridFunction &temperature)
{

  constexpr int skip_zeros = 2;

  DSADiffusiveCoefficient dsa_diffusive_func{g, mat_lib, temperature};
  DSARemovalCoefficient dsa_removal_func{g, mat_lib, temperature};

  ConstantCoefficient albedo{0.5};

  ParBilinearForm d{scalar_flux_fespace};
  d.AddDomainIntegrator(new DiffusionIntegrator{dsa_diffusive_func});
  d.AddDomainIntegrator(new MassIntegrator{dsa_removal_func});
  d.AddBoundaryIntegrator(new MassIntegrator{albedo});

  d.Assemble(skip_zeros);
  d.Finalize(skip_zeros);

  return d.ParallelAssemble();

}

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

  // transport streaming+collision operator
  std::vector<std::unique_ptr<HypreParMatrix>> Amat(mat_lib.ngroup());
  std::vector<std::unique_ptr<HypreBoomerAMG>> Aprec(mat_lib.ngroup());
  std::vector<std::unique_ptr<HyprePCG>> Asol(mat_lib.ngroup());

  time.start(time.matrix_assembly);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Amat[g].reset(build_amat(g, mat_lib, quad, angular_flux_fespace, temperature));

    std::unique_ptr<HypreBoomerAMG> prec{std::make_unique<HypreBoomerAMG>(*Amat[g])};
    prec->SetPrintLevel(inp.verbose ? 1 : 0);
    HYPRE_BoomerAMGSetNumFunctions(*prec, quad->nd());
    Aprec[g] = std::move(prec);

    std::unique_ptr<HyprePCG> this_sol{std::make_unique<HyprePCG>(MPI_COMM_WORLD)};
    this_sol->SetPrintLevel(inp.verbose ? 2 : 0);
    this_sol->SetPreconditioner(*Aprec[g]); // should be before operator
    this_sol->SetOperator(*Amat[g]);
    this_sol->SetTol(inp.tol.inner_rel);
    this_sol->SetAbsTol(inp.tol.inner_abs);
    this_sol->SetMaxIter(inp.tol.max_inner);
    Asol[g] = std::move(this_sol);

  } // g = 1,mat_lib.ngroup()

  // dsa streaming+removal operator
  std::vector<std::unique_ptr<HypreParMatrix>> Dmat(mat_lib.ngroup());
  std::vector<std::unique_ptr<HypreBoomerAMG>> Dprec(mat_lib.ngroup());
  std::vector<std::unique_ptr<HyprePCG>> Dsol(mat_lib.ngroup());

  if (inp.tol.use_dsa)
  { // DSA operator

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      Dmat[g].reset(build_dmat(g, mat_lib, scalar_flux_fespace, temperature));

      std::unique_ptr<HypreBoomerAMG> prec{std::make_unique<HypreBoomerAMG>(*Dmat[g])};
      prec->SetPrintLevel(0);
      Dprec[g] = std::move(prec);

      std::unique_ptr<HyprePCG> this_sol{std::make_unique<HyprePCG>(MPI_COMM_WORLD)};
      this_sol->SetPreconditioner(*Dprec[g]);
      this_sol->SetOperator(*Dmat[g]);
      this_sol->SetTol(inp.tol.inner_rel);
      this_sol->SetAbsTol(inp.tol.inner_abs);
      this_sol->SetMaxIter(inp.tol.max_inner);
      Dsol[g] = std::move(this_sol);

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

  std::vector<SelfscatterCoefficient> self_scatter_func;
  self_scatter_func.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    self_scatter_func.emplace_back(g, mat_lib, temperature);

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

  temperature = inp.temp_heat_pipe;

  time.start(time.solver);
  for (int iter{0}; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    if (inp.feedback)
    {

      time.start("power");
      calc_power_vector(scalar_flux, mat_lib, inp.rxpwr, temperature, power_quad);
      time.stop("power");

      time.start("temperature");
      temperature_heat_conduction(inp, mat_lib, power_quad, temperature);
      time.stop("temperature");

      time.start(time.matrix_assembly);

      for (int g = 0; g < mat_lib.ngroup(); ++g)
      {

        Amat[g].reset(build_amat(g, mat_lib, quad, angular_flux_fespace, temperature));
        Aprec[g]->SetOperator(*Amat[g]);
        Asol[g]->SetOperator(*Amat[g]);
        Asol[g]->SetPreconditioner(*Aprec[g]);

        if (inp.tol.use_dsa)
        {
          Dmat[g].reset(build_dmat(g, mat_lib, scalar_flux_fespace, temperature));
          Dprec[g]->SetOperator(*Dmat[g]);
          Dsol[g]->SetOperator(*Dmat[g]);
          Dsol[g]->SetPreconditioner(*Dprec[g]);
        }

      }

      time.stop(time.matrix_assembly);

    }

    time.start("calc_fiss_upscatter_vector");
    calc_fiss_upscatter_vector(scalar_flux, mat_lib, quad, temperature,
      fiss_upscatter_quad, saaf_fiss_upscatter_quad, keff, inp.flux_free_chi);
    time.stop("calc_fiss_upscatter_vector");

    for (int g{0}; g < mat_lib.ngroup(); ++g)
    {

      if (myid == 0) std::cout << "BEGIN_GROUP " << g << std::endl;

      time.start("calc_downscatter_vector");
      calc_downscatter_vector(g, scalar_flux, mat_lib, quad, temperature,
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
        calc_selfscatter_vector(g, *scalar_flux[g], mat_lib, quad, temperature,
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
    calc_fiss_src(scalar_flux, mat_lib, temperature, fiss_src_quad);
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
                                const ParGridFunction &temperature,
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

      Vector temp_locvec(dofs.Size());
      temperature.GetSubVector(dofs, temp_locvec.HostWrite());
      const double temp = dofval*temp_locvec;

      std::vector<double> spectrum(mat_lib.ngroup());

      // get all xs at this point
      // this will eventually be temperautre dependent

      std::vector<double> tot(mat_lib.ngroup());
      for (int g = 0; g < mat_lib.ngroup(); ++g)
        tot[g] = xscalc_sigma_t(g, rec, temp);

      // TODO
      std::vector<double> chi; // to be populated later

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
          fiss_sum += xscalc_nusf(gprime, rec, temp)*spectrum[gprime];

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
          xscalc_sigma_s(g, rec, temp, col_idx, scat_dat);

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
                             const ParGridFunction &temperature,
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

      Vector temp_locvec(dofs.Size());
      temperature.GetSubVector(dofs, temp_locvec.HostWrite());
      const double temp = dofval*temp_locvec;

      // get all xs at this point
      // this will eventually be temperature dependent
      const double tot = xscalc_sigma_t(g, rec, temp);

      Array<int> col_idx;
      Vector scat_dat;
      xscalc_sigma_s(g, rec, temp, col_idx, scat_dat);

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
                             const ParGridFunction &temperature,
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
      const double temp = temperature.GetValue(e, ip);
      const double scat = xscalc_selfscatter(g, rec, temp);
      const double tot = xscalc_sigma_t(g, rec, temp);

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
                   const ParGridFunction &temperature,
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

    const FiniteElement* const fe{fespace->GetFE(e)};
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i{0}; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip{ir.IntPoint(i)};
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      Vector temp_locvec(dofs.Size());
      temperature.GetSubVector(dofs, temp_locvec.HostWrite());
      const double temp = dofval*temp_locvec;

      for (int gprime{0}; gprime < mat_lib.ngroup(); ++gprime)
      {
        Vector locvec(dofs.Size());
        scalar_flux[gprime]->GetSubVector(dofs, locvec.HostWrite());
        const double flux{dofval*locvec};
        fiss[idx] += xscalc_nusf(gprime, rec, temp)*flux;
      }

      ++idx;
    }

  }

  return;
}

void calc_power_vector(const std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux,
                       const Material_library &mat_lib,
                       const double rxpwr,
                       const ParGridFunction &temperature,
                       QuadratureFunction &power_quad)
{

  constexpr double kappa_over_nu = 1.0; // TODO this should be group dependent

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

    const FiniteElement *fe = fespace->GetFE(e);
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip = ir.IntPoint(i);
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      Vector temp_locvec(dofs.Size());
      temperature.GetSubVector(dofs, temp_locvec.HostWrite());
      const double temp = dofval*temp_locvec;

      std::vector<double> kappasf(mat_lib.ngroup());
      for (int g = 0; g < mat_lib.ngroup(); ++g)
        kappasf[g] = kappa_over_nu*xscalc_nusf(g, rec, temp);

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

  // calculate power normalization factor

  ConstantCoefficient one{1.0};
  ParLinearForm b{scalar_flux[0]->ParFESpace()};
  b.AddDomainIntegrator(new DomainLFIntegrator{one});
  b.Assemble();

  QuadratureFunctionCoefficient power_func{power_quad};
  ParGridFunction power{scalar_flux[0]->ParFESpace()};
  power.ProjectCoefficient(power_func);

  const double factor = rxpwr / b(power); // use the inner product in the space

  power_quad *= factor;

  return;
}
