#include "solver_spn_power_iteration.hpp"

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
void spn_power_iteration(Timer &time,
                         Input &inp,
                         Material_library &mat_lib,
                         std::vector<std::unique_ptr<ParGridFunction>> &phi,
                         double &keff)
{

  ParFiniteElementSpace *fespace = phi[0]->ParFESpace();
  const int myid = fespace->GetMyRank();
  ParMesh *pmesh = fespace->GetParMesh();

  constexpr int skip_zeros = 2;

  const int vdim = mat_lib.ngroup()*inp.spn_eq();

  mat_lib.set_nmoment(inp.spn_order()+1);

  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
  ess_bdr = 1; // default is all essential
  for (int i = 0; ((i < inp.boundary_conditions.Size()) && (i < ess_bdr.Size())); ++i)
    ess_bdr[i] = inp.boundary_conditions[i];

  Array<int> vacuum_condition(pmesh->bdr_attributes.Max());
  vacuum_condition = 0;
  bool is_zero_flux = false;
  bool is_vacuum = false;
  for (int i = 0; i < ess_bdr.Size(); ++i)
  {
    switch (ess_bdr[i])
    {
      case (1):
        is_zero_flux = true;
        break;
      case (2):
        is_vacuum = true;
        vacuum_condition[i] = 1;
        break;
    }
  }

  Array<int> ess_tdof_list;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  ConstantCoefficient zero{0.0};

  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      const int idx = g+n*mat_lib.ngroup();
      *phi[idx] = (n == 0) ? 1.0 : 0.0;
      phi[idx]->ProjectBdrCoefficient(zero, ess_bdr);
      phi[idx]->UseDevice(true);
    }
  }

  keff = 1.0;

  std::vector<PWConstCoefficient> vacuum_bc_func;
  vacuum_bc_func.reserve(mat_lib.ngroup()*inp.spn_eq()*inp.spn_eq());
  for (int g = 0; g < mat_lib.ngroup(); ++g)
  {
    for (int n = 0; n < inp.spn_eq(); ++n)
    {
      for (int nprime = 0; nprime < inp.spn_eq(); ++nprime)
      {

        Vector val(pmesh->attributes.Max());
        val = 0.0;
        for (int nsum = 0; nsum < (inp.spn_eq()-n); ++nsum)
        {

          const int rem_idx = 2*(n+nsum)+1;
          const double coeff = spn_bc_mat[n](nprime,nsum);

          for (int i = 0; i < pmesh->attributes.Max(); ++i)
          {
            const double rem = mat_lib.mat[i].rec.sigma_r(g,rem_idx);
            val(i) += coeff*rem;
          }

        }
        vacuum_bc_func.emplace_back(val);

      }
    }
  }

  std::vector<PWConstCoefficient> leakage_func;
  leakage_func.reserve(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {

    const double dble_n = 2.0*n;
    const double xnext = (dble_n+1.0)*(dble_n+1.0)/((2.0*dble_n+1.0)*(2.0*dble_n+3.0));
    const double xprev = dble_n*dble_n/((2.0*dble_n+1.0)*(2.0*dble_n-1.0));

    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {

      Vector leakage(pmesh->attributes.Max());
      for (int i = 0; i < pmesh->attributes.Max(); ++i)
      {
        leakage(i) = xnext / mat_lib.mat[i].rec.sigma_r(g,2*n+1);
        if (n > 0)
          leakage(i) += xprev / mat_lib.mat[i].rec.sigma_r(g,2*n-1);
      }
      leakage_func.emplace_back(leakage);

    }

  }


  std::vector<std::unique_ptr<HypreParMatrix>> Amat(vdim);
  std::vector<std::unique_ptr<HypreBoomerAMG>> Aprec(vdim);
  std::vector<std::unique_ptr<HyprePCG>> Asol(vdim);

  time.start(time.matrix_assembly);
  {

    for (int n = 0; n < inp.spn_eq(); ++n)
    {

      for (int g = 0; g < mat_lib.ngroup(); ++g)
      {

        const int idx = g + n*mat_lib.ngroup();
        const int bc_idx = n + n*inp.spn_eq() + g*inp.spn_eq()*inp.spn_eq();

        Vector removal(pmesh->attributes.Max());
        for (int i = 0; i < pmesh->attributes.Max(); ++i)
          removal(i) = mat_lib.mat[i].rec.sigma_r(g,2*n);
        PWConstCoefficient removal_func{removal};

        ProductCoefficient intermediate_vacuum_bc_product_func{
          leakage_func[idx], vacuum_bc_func[bc_idx]};
        ProductCoefficient vacuum_bc_product_func{-1.0, intermediate_vacuum_bc_product_func};

        std::unique_ptr<ParBilinearForm> a{std::make_unique<ParBilinearForm>(fespace)};
        a->AddDomainIntegrator(new DiffusionIntegrator{leakage_func[idx]});
        a->AddDomainIntegrator(new MassIntegrator{removal_func});
        a->AddBoundaryIntegrator(new MassIntegrator{vacuum_bc_product_func}, vacuum_condition);

        a->Assemble(skip_zeros);
        if (is_zero_flux) a->EliminateEssentialBCDiag(ess_bdr, 1.0);
        a->Finalize(skip_zeros);
        Amat[idx].reset(a->ParallelAssemble());

        Aprec[idx] = std::make_unique<HypreBoomerAMG>(*Amat[idx]);
        Aprec[idx]->SetPrintLevel(0);

        Asol[idx] = std::make_unique<HyprePCG>(MPI_COMM_WORLD);
        Asol[idx]->SetPrintLevel(inp.verbose ? 2 : 0);
        Asol[idx]->SetPreconditioner(*Aprec[idx]);
        Asol[idx]->SetOperator(*Amat[idx]);
        Asol[idx]->SetTol(inp.tol.inner_rel);
        Asol[idx]->SetAbsTol(inp.tol.inner_abs);
        Asol[idx]->SetMaxIter(inp.tol.max_inner);

      }
    }

  }
  time.stop(time.matrix_assembly);

  const int integration_order = fespace->GetOrder(0)+1;
  QuadratureSpace qspace{pmesh, integration_order};

  std::vector<QuadratureFunction> fiss_upscatter_quad;
  fiss_upscatter_quad.reserve(mat_lib.ngroup());
  for (int g = 0; g < mat_lib.ngroup(); ++g)
  {
    fiss_upscatter_quad.emplace_back(&qspace);
    fiss_upscatter_quad.back().UseDevice(true);
  }

  QuadratureFunction dnscatter_quad{&qspace};

  QuadratureFunctionCoefficient src_quad_func{dnscatter_quad};

  QuadratureFunction spn_prev_quad{&qspace, pmesh->Dimension()};

  VectorQuadratureFunctionCoefficient spn_prev_quad_func{spn_prev_quad};

  std::vector<QuadratureFunction> spn_next_quad;
  spn_next_quad.reserve(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      spn_next_quad.emplace_back(&qspace, pmesh->Dimension());
      spn_next_quad.back().UseDevice(true);
    }
  }

  std::vector<VectorQuadratureFunctionCoefficient> spn_next_quad_func;
  spn_next_quad_func.reserve(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      const int idx = g+n*mat_lib.ngroup();
      spn_next_quad_func.emplace_back(spn_next_quad[idx]);
    }
  }

  std::vector<GridFunctionCoefficient> phi_func;
  std::vector<GradientGridFunctionCoefficient> dphi_func;
  phi_func.reserve(vdim);
  dphi_func.reserve(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      const int idx = g + n*mat_lib.ngroup();
      phi_func.emplace_back(phi[idx].get());
      dphi_func.emplace_back(phi[idx].get());
    }
  }

  std::vector<PWConstCoefficient> inv_removal_func;
  inv_removal_func.reserve(mat_lib.ngroup()*(inp.spn_order()+1));
  for (int n = 0; n < inp.spn_order()+1; ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {

      Vector inv_removal(pmesh->attributes.Max());
      for (int i = 0; i < pmesh->attributes.Max(); ++i)
        inv_removal(i) = 1.0 / mat_lib.mat[i].rec.sigma_r(g,n);
      inv_removal_func.emplace_back(inv_removal);

    }
  }

  std::vector<ConstantCoefficient> spn_src_xnext_func, spn_src_xprev_func;
  spn_src_xnext_func.reserve(inp.spn_eq());
  spn_src_xprev_func.reserve(inp.spn_eq());
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    const double dble_n = 2.0*n;
    const double xnext = (dble_n+1.0)*(dble_n+2.0)/((2.0*dble_n+1.0)*(2.0*dble_n+3.0));
    const double xprev = (dble_n)*(dble_n-1.0)/((2.0*dble_n+1.0)*(2.0*dble_n-1.0));
    spn_src_xnext_func.emplace_back(xnext);
    spn_src_xprev_func.emplace_back(xprev);
  }

  std::vector<ProductCoefficient> spn_intermediate_src_xnext_func, spn_intermediate_src_xprev_func;
  spn_intermediate_src_xnext_func.reserve(vdim);
  spn_intermediate_src_xprev_func.reserve(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      spn_intermediate_src_xnext_func.emplace_back(spn_src_xnext_func[n], inv_removal_func[g+(2*n+1)*mat_lib.ngroup()]);
      spn_intermediate_src_xprev_func.emplace_back(spn_src_xprev_func[n], inv_removal_func[g+(2*n-1)*mat_lib.ngroup()]);
    }
  }

  std::vector<ScalarVectorProductCoefficient> spn_src_next_func, spn_src_prev_func;
  spn_src_next_func.reserve(vdim);
  spn_src_prev_func.reserve(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      const int i = g + n*mat_lib.ngroup();
      const int inext = g + (n+1)*mat_lib.ngroup();
      const int iprev = g + (n-1)*mat_lib.ngroup();
      spn_src_next_func.emplace_back(spn_intermediate_src_xnext_func[i], dphi_func[inext]);
      spn_src_prev_func.emplace_back(spn_intermediate_src_xprev_func[i], dphi_func[iprev]);
    }
  }

  std::vector<ProductCoefficient> spn_lf_bc_func;
  spn_lf_bc_func.reserve(2*mat_lib.ngroup()*(inp.spn_eq()-1)*inp.spn_eq());
  std::vector<std::unique_ptr<ParLinearForm>> lf(vdim);
  for (int n = 0; n < inp.spn_eq(); ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      const int idx = g+n*mat_lib.ngroup();
      lf[idx] = std::make_unique<ParLinearForm>(fespace);
      lf[idx]->UseDevice(true);

      if (n == 0)
        lf[idx]->AddDomainIntegrator(new DomainLFIntegrator{src_quad_func});

      if (n > 0)
      {
        lf[idx]->AddDomainIntegrator(new DomainLFGradIntegrator{spn_prev_quad_func});
        if (is_vacuum)
          lf[idx]->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator{spn_src_prev_func[idx]},
            vacuum_condition);
      }

      if ((n+1) < inp.spn_eq())
      {
        lf[idx]->AddDomainIntegrator(new DomainLFGradIntegrator{spn_next_quad_func[idx]});
        if (is_vacuum)
          lf[idx]->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator{spn_src_next_func[idx]},
            vacuum_condition);
      }

      if (is_vacuum)
      {
        for (int nprime = 0; nprime < inp.spn_eq(); ++nprime)
        {
          if (nprime == n) continue;
          const int idx_prime = g + nprime*mat_lib.ngroup();
          const int bc_idx_prime = nprime + n*inp.spn_eq() + g*inp.spn_eq()*inp.spn_eq();
          spn_lf_bc_func.emplace_back(leakage_func[idx], vacuum_bc_func[bc_idx_prime]);
          spn_lf_bc_func.emplace_back(spn_lf_bc_func.back(), phi_func[idx_prime]);
          // TODO this adds a lot of integrators which is potentially slow
          lf[idx]->AddBoundaryIntegrator(new BoundaryLFIntegrator{spn_lf_bc_func.back()}, vacuum_condition);
        }
      }

    }
  }

  QuadratureFunction fiss_src_quad_old{&qspace};
  fiss_src_quad_old.UseDevice(true);
  fiss_src_quad_old = 1.0;

  double k_conv = inp.tol.k * 1e1 + 1e-5;
  double phi_conv = inp.tol.phi * 1e1 + 1e-5;
  double keff_old = keff;
  double fiss_sum_old = 1.0;

  int solve_counter = 0;

  time.start(time.solver);

  for (int iter = 0; (iter < inp.tol.max_outer)
                 && ((k_conv > inp.tol.k) || (phi_conv > inp.tol.phi)); ++iter)
  {

    time.start("fiss_upscatter");
    spn_calc_fiss_upscatter(phi, keff, mat_lib, fiss_upscatter_quad, inp.flux_free_chi);
    time.stop("fiss_upscatter");

    if (inp.spn_eq() > 1)
    {
      time.start("spn_next");
      spn_calc_spn_next(phi, inp.spn_eq(), mat_lib, spn_next_quad);
      time.stop("spn_next");
    }

    for (int n = 0; n < inp.spn_eq(); ++n)
    {

      for (int g = 0; g < mat_lib.ngroup(); ++g)
      {

        const int idx = g + n*mat_lib.ngroup();

        if (n == 0)
        {
          time.start("downscatter");
          spn_calc_downscatter(g, phi, mat_lib, dnscatter_quad);
          dnscatter_quad += fiss_upscatter_quad[g];
          time.stop("downscatter");
        }
        else
        {
          time.start("spn_prev");
          spn_calc_spn_prev(g, n, phi, mat_lib, spn_prev_quad);
          time.stop("spn_prev");
        }

        time.start("lf_assemble");
        lf[idx]->Assemble();
        time.stop("lf_assemble");

        if (is_zero_flux)
        {
          lf[idx]->SetSubVector(ess_tdof_list, 0.0);
          phi[idx]->SetSubVector(ess_tdof_list, 0.0);
        }
          
        std::unique_ptr<HypreParVector> b{lf[idx]->ParallelAssemble()};
        std::unique_ptr<HypreParVector> x{phi[idx]->ParallelAssemble()};

        time.start("Asol");
        Asol[idx]->Mult(*b, *x);
        phi[idx]->Distribute(*x);
        time.stop("Asol");

        ++solve_counter;

      } // g

    } // n

    time.start("fiss_src");
    QuadratureFunction fiss_src_quad{&qspace};
    fiss_src_quad.UseDevice(true);
    spn_calc_fiss_src(phi, mat_lib, fiss_src_quad);
    time.stop("fiss_src");

    const double local_fiss_sum = fiss_src_quad.Sum();
    double fiss_sum;
    MPI_Allreduce(&local_fiss_sum, &fiss_sum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    fiss_src_quad_old -= fiss_src_quad;
    phi_conv = par_norml2(fiss_src_quad_old)/par_norml2(fiss_src_quad);

    if (iter > 0) keff *= fiss_sum / fiss_sum_old;
    k_conv = std::abs(keff - keff_old);

    if (myid == 0)
      std::cout << iter
                << " keff=" << keff
                << " dk=" << k_conv
                << " dphi=" << phi_conv
                << std::endl;

    // store old values
    keff_old = keff;
    fiss_sum_old = fiss_sum;
    fiss_src_quad_old = fiss_src_quad;

  }

  time.stop(time.solver);

  return;
}

void spn_calc_fiss_upscatter(const std::vector<std::unique_ptr<ParGridFunction>> &phi,
                             double keff,
                             const Material_library &mat_lib,
                             std::vector<QuadratureFunction> &fiss_upscatter_quad,
                             bool flux_free_chi)
{

  const QuadratureSpace * const qspace = fiss_upscatter_quad[0].GetSpace();
  const ParFiniteElementSpace * const fespace = phi[0]->ParFESpace();
  const ParMesh * const pmesh = fespace->GetParMesh();

  for (int g = 0; g < mat_lib.ngroup(); ++g)
    fiss_upscatter_quad[g] = 1.0;

  int idx = 0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = pmesh->GetAttribute(e)-1;
    const Material_record &mat = mat_lib.mat[attribute];
    const XSRecord &rec = mat.rec;

    const FiniteElement * const fe = fespace->GetFE(e);
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip = ir.IntPoint(i);
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      std::vector<double> spectrum(mat_lib.ngroup());

      const std::vector<double> &nusf = rec.nusf;
      const SparseMatrix &scat = rec.sigma_s;
      std::vector<double> chi(mat_lib.ngroup());

      if (rec.is_fiss)
      {

        // accumulate flux spectrum
        for (int g{0}; g < mat_lib.ngroup(); ++g)
        {
          Vector locvec(dofs.Size());
          phi[g]->GetSubVector(dofs, locvec.HostWrite());
          spectrum[g] = dofval*locvec;
        }

        if (flux_free_chi)
          chi = rec.chi;
        else if (mat_lib.rx_wide_chi)
          chi = mat_lib.chi;
        else
          chi = mat.calculate_chi(spectrum);

      }
      else
      {
        for (int g = 0; g < mat_lib.ngroup(); ++g)
          spectrum[g] = -1.0;
      }

      Vector val(mat_lib.ngroup());
      val = 0.0;

      // fission
      if (rec.is_fiss)
      {

        double fiss_sum = 0.0;
        for (int gprime = 0; gprime < mat_lib.ngroup(); ++gprime)
          fiss_sum += nusf[gprime]*spectrum[gprime];

        for (int g = 0; g < mat_lib.ngroup(); ++g)
          val(g) = chi[g]*fiss_sum;

        val /= keff;

      }

      // upscatter
      for (int g = 0; g < mat_lib.ngroup(); ++g)
      {
        if (rec.is_upscatter[g])
        {

          Array<int> col_idx;
          Vector scat_dat;
          scat.GetRow(g, col_idx, scat_dat);

          for (int gidx = 0; gidx < col_idx.Size(); ++gidx)
          {
            const int gprime = col_idx[gidx];
            if (gprime > g)
            {
              if (spectrum[gprime] < 0.0)
              {
                Vector locvec(dofs.Size());
                phi[gprime]->GetSubVector(dofs, locvec.HostWrite());
                spectrum[gprime] = dofval*locvec;
              }
              val(g) += scat_dat(gidx)*spectrum[gprime];
            }
          }
        }
      }

      for (int g = 0; g < mat_lib.ngroup(); ++g)
      {
        double *fquad = fiss_upscatter_quad[g].HostWrite();
        const double *vval = val.HostRead();
        fquad[idx] = vval[g];
      }

      ++idx;
    }
  }

  return;
}

void spn_calc_spn_next(const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                       int spn_eq,
                       const Material_library &mat_lib,
                       std::vector<QuadratureFunction> &spn_next_quad)
{

  const QuadratureSpace * const qspace = spn_next_quad[0].GetSpace();
  const ParFiniteElementSpace * const fespace = phi[0]->ParFESpace();
  const ParMesh * const pmesh = fespace->GetParMesh();
  const int dim = pmesh->Dimension();

  for (int n = 0; n < spn_eq; ++n)
  {
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      const int idx = g + n*mat_lib.ngroup();
      spn_next_quad[idx] = 0.0;
    }
  }

  int idx = 0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = pmesh->GetAttribute(e)-1;
    const Material_record &mat = mat_lib.mat[attribute];
    const XSRecord &rec = mat.rec;
    const DenseMatrix &sigma_r = rec.sigma_r;

    ElementTransformation *T = fespace->GetElementTransformation(e);

    Vector grad_dat(dim*mat_lib.ngroup()*(spn_eq-1));

    for (int n = 0; n < spn_eq-1; ++n)
    {

      const double dble_n = 2.0*n;
      const double xmult = (dble_n+1.0)*(dble_n+2.0)/((2.0*dble_n+1.0)*(2.0*dble_n+3.0));

      for (int g = 0; g < mat_lib.ngroup(); ++g)
      {

        // TODO look into unrolling the call to GetGradient but it is hard
        Vector grad(dim);
        phi[g+(n+1)*mat_lib.ngroup()]->GetGradient(*T, grad);
        grad *= -xmult;

        for (int d = 0; d < dim; ++d)
          grad_dat(d + g*dim + n*dim*mat_lib.ngroup()) = grad(d);
      }
    }

    const double *ggrad_dat = grad_dat.HostRead();

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
      for (int d = 0; d < dim; ++d)
      {
        for (int n = 0; n < spn_eq-1; ++n)
        {
          for (int g = 0; g < mat_lib.ngroup(); ++g)
          {
            const double xs_val = sigma_r(g,2*n+1);
            const double val = ggrad_dat[d + g*dim + n*dim*mat_lib.ngroup()] / xs_val;
            double *nquad = spn_next_quad[g+n*mat_lib.ngroup()].HostWrite();
            nquad[idx] = val;
          }
        }
        ++idx;
      }
    }

  }

  return;
}

void spn_calc_downscatter(int g,
                          const std::vector<std::unique_ptr<ParGridFunction>> &phi,
                          const Material_library &mat_lib,
                          QuadratureFunction &dnscatter_quad)
{

  const QuadratureSpace * const qspace = dnscatter_quad.GetSpace();
  const ParFiniteElementSpace * const fespace = phi[0]->ParFESpace();
  const ParMesh * const pmesh = fespace->GetParMesh();

  dnscatter_quad = 0.0;
  double *dquad = dnscatter_quad.HostWrite();

  int idx = 0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = pmesh->GetAttribute(e)-1;
    const Material_record &mat = mat_lib.mat[attribute];
    const XSRecord &rec = mat.rec;

    const FiniteElement * const fe = fespace->GetFE(e);
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip = ir.IntPoint(i);
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      const SparseMatrix &scat = rec.sigma_s;
      Array<int> col_idx;
      Vector scat_dat;
      scat.GetRow(g, col_idx, scat_dat);

      double val = 0.0;

       // assume column indices are sorted
      for (int gidx = 0; gidx < col_idx.Size(); ++gidx)
      {

        const int gprime = col_idx[gidx];
        if (gprime >= g) break;

        Vector locvec(dofs.Size());
        phi[gprime]->GetSubVector(dofs, locvec.HostWrite());
        const double flux = dofval*locvec;

        val += scat_dat(gidx)*flux;

      }

      dquad[idx] = val;
      ++idx;

    }

  }

  return;
}

void spn_calc_spn_prev(int g, int n,
                       const std::vector<std::unique_ptr<ParGridFunction>> &phi,
                       const Material_library &mat_lib,
                       QuadratureFunction &spn_prev_quad)
{

  spn_prev_quad = 0.0;
  if (n == 0) return;

  const QuadratureSpace * const qspace = spn_prev_quad.GetSpace();
  const ParFiniteElementSpace * const fespace = phi[0]->ParFESpace();
  const ParMesh * const pmesh = fespace->GetParMesh();
  const int dim = pmesh->Dimension();

  int idx = 0;

  double *pquad = spn_prev_quad.HostWrite();
  const double dble_n = 2.0*n;
  const double xmult = (dble_n)*(dble_n-1.0)/((2.0*dble_n+1.0)*(2.0*dble_n-1.0));


  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = pmesh->GetAttribute(e)-1;
    const Material_record &mat = mat_lib.mat[attribute];
    const XSRecord &rec = mat.rec;

    ElementTransformation *T = fespace->GetElementTransformation(e);

    Vector grad(dim);
    phi[g+(n-1)*mat_lib.ngroup()]->GetGradient(*T, grad);
    grad *= -xmult;

    const double *ggrad = grad.HostRead();

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const DenseMatrix &sigma_r = rec.sigma_r;
      const double xs_val = sigma_r(g,2*n-1);

      for (int d = 0; d < dim; ++d)
      {
        pquad[idx] = ggrad[d]/xs_val;
        ++idx;
      }
    }

  }

  return;
}

void spn_calc_fiss_src(const std::vector<std::unique_ptr<ParGridFunction>> &phi,
                       const Material_library &mat_lib,
                       QuadratureFunction &fiss_src_quad)
{

  const QuadratureSpace * const qspace = fiss_src_quad.GetSpace();
  const ParFiniteElementSpace * const fespace = phi[0]->ParFESpace();
  const ParMesh * const pmesh = fespace->GetParMesh();

  fiss_src_quad = 0.0;
  double *fiss = fiss_src_quad.HostWrite();

  int idx = 0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    const int attribute = pmesh->GetAttribute(e)-1;
    const Material_record &mat = mat_lib.mat[attribute];
    const XSRecord &rec = mat.rec;

    if (!rec.is_fiss)
    {
      idx += ir.GetNPoints();
      continue;
    }

    const FiniteElement * const fe = fespace->GetFE(e);
    Array<int> dofs;
    fespace->GetElementDofs(e, dofs);

    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip = ir.IntPoint(i);
      Vector dofval(dofs.Size());
      fe->CalcShape(ip, dofval);

      const std::vector<double> &nusf = rec.nusf;

      double val = 0.0;
      for (int gprime = 0; gprime < mat_lib.ngroup(); ++gprime)
      {
        Vector locvec(dofs.Size());
        phi[gprime]->GetSubVector(dofs, locvec.HostWrite());
        const double flux = dofval*locvec;
        val += nusf[gprime]*flux;
      }

      fiss[idx] = val;
      ++idx;

    }

  }

  return;
}
