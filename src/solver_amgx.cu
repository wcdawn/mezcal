#include "solver_amgx.h"

#include "matrix_pwconst_coefficient.hpp"
#include "vector_pwconst_coefficient.hpp"

// amgx
#include "vector.h"
#include "matrix.h"

using namespace mfem;

void spn_amgx_cuda_vector(Timer& time, const Input& inp,
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

  const unsigned int N{static_cast<unsigned int>(res(0).Size())};
  const amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> amgx_device;
  amgx_device.print();

  amgx::Vector<amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> x(N);
  amgx::Vector<amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> x_old(N);
  amgx::Vector<amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> v(N);
  amgx::Vector<amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> b(N);

  //amgx::Matrix<amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> Amat;
  //amgx::Matrix<amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> Mmat;

  delete Ahypre;
  delete Mhypre;

  return;
}

