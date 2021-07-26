#include "output.hpp"

#include "saaf_void_tau_coefficient.hpp"

using namespace mfem;

void total_xs_plot(ParMesh *pmesh,
                   const Material_library &mat_lib,
                   const std::string out_name)
{

  H1_FECollection fec{1, pmesh->Dimension()}; // linear elements...
  ParFiniteElementSpace fespace{pmesh, &fec};

  ParaViewDataCollection output{out_name, pmesh};

  std::vector<ParGridFunction> tot_pgf;
  tot_pgf.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    tot_pgf.emplace_back(&fespace);

  std::vector<ParGridFunction> void_pgf;
  void_pgf.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    void_pgf.emplace_back(&fespace);

  const size_t group_width{std::to_string(mat_lib.ngroup()-1).length()};

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Vector tot_vec(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      tot_vec(i) = mat_lib.mat[i].rec.sigma_t[g];
    PWConstCoefficient tot_func{tot_vec};
    tot_pgf[g].ProjectCoefficient(tot_func);

    // a good value is probably closer to 1e-3
    constexpr double squigma = 4.5e-2; // default=0.5
    SAAFVoidTauCoefficient saaf_void_func{pmesh, mat_lib, g, squigma};
    //SAAFVoidTauCoefficient saaf_void_func{pmesh, mat_lib, g};
    void_pgf[g].ProjectCoefficient(saaf_void_func);

    std::ostringstream record_name;
    record_name << "total_xs_g"
                << std::setfill('0')
                << std::setw(group_width)
                << g;
    output.RegisterField(record_name.str(), &tot_pgf[g]);

    std::ostringstream saaf_record_name;
    saaf_record_name << "saaf_void_inv_tot_g"
                     << std::setfill('0')
                     << std::setw(group_width)
                     << g;
    output.RegisterField(saaf_record_name.str(), &void_pgf[g]);

    if (fespace.GetMyRank() == 0)
      std::cout << "total_xs_g=" << g
                << " ratio=" << tot_vec.Max() / tot_vec.Min()
                << " modified_ratio=" << void_pgf[g].Max() / void_pgf[g].Min()
                << std::endl;

  }

  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  return;
}

void coarse_energy_plot(ParMesh *pmesh,
                        const Result &scalar_flux,
                        const std::vector<int> collapse_vector,
                        const std::string out_name)
{

  const int ncoarse{static_cast<int>(collapse_vector.size())+1};

  std::vector<ParGridFunction> coarse_flux;
  coarse_flux.reserve(ncoarse);
  for (int i{0}; i < ncoarse; ++i)
    coarse_flux.emplace_back(scalar_flux(0));

  for (ParGridFunction &pgf : coarse_flux)
    pgf = 0.0;

  int coarse_idx{0};
  
  for (int g{0}; g < scalar_flux.ngroup(); ++g)
  {

    // see if we have passed the coarse energy boundary but also make sure we
    // don't access collapse_vector out-of-bounds
    // note: takes advantage of short-circuiting
    if ((coarse_idx < static_cast<int>(collapse_vector.size())) && 
        (g >= collapse_vector[coarse_idx]))
      ++coarse_idx;

    coarse_flux[coarse_idx] += scalar_flux(g);

  }

  const size_t group_width{std::to_string(scalar_flux.ngroup()).length()};

  ParaViewDataCollection output{out_name, pmesh};

  for (int g{0}; g < ncoarse; ++g)
  {
    std::ostringstream coarse_flux_record;
    coarse_flux_record << "coarse_flux_g"
                       << std::setfill('0')
                       << std::setw(group_width)
                       << g;
    output.RegisterField(coarse_flux_record.str(), &coarse_flux[g]);
  }

  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  return;
}

void mesh_stats(ParMesh *pmesh,
                std::ostream &os,
                Material_library *mat_lib)
{

  const int myid{pmesh->GetMyRank()};

  Vector area_vol(pmesh->attributes.Max());
  area_vol = 0.0;

  for (int e{0}; e < pmesh->GetNE(); ++e)
  {
    const int att{pmesh->GetAttribute(e)-1};
    const double vol{pmesh->GetElementVolume(e)};
    area_vol(att) += vol;
  }

  const auto old_precision{os.precision()};
  os << std::setprecision(20);

  if (myid == 0) os << "=== AREA/VOLUME ===" << std::endl;
  if (myid == 0) os << "mat # | vol [cm^2 or cm^3] | exact vol | rel. err. [%] | correction factor" << std::endl;
  double total_area_vol{0.0};
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    double sum_area_vol{0.0};
    MPI_Allreduce(&area_vol(i), &sum_area_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    total_area_vol += sum_area_vol;
    if (myid == 0) os << i+1 << " " << sum_area_vol;

    if ((mat_lib) && (mat_lib->mat[i].exact_area_volume >= 0.0))
    {
      const double &exact_area_vol{mat_lib->mat[i].exact_area_volume};
      const double rel_error{(sum_area_vol - exact_area_vol) / exact_area_vol};
      const double factor{1.0 / (1.0 + rel_error)};
      XSRecord &rec{mat_lib->mat[i].rec};

      for (Material_isotope &isotope : mat_lib->mat[i].iso)
        isotope.nden *= factor;

      for (int g{0}; g < mat_lib->ngroup(); ++g)
      {
        rec.sigma_t[g] *= factor;
        rec.sigma_tr[g] *= factor;
        rec.nusf[g] *= factor;
      }

      rec.sigma_r *= factor;
      rec.sigma_s *= factor;

      if (myid == 0)
        std::cout << " " << exact_area_vol
                  << " " << rel_error*1e2
                  << " " << factor;

    }

    if (myid == 0) std::cout << std::endl;

  }
  if (myid == 0) os << "total " << total_area_vol << std::endl;
  if (myid == 0) os << std::endl;

  os << std::setprecision(old_precision);

  return;
}
