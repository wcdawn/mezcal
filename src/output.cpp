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
