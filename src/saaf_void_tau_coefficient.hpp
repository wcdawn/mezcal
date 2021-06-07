#ifndef SAAF_VOID_TAU_COEFFICIENT_HPP
#define SAAF_VOID_TAU_COEFFICIENT_HPP

#include "mfem.hpp"

#include "material_library.hpp"

#include <map>

class SAAFVoidTauCoefficient : public mfem::Coefficient
{

public:

  SAAFVoidTauCoefficient(mfem::ParMesh *pm, const Material_library &mat_lib_lib,
                         int group, double squigma_in=0.5)
    : pmesh{pm}, mat_lib{mat_lib_lib}, g{group}, squigma{squigma_in}, c{c_in}
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  }

  virtual double Eval(mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip) override;

  virtual ~SAAFVoidTauCoefficient()
  {
    const int local_modified{modified};
    MPI_Allreduce(&local_modified, &modified, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if ((modified > 0) && (myid == 0))
      std::cout << "WARNING: SAAF void modification has been applied!"
                << " (g=" << g << ")"
                << " modified=" << modified
                << std::endl;
  }

private:

  mfem::ParMesh *pmesh{nullptr};

  const Material_library &mat_lib;

  const int g;

  const double squigma;
  const double c;

  int myid;
  int modified{0};

  double max_vertex_distance(const mfem::ElementTransformation &T);
  double effective_diameter(const mfem::ElementTransformation &T);

  static std::unordered_map<int,double> element_h;

};

#endif
