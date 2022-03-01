#include "multiphysics.hpp"

#include <cmath>

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

void multiphysics(const int state,
                  const Quadrature * const quad,
                  Timer &time,
                  Input &inp,
                  const Material_library &mat_lib,
                  std::vector<std::unique_ptr<ParGridFunction>> &angular_flux,
                  std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux,
                  double &keff,
                  ParGridFunction &power,
                  ParGridFunction &temperature)
{

  constexpr int max_iter = 10;
  constexpr double ktol = 1e-7;
  constexpr double powertol = 1e-6;

  ParFiniteElementSpace *scalar_flux_fespace = scalar_flux[0]->ParFESpace();
  ParMesh *pmesh = scalar_flux_fespace->GetParMesh();
  const int myid = scalar_flux_fespace->GetMyRank();

  double keff_old = keff;

  double kconv = ktol*1e2 + 1.0;
  double powerconv = powertol*1e2 + 1.0;

  const int integration_order = scalar_flux_fespace->GetOrder(0)+1;
  QuadratureSpace qspace {pmesh, integration_order};

  QuadratureFunction power_quad{&qspace};
  QuadratureFunction power_quad_old{&qspace};

  if ((state == 0) && (!inp.restart.flux))
  {
    temperature = inp.temp_heat_pipe;
    for (int g = 0; g < mat_lib.ngroup(); ++g)
    {
      *scalar_flux[g] = 1.0;
      *angular_flux[g] = 1.0;
    }
  }

  for (int iter = 0; (iter < max_iter) && ((kconv > ktol) || (powerconv > powertol)); ++iter)
  {

    const bool init = ((state == 0) && (iter == 0) && (!inp.restart.flux));

    calc_power_vector(scalar_flux, inp, mat_lib, inp.rxpwr, temperature, power_quad_old);
    temperature_heat_conduction(inp, mat_lib, power_quad_old, temperature);
    saaf_angle_power_iteration(init, quad, time, inp, mat_lib, angular_flux, scalar_flux, keff, power, temperature);
    calc_power_vector(scalar_flux, inp, mat_lib, inp.rxpwr, temperature, power_quad);

    kconv = std::abs(keff_old - keff);
    power_quad_old -= power_quad;
    powerconv = (inp.rxpwr > 0.0) ?  par_norml2(power_quad_old)/par_norml2(power_quad) : 0.0;

    if (myid == 0)
      std::cout << "multiphysics_iter=" << iter
                << " multiphysics_dk=" << kconv 
                << " multiphysics_dpow=" << powerconv
                << std::endl;

    keff_old = keff;

  }

  if (myid == 0) 
    std::cout << "MULTIPHYSICS_FINISHED!!! :)" << std::endl;

  return;
}
