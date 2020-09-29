#include "legendre_chebyschev_quadrature.hpp"

using namespace mfem;

using std::array;

void Legendre_chebyschev_quadrature::initialize_weights_and_abscissae()
{
  chebyschev_weight = chebyschev_weight_data(nc);
  chebyschev_abscissae = chebyschev_abscissae_data(nc);

  legendre_weight = legendre_weight_data(nl);
  legendre_abscissae = legendre_abscissae_data(nl);
}

array<int,2> Legendre_chebyschev_quadrature::calc_idx(int n) const
{
  const int idx_legendre{n / nc2};
  const int idx_chebyschev{n - nc2 * idx_legendre};
  return {idx_legendre, idx_chebyschev};
}

int Legendre_chebyschev_quadrature::calc_idx_legendre(int n) const
{
  return calc_idx(n)[0];
}

int Legendre_chebyschev_quadrature::calc_idx_chebyschev(int n) const
{
  return calc_idx(n)[1];
}

double Legendre_chebyschev_quadrature::omega_x_octant(int n) const
{
  const array<int,2> idx{calc_idx(n)};
  const double mu_tilde{std::sqrt(1.0 
    - std::pow(legendre_abscissae[idx[0]],2))};
  return mu_tilde * std::cos(chebyschev_abscissae[idx[1]]);
}

double Legendre_chebyschev_quadrature::omega_y_octant(int n) const
{
  const array<int,2> idx{calc_idx(n)};
  const double mu_tilde{std::sqrt(1.0 
    - std::pow(legendre_abscissae[idx[0]],2))};
  return mu_tilde * std::sin(chebyschev_abscissae[idx[1]]);
}

double Legendre_chebyschev_quadrature::omega_z_octant(int n) const
{
  return legendre_abscissae[calc_idx_legendre(n)];
}

double Legendre_chebyschev_quadrature::weight_octant(int n) const
{
  // adjusted such that weights are normalized to 1.0 per quadrant
  // this saves us from dividing by 4pi anywhere
  return legendre_weight[calc_idx_legendre(n)] * chebyschev_weight * 2.0 / M_PI;
}

Vector Legendre_chebyschev_quadrature::omega_octant(int n) const
{
  const array<int,2> idx{calc_idx(n)};
  const double mu_tilde{std::sqrt(1.0
    - std::pow(legendre_abscissae[idx[0]],2))};

  Vector dir(3);
  dir(0) = mu_tilde * std::cos(chebyschev_abscissae[idx[1]]);
  dir(1) = mu_tilde * std::sin(chebyschev_abscissae[idx[1]]);
  dir(2) = legendre_abscissae[idx[0]];
  return dir;
}
