#include "level_symmetric_quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stddef.h>

using namespace mfem;

using std::size_t;

using std::array;
using std::vector;

using std::sort;
using std::greater;
using std::distance;
using std::unique;

array<int,3> Level_symmetric_quadrature::calc_ijk(int n) const
{
  int count{n};

  int i{1};
  int j{count+1};

  while (count >= 0)
  {
    count -= ord2 - i + 1;
    ++i;
    if (count >= 0)
    {
      j = count+1;
    }
  }

  i -= 1;
  const int k{ord2 + 2 - i - j};

  return {i, j, k};

}

int Level_symmetric_quadrature::calc_i(int n) const
{
  const std::array<int,3> ijk{calc_ijk(n)};
  return ijk[0];
}

int Level_symmetric_quadrature::calc_j(int n) const
{
  const std::array<int,3> ijk{calc_ijk(n)};
  return ijk[1];
}

int Level_symmetric_quadrature::calc_k(int n) const
{
  const std::array<int,3> ijk{calc_ijk(n)};
  return ijk[2];
}

void Level_symmetric_quadrature::populate_weight_id()
{
  weight_id.resize(n_dir_per_octant);
  
  int count{0};
  for (int i{1}; i < (ord2 + 1); ++i)
  {
    for (int j{1}; j < (ord2 - i + 1 + 1); ++j)
    {
      const int k{ord2 + 2 - i - j};
      weight_id[count] = std::pow(2,i-1) + std::pow(2,j-1) + std::pow(2,k-1);
      ++count;
    }
  }

  vector<int> weight_map{weight_id};
  sort(weight_map.begin(), weight_map.end(), greater<int>()); // descending sort
  const auto it{unique(weight_map.begin(), weight_map.end())};
  weight_map.resize(distance(weight_map.begin(), it));

  for (size_t i{0}; i < weight_map.size(); ++i)
  {
    for (size_t j{0}; j < weight_id.size(); ++j)
    {
      if (weight_id[j] == weight_map[i])
        weight_id[j] = static_cast<int>(i);
    }
  }

}

Vector Level_symmetric_quadrature::omega_octant(int n) const
{
  const array<int,3> ijk{calc_ijk(n)};

  Vector dir(3);
  dir(0) = mu[ijk[0]-1];
  dir(1) = mu[ijk[1]-1];
  dir(2) = mu[ijk[2]-1];

  return dir;
}

void Level_symmetric_quadrature::populate_mu()
{
  const double mu_one{mu_one_data(ord)};
  mu.resize(ord2);

  if (ord == 2)
  {
    mu[0] = mu_one;
    return;
  }

  const double mu_one_sq{mu_one*mu_one};
  for (int i{1}; i < ord2+1; ++i)
  {
    mu[i-1] = std::sqrt(mu_one_sq + (i-1)*(2.0*(1.0 - 3.0*mu_one_sq))/(ord-2.0));
  }

}

void Level_symmetric_quadrature::populate_discrete_weight()
{
  discrete_weight = discrete_weight_data(ord);
}
