#include "nda_boundary_factor_coefficient.hpp"

using namespace mfem;

void NDABoundaryFactorCoefficient::Eval(Vector &v,
                                        ElementTransformation &T,
                                        const IntegrationPoint &ip)
{

  v.SetSize(vdim);

  Vector current_vec(current.VectorDim());
  current.GetVectorValue(T, ip, current_vec);

  Vector scalar_flux_vec(scalar_flux.VectorDim());
  scalar_flux.GetVectorValue(T, ip, scalar_flux_vec);

  Vector angular_flux_vec(angular_flux.VectorDim());
  angular_flux.GetVectorValue(T, ip, angular_flux_vec);

  /*
  v = current_vec;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    for (int d{0}; d < quad->dim(); ++d)
      v(d+g*quad->dim()) /= scalar_flux_vec(g);
  */

  v = 0.0;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    double flux{0.0};
    for (int n{0}; n < quad->nd(); ++n)
    {
      flux += quad->weight(n)*angular_flux_vec(n+g*quad->nd());
      const Vector dir{quad->omega(n)};
      for (int d{0}; d < quad->dim(); ++d)
      {
        v(d+g*quad->dim()) += 
          quad->weight(n)*dir(d)*angular_flux_vec(n+g*quad->nd());
      }
    }
    for (int d{0}; d < quad->dim(); ++d)
      v(d+g*quad->dim()) /= flux;
  }

  return;
}

/*
void NDABoundaryFactorCoefficient::Eval(Vector &v,
                                        ElementTransformation &T,
                                        const IntegrationPoint &ip)
{

  v.SetSize(vdim);

  Vector angular_flux_vec(angular_flux.VectorDim());
  angular_flux.GetVectorValue(T, ip, angular_flux_vec);

  Vector nor(quad->dim());
  CalcOrtho(T.Jacobian(), nor);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    double numerator{0.0};
    double denominator{0.0};
    for (int n{0}; n < quad->nd(); ++n)
    {
      const Vector dir{quad->omega(n)};
      const double eval{nor*dir};
      if (eval >= 0.0)
        numerator += eval*quad->weight(n)*angular_flux_vec(n+g*quad->nd());
      denominator += quad->weight(n)*angular_flux_vec(n+g*quad->nd());
    }
    v(g) = numerator / denominator;
  }

  return;
}
*/

/*
void NDABoundaryFactorCoefficient::Eval(Vector &v,
                                        ElementTransformation &T,
                                        const IntegrationPoint &ip)
{

  v.SetSize(vdim);

  Vector nor(quad->dim());
  CalcOrtho(T.Jacobian(), nor);

  Vector scalar_flux_vec(scalar_flux.VectorDim());
  scalar_flux.GetVectorValue(T, ip, scalar_flux_vec);

  Vector current_vec(current.VectorDim());
  current.GetVectorValue(T, ip, current_vec);

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    Vector cur(quad->dim());
    for (int d{0}; d < quad->dim(); ++d)
      cur(d) = current_vec(d+g*quad->dim());
    v(g) = cur*nor/scalar_flux_vec(g);
  }

  return;
}
*/
