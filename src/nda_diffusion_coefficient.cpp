#include "nda_diffusion_coefficient.hpp"

using namespace mfem;

void NDADiffusionCoefficient::Eval(Vector &v,
                                   ElementTransformation &T,
                                   const IntegrationPoint &ip)
{

  const int attribute{T.Attribute-1};

  Vector current_vec(current.VectorDim());
  current.GetVectorValue(T, ip, current_vec);

  Vector scalar_flux_vec(scalar_flux.VectorDim());
  scalar_flux.GetVectorValue(T, ip, scalar_flux_vec);

  DenseMatrix scalar_flux_grad;
  T.SetIntPoint(&ip);
  scalar_flux.GetVectorGradient(T, scalar_flux_grad);

  Vector angular_flux_vec(angular_flux.VectorDim());
  angular_flux.GetVectorValue(T, ip, angular_flux_vec);

  DenseMatrix angular_flux_grad;
  T.SetIntPoint(&ip);
  angular_flux.GetVectorGradient(T, angular_flux_grad);

  v.SetSize(vdim);
  v = 0.0;
  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {
    const double diffusion{1.0 / (3.0 * mat_lib.mat[attribute].rec.sigma_t[g])};

    double flux{0.0};

    /*
    for (int n{0}; n < quad->nd(); ++n)
    {
      Vector dir{quad->omega(n)};
      Vector grad(quad->dim());
      angular_flux_grad.GetRow(n+g*quad->nd(), grad);
      flux += quad->weight(n)*angular_flux_vec(n+g*quad->nd());
      for (int d{0}; d < quad->dim(); ++d)
      {
        v(d+g*quad->dim()) += quad->weight(n)*(
          dir(d)*angular_flux_vec(n+g*quad->nd()) + diffusion*grad(d));
      }
    }
    for (int d{0}; d < quad->dim(); ++d)
      v(d+g*quad->dim()) /= flux;
    */

    Vector grad(quad->dim());
    scalar_flux_grad.GetRow(g, grad);
    grad *= diffusion;
    for (int d{0}; d < quad->dim(); ++d)
    {
      //v(d+g*quad->dim()) = grad(d);
      //v(d+g*quad->dim()) = scalar_flux_vec(g);
      //v(d+g*quad->dim()) = current_vec(d+g*quad->dim());
      v(d+g*quad->dim()) = (current_vec(d+g*quad->dim()) + grad(d)) / scalar_flux_vec(g);
    }

  }

  return;
}
