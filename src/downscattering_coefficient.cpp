#include "downscattering_coefficient.hpp"

using namespace mfem;

double DownscatteringCoefficient::Eval(ElementTransformation& T, 
                                     const IntegrationPoint& ip)
{
  const int attribute{T.Attribute - 1};

  double val{0.0};
  
  Vector scat_dat;
  Array<int> scat_idx;
  mat_lib.mat[attribute].rec.sigma_s.GetRow(g, scat_idx, scat_dat);

  for (int gidx{0}; gidx < scat_idx.Size(); ++gidx)
  {
    const int gprime{scat_idx[gidx]};

    val += scalar_flux(gprime).GetValue(T, ip)*scat_dat[gidx];

    /*
    if (gprime >= g) // relies on the column indices being sorted
    {
      break;
    }
    else
    {
      const double flux{scalar_flux(gprime).GetValue(T, ip)};
      const double scatter{scat_dat[gidx]};
      val += scatter*flux;
    }
    */

  }

  return val;
}
