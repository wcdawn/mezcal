#include "upscattering_coefficient.hpp"

using namespace mfem;

constexpr int xs_temp_idx = 0;

double UpscatteringCoefficient::Eval(ElementTransformation& T, 
                                     const IntegrationPoint& ip)
{
  const int attribute{T.Attribute - 1};

  if (!is_upscatter[attribute])
    return 0.0;

  double val{0.0};
  
  Vector scat_dat;
  Array<int> scat_idx;
  mat_lib.mat[attribute].rec[xs_temp_idx].sigma_s.GetRow(g, scat_idx, scat_dat);

  for (int gidx{0}; gidx < scat_idx.Size(); ++gidx)
  {
    const int gprime{scat_idx[gidx]};
    if (gprime > g)
    {
      const double flux{scalar_flux(gprime).GetValue(T, ip)};
      const double scatter{scat_dat[gidx]};
      val += scatter*flux;
    }
  }

  return val;
}

std::vector<bool> UpscatteringCoefficient::check_if_upscatter_present()
{

  std::vector<bool> upscatter_present(mat_lib.nmat());

  for (int i{0}; i < mat_lib.nmat(); ++i)
  {

    Vector scat_dat;
    Array<int> scat_idx;
    mat_lib.mat[i].rec[xs_temp_idx].sigma_s.GetRow(g, scat_idx, scat_dat);

    upscatter_present[i] = false;

    for (int gidx{0}; gidx < scat_idx.Size(); ++gidx)
    {
      const int gprime{scat_idx[gidx]};
      if (gprime > g) // relies on the column indices being sorted
      {
        upscatter_present[i] = true;
        break;
      }
    }
  }

  return upscatter_present;

}
