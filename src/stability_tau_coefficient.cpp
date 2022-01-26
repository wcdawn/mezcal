#include "stability_tau_coefficient.hpp"

#include "xslibrary.hpp"

#include <cmath>

using namespace mfem;

constexpr int xs_temp_idx = 0;

double StabilityTauCoefficient::Eval(ElementTransformation& T, 
                                     const IntegrationPoint& ip)
{

  const int attribute{T.Attribute - 1};
  const XSRecord& record{mat_lib.mat[attribute].rec[xs_temp_idx]};
  const double sigma_tr{record.sigma_tr[g]};
  //const double sigma_tr{record.sigma_t[g]};

  //std::cout << record.sigma_tr[g] << ' ' << record.sigma_t[g] << ' ' << record.sigma_s.Elem(0,0) << std::endl;

  // TODO in the original work, this is the maximum diameter of the element.
  // In my implementation, it is the average (effective) diameter of the element
  // I'm not sure if this makes any difference but it may need to be changed.
  const double volume{mesh->GetElementVolume(T.ElementNo)};
  //const double hk{std::cbrt(6.0/M_PI*volume)}; // effective diameter of the element (sphere)
  //const double hk{std::sqrt(4.0/M_PI*volume)}; // effective diameter of the element (circle)
  const double hk{std::sqrt(2.0*volume)}; // maximum diameter of the element for a square

  const double p{static_cast<double>(T.Order())}; // order of basis funciton

  const double peclet{std::sqrt(3.0)*hk*sigma_tr*0.5/p};
  const double zeta{std::min(peclet, 1.0)};

  //std::cout << hk*zeta*0.5/p << std::endl;
  //return 0.5;
  // refine 5, factor 3
  //std::cout << "WARN:1000" << std::endl;
  return hk*zeta*0.5/p;

}
