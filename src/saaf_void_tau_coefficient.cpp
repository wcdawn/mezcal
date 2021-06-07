#include "saaf_void_tau_coefficient.hpp"

#include <cmath>

using namespace mfem;

double SAAFVoidTauCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{

  const int attribute{T.Attribute-1};
  const double sigma_t{mat_lib.mat[attribute].rec.sigma_t[g]};

  //constexpr bool max_vertex_distance{true};
  const bool max_vertex_distance{g == 0}; // TODO

  // TODO split this out into private member function
  // TODO implement cacheing
  double h{0.0};
  if (max_vertex_distance)
  {
    Array<int> vertices;
    pmesh->GetElementVertices(T.ElementNo, vertices);
    for (int i{0}; i < vertices.Size(); ++i)
    {
      const double *v1{pmesh->GetVertex(vertices[i])};
      for (int j{0}; j < vertices.Size(); ++j)
      {
        if (i == j) continue;
        const double *v2{pmesh->GetVertex(vertices[j])};
        double dist = 0.0;
        for (int d{0}; d < pmesh->Dimension(); ++d)
          dist += std::pow(v1[d] - v2[d], 2);
        dist = std::sqrt(dist);
        h = std::max(h, dist);
      }
    }
  }
  else
  {
    // h is supposed to represent "maximum vertex separation for points within an element"
    // Here, I approximate it using the effective diameter of a sphere or circle
    // for an element with equivalent volume.
    const double volume{pmesh->GetElementVolume(T.ElementNo)};
    if (pmesh->Dimension() == 3)
      h = std::cbrt(6.0/M_PI*volume);
    else if (pmesh->Dimension() == 2)
      h = std::sqrt(4.0/M_PI*volume);
    else
      MFEM_ABORT("SAAFVoidTauCoefficient effective diameter only implmented for 2d & 3d");
  }

  constexpr bool debug{true};

  return h;

  if (c*h*sigma_t >= squigma) return (debug) ? 0.0 : 1.0/(c*sigma_t); // unmodified
  ++modified;
  return (debug) ? 1.0 : h/squigma; // modified

}
