#include "saaf_void_tau_coefficient.hpp"

#include <cmath>

using namespace mfem;

std::unordered_map<int,double> SAAFVoidTauCoefficient::element_h(0);

double SAAFVoidTauCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{

  const int attribute{T.Attribute-1};
  const double sigma_t{mat_lib.mat[attribute].rec.sigma_t[g]};

  constexpr bool use_max_vertex_distance{true};

  // evaluate using cacheing
  double h{0.0};
  const auto h_idx{element_h.find(T.ElementNo)};
  if (h_idx != element_h.end())
  {
    h = h_idx->second;
  }
  else
  {
    h = (use_max_vertex_distance) ? max_vertex_distance(T) : effective_diameter(T);
    element_h.insert({T.ElementNo, h});
  }

  // TODO
  // this is for plotting and determining an acceptable value for squigma
  return h*sigma_t;

  constexpr bool debug{true};
  if (h*sigma_t >= squigma) return (debug) ? 0.0 : 1.0/sigma_t; // unmodified
  ++modified;
  return (debug) ? 1.0 : h/squigma; // modified

}

double SAAFVoidTauCoefficient::max_vertex_distance(const ElementTransformation &T)
{
  double h{0.0};
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
  return h;
}

double SAAFVoidTauCoefficient::effective_diameter(const ElementTransformation &T)
{
    // approximate it using the effective diameter of a sphere or circle for an
    // element with equivalent volume.
    const double volume{pmesh->GetElementVolume(T.ElementNo)};
    switch (pmesh->Dimension())
    {
      case (3):
        return std::cbrt(6.0/M_PI*volume);
      case (2):
        return std::sqrt(4.0/M_PI*volume);
      default:
        MFEM_ABORT("SAAFVoidTauCoefficient effective diameter only implmented for 2d & 3d");
    }
    return -1.0;
}
