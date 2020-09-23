#include "quadrature.hpp"

std::string quadrature_choice_enum_to_str(Quadrature_choice q)
{
  switch (q)
  {
    case (Quadrature_choice::level_symmetric): return "level_symmetric";
    case (Quadrature_choice::legendre_chebyschev): return "legendre_chebyschev";
    default: return "invalid";
  }
}

Quadrature_choice quadrature_choice_str_to_enum(const std::string& q)
{
  if (q == "level_symmetric") return Quadrature_choice::level_symmetric;
  if (q == "legendre_chebyschev") return Quadrature_choice::legendre_chebyschev;
  return Quadrature_choice::invalid;
}
