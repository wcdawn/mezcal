#ifndef QUADRATURE
#define QUADRATURE

# include "mfem.hpp"

#include <exception>
#include <string>
#include <sstream>
#include <array>
#include <vector>

enum class Quadrature_choice
{
  invalid = -1,
  level_symmetric,
  legendre_chebyschev
};

std::string quadrature_choice_enum_to_str(Quadrature_choice q);
Quadrature_choice quadrature_choice_str_to_enum(const std::string& q);

class Invalid_quadrature : public std::exception
{
public:
  Invalid_quadrature(const std::string& str) : msg{str} {}
  const char* what() const noexcept override { return msg.c_str(); }
private:
  std::string msg;
};

class Quadrature
{

protected:

  Quadrature(int spatial_dimension) : dimension{spatial_dimension} {}

public:

  int nd() const { return n_dir; }
  int nd_octant() const { return n_dir_per_octant; }
  int dim() const { return dimension; }
  int order() const { return ord; }

  virtual mfem::Vector omega(int n) const
  { 
    const std::array<int,3> dir_sign{sign(n)};
    const mfem::Vector dir_octant{omega_octant(n % n_dir_per_octant)};
    mfem::Vector dir(dimension);
    for (int i{0}; i < dir.Size(); ++i)
      dir(i) = dir_sign[i]*dir_octant(i);
    return dir;
  }

  virtual double omega_x(int n) const { return sign(n)[0] * omega_x_octant(n % n_dir_per_octant); }
  virtual double omega_y(int n) const { return sign(n)[1] * omega_y_octant(n % n_dir_per_octant); }
  virtual double omega_z(int n) const { return sign(n)[2] * omega_z_octant(n % n_dir_per_octant); }
  virtual double weight(int n)  const { return weight_octant(n % n_dir_per_octant) / std::pow(2, dimension); }

  std::array<int,3> sign(int n) const
  {
    switch (n / n_dir_per_octant) // warning: integer division
    {
      case (0):
        return {1, 1, 1};
      case (1):
        return {1, -1, 1};
      case (2):
        return {-1, -1, 1};
      case (3):
        return {-1, 1, 1};
      case (4):
        return {1, 1, -1};
      case (5):
        return {1, -1, -1};
      case (6):
        return {-1, -1, -1};
      case (7):
        return {-1, 1, -1};
      default:
        throw Invalid_quadrature{"Could not calculate octant."};
    }
  }

protected:

  virtual mfem::Vector omega_octant(int n) const
  { 
    mfem::Vector dir(3);
    dir(0) = omega_x_octant(n);
    dir(1) = omega_y_octant(n);
    dir(2) = omega_z_octant(n);
    return dir;
  }

  virtual double omega_x_octant(int n) const =0;
  virtual double omega_y_octant(int n) const =0;
  virtual double omega_z_octant(int n) const =0;
  virtual double weight_octant(int n)  const =0;

  // number of spatial dimensions
  // allowable values shall be 2 or 3
  // if dimension==2, assume x-y plane (omega_z is not used)
  const int dimension;

  // (approximate) order of the quadrature.
  // the maximum order of the spherical harmonics that can be integrated exactly
  // using this quadrature.
  int ord{-1};

  // Total number of unique angles in the quadrature for the entire unit sphere.
  int n_dir{-1};

  // Number of unique angles in each octant for this quadrature.
  int n_dir_per_octant{-1};

};

#endif
