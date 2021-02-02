#ifndef QUADRATURE
#define QUADRATURE

#include "mfem.hpp"

#include "timer.hpp"

#include <exception>
#include <string>
#include <sstream>
#include <array>
#include <vector>
#include <unordered_map>
#include <cmath>

#include "delaunay.hpp"

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

class Direction
{

public:

  Direction (double omega_x, double omega_y, double omega_z)
    : dir{omega_x, omega_y, omega_z} {}
  Direction (double phi, double theta)
    : dir{std::sin(theta)*std::cos(phi), std::sin(theta)*std::sin(phi), std::cos(theta)} {}
  Direction(const mfem::Vector& vec) : dir{vec(0), vec(1), (vec.Size() == 2) ? 0.0 : vec(2)} {}

  double& operator[](int i){ return dir[i]; }
  const double operator[](int i) const { return dir[i]; }

  Direction operator+(const Direction& d) const
  { 
    return {dir[0]+d[0], dir[1]+d[1], dir[2]+d[2]};
  }
  Direction operator-(const Direction& d) const
  { 
    return {dir[0]-d[0], dir[1]-d[1], dir[2]-d[2]};
  }
  void operator +=(const Direction& d)
  {
    for (int i{0}; i < 3; ++i)
      dir[i] += d[i];
  }
  void operator -=(const Direction& d)
  {
    for (int i{0}; i < 3; ++i)
      dir[i] -= d[i];
  }

  Direction operator*(double x) const { return {dir[0]*x, dir[1]*x, dir[2]*x}; }
  Direction operator/(double x) const { return {dir[0]/x, dir[1]/x, dir[2]/x}; }

  double operator*(const Direction& x) const { return dir[0]*x[0] + dir[1]*x[1] + dir[2]*x[2]; }

  void operator*=(double x)
  {
    for (int i{0}; i < 3; ++i)
      dir[i] *= x;
  }
  void operator/=(double x)
  {
    for (int i{0}; i < 3; ++i)
      dir[i] /= x;
  }

  double dot(const Direction& d)
  {
    double sum{0.0};
    for (int i{0}; i < 3; ++i)
      sum += dir[i]*d[i];
    return sum;
  }

  Direction reflection(const Direction& nhat)
  {
    return (*this) - nhat*2.0*this->dot(nhat);
  }

  std::array<double,2> angle() const
  {
    const double theta{0.5*M_PI - std::acos(dir[2])};
    double phi{std::acos(dir[0] / std::sqrt(1.0 - dir[2]*dir[2]))};
    if (dir[1] < 0.0)
      phi += M_PI;
    return {phi,theta};
  }

  double phi() const { return angle()[0]; }
  double theta() const { return angle()[1]; }

  mfem::Vector vec(int dim=3) const
  {
    mfem::Vector ans(dim);
    for (int i{0}; i < dim; ++i)
      ans(i) = dir[i];
    return ans;
  }

private:

  std::array<double,3> dir{0.0,0.0,0.0};

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

  virtual mfem::Vector omega(int n, int dim=-1) const
  { 
    const std::array<int,3> dir_sign{sign(n)};
    const mfem::Vector dir_octant{omega_octant(n % n_dir_per_octant)};
    mfem::Vector dir{dim > 0 ? dim : dimension};
    for (int i{0}; i < dir.Size(); ++i)
      dir(i) = dir_sign[i]*dir_octant(i);
    return dir;
  }

  virtual Direction dir(int n, int dim=-1) const
  {
    return Direction{omega(n,dim)};
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

  TriangularMesh& get_mesh() { return msh; }

  Direction reflection(int n, const Direction& nhat, int dim=-1) const
  {
    return dir(n,dim).reflection(nhat);
  }

  std::unordered_multimap<int,double> interpolate_reflection(int n, const Direction& nhat) const
  {
    const Direction omega_prime{reflection(n, nhat, 3)};
    const Point reflected_angle{omega_prime.angle()};
    return interpolate_weight(reflected_angle, msh);
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

  TriangularMesh msh;
  void populate_mesh()
  {
    populate_point_list();
    populate_element_list();

    switch (dimension)
    {
      case (2):
        add_reflected_points_half(msh);
        break;
      case (3):
      default:
        add_reflected_points_full(msh);
        break;
    }

    populate_element_list();
    cleanup_mesh();

  }

private:

  void populate_point_list()
  {
    std::vector<Point> pts(n_dir);
    for (int i{0}; i < n_dir; ++i)
    {
      const Direction dir{omega(i,3)};
      pts[i] = dir.angle();
    }
    msh.associate_point_list(pts);
  }

  void populate_element_list()
  {
    delaunay(msh);
  }

  void cleanup_mesh()
  {
    cleanup_elements(msh);
    cleanup_points(msh);
  }

};

#endif
