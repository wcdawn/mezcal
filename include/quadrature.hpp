#ifndef QUADRATURE
#define QUADRATURE

#include "mfem.hpp"

#include <exception>
#include <string>
#include <sstream>
#include <vector>

enum class Quadrature_type
{
  invalid = -1,
  level_symmetric,
  legendre_chebyschev
};

class Quadrature
{

public:

  int nd() const { return n_dir; }
  int order() const { return ord; }

  virtual double omega_x(int n) const =0;
  virtual double omega_y(int n) const =0;
  virtual double omega_z(int n) const =0;
  virtual double weight(int n)  const =0;

  virtual mfem::Vector omega(int n) const
  {
    mfem::Vector dir(3);
    dir[0] = omega_x(n);
    dir[1] = omega_y(n);
    dir[2] = omega_z(n);
    return dir;
  }

protected:

  // (approximate) order of the quadrature.
  // the maximum order of the spherical harmonics that can be integrated exactly
  // using this quadrature.
  int ord{-1};

  // Number of unique directions (angles) in this quadrature.
  int n_dir{-1};

};

class Invalid_quadrature : public std::exception
{
public:
  Invalid_quadrature(const std::string& str) : msg{str} {}
  const char* what() const noexcept override { return msg.c_str(); }
private:
  std::string msg;
};

#endif
