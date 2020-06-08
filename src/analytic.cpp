#include "analytic.hpp"

#include <cmath>
#define _USE_MATH_DEFINES

#include <iostream>

// TODO remove this block
using std::cout;

Analytic::Analytic(Reference ref, const Mesh* const mesh, 
    const GridFunction& grid_sol, double k) : 
  reference_solution{ref}, nVert{mesh->GetNV()}, nDim{mesh->SpaceDimension()}, keff{k}
{
  // set node
  Vector vec_node;
  mesh->GetVertices(vec_node);

  // put node in a useful format
  node.reserve(nVert);
  for (int i{0}; i < nVert; ++i)
  {
    if (nDim == 3)
      node.push_back(Vertex{vec_node[0*nVert+i], vec_node[1*nVert+i], vec_node[2*nVert+i]});
    else
      node.push_back(Vertex{vec_node[0*nVert+i], vec_node[1*nVert+i]});
  }

  // set sol
  grid_sol.GetNodalValues(sol, 1); // vdim=1
  if (reference_solution != fixed_src) 
  {
    for (int i{0}; i < sol.Size(); ++i)
      sol[i] = std::abs(sol[i]);
    sol /= sol.Max(); // normalize to maximal value 1 ($\phi_0$ = 1)
  }

  // set fun to appropriate solution function
  switch (ref)
  {
    case (fixed_src):
      fun = &Analytic::fixed_src_fun;
      break;
    case (oned_oneg):
      fun = &Analytic::oned_oneg_fun;
      analytic_keff = &Analytic::oned_oneg_keff;
      break;
    case (twod_oneg):
    case (square):
      fun = &Analytic::twod_oneg_fun;
      analytic_keff = &Analytic::twod_oneg_keff;
      break;
    case (finite_cyl):
      fun = &Analytic::finite_cyl_fun;
      analytic_keff = &Analytic::finite_cyl_keff;
      break;
    default:
      throw InvalidReferenceName{};
  }
}

void Analytic::print(std::ostream& os) const
{
  for (int i{0}; i < nVert; ++i)
  {
    const double val{fun(this, node[i])};
    const double err{std::abs(sol[i] - val)};
    os << node[i](0) << ',' << node[i](1) << ',' << node[i](2) << ',' << sol[i] 
       << ',' << val << ',' << err << '\n';
  }
}

void Analytic::analyze(std::ostream& os) const
{
  double L1norm{0.0};
  double L2norm{0.0};
  double inferr{0.0};

  for (int i{0}; i < nVert; ++i)
  {
    const double val{fun(this, node[i])};
    const double err{std::abs(sol[i] - val)};
    L1norm += err;
    L2norm += err*err;
    inferr = std::max(inferr, err);
  }

  const double rmserr{L2norm / nVert};

  cout << "\n=== ANALYZE ===\n";

  if (reference_solution != fixed_src)
  {
    const double kref{analytic_keff(this)};
    const auto previous_precision{os.precision()};
    os << "kref = " << std::setprecision(20) << kref << '\n';
    os << "keff_err_[pcm] = " << (kref - keff) * 1e5 << '\n';
    os << std::setprecision(previous_precision);
  }

  os << "L1norm = " << L1norm << '\n';
  os << "L2norm = " << L2norm << '\n';
  os << "rmserr = " << rmserr << '\n';
  os << "inferr = " << inferr << '\n';
  cout << '\n';
}

double Analytic::fixed_src_fun(const Vertex& v) const
{
  // TODO these need to come as input to the constructor
  static constexpr double sigma_tr{1.0 / 3.0};
  static constexpr double sigma_r{1.0};
  static constexpr double q_fixed{1.0};

  static constexpr double Lx_small{1.0};
  static constexpr double diffusion{1.0 / (3.0 * sigma_tr)};

  const double kappa{std::sqrt(sigma_r / diffusion)};
  const double x{v(0)}; // x is the zeroth coordinate

  return (q_fixed / diffusion) * (1.0 - std::cosh(kappa * x) + 
      (std::cosh(kappa * Lx_small) - 1.0) / (std::sinh(kappa * Lx_small)) * 
      std::sinh(kappa * x));
}

double Analytic::oned_oneg_fun(const Vertex& v) const
{
  static constexpr double Lx{100.0};
  const double x{v(0)}; // x is the zeroth coordinate
  return std::sin((M_PI / Lx) * x);
}

double Analytic::twod_oneg_fun(const Vertex& v) const
{
  static constexpr double Lx{100.0};
  static constexpr double Ly{100.0};
  const double x{v(0)}; // x is the zeroth coordinate
  const double y{v(1)}; // x is the first coordinate
  return std::sin((M_PI / Lx) * x) * std::sin((M_PI / Ly) * y);
}

double Analytic::finite_cyl_fun(const Vertex& v) const
{
  static constexpr double R{50.0};
  static constexpr double H{100.0};

  const double r{std::sqrt(v(0)*v(0) + v(1)*v(1))}; // distance to origin
  const double z{v(2)};

  return std::cyl_bessel_j(0, (alpha0 / R) * r) * std::sin((M_PI / H) * z);
}

double Analytic::oned_oneg_keff() const
{
  static constexpr double transport{3.619027e-2};
  static constexpr double removal{1.532090e-1};
  static constexpr double nusf{1.566769e-1};

  static constexpr double Lx{100.0};

  static constexpr double diffusion{1.0 / (3.0 * transport)};

  return nusf / (diffusion * std::pow(M_PI / Lx, 2) + removal);
}

double Analytic::twod_oneg_keff() const
{
  static constexpr double transport{3.619027e-2};
  static constexpr double removal{1.532090e-1};
  static constexpr double nusf{1.566769e-1};

  static constexpr double Lx{100.0};
  static constexpr double Ly{100.0};

  static constexpr double diffusion{1.0 / (3.0 * transport)};

  return nusf / (diffusion * (std::pow(M_PI / Lx, 2) + std::pow(M_PI / Ly, 2))
    + removal);
}

double Analytic::finite_cyl_keff() const
{
  static constexpr double transport{1.0 / 3.0};
  static constexpr double removal{1.0};
  static constexpr double nusf{1.0};

  static constexpr double R{50.0};
  static constexpr double H{100.0};

  static constexpr double diffusion{1.0 / (3.0 * transport)};

  return nusf / (diffusion * (std::pow(alpha0 / R, 2) * std::pow(M_PI / H, 2))
    + removal);
}
