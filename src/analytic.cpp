#include "analytic.hpp"

#include <cmath>
#include <vector>
#define _USE_MATH_DEFINES

#include <iostream>
#include <string>

using std::ostream;
using std::setprecision;

using std::string;
using std::vector;

using std::max;
using std::pow;
using std::sqrt;

using std::abs;
using std::cos;
using std::cosh;
using std::cyl_bessel_j; // NOTE: requires c++17
using std::sin;
using std::sinh;
using std::tanh;

double oned_oneg_c_fun(const Vector& v);

Analytic::Analytic(const Reference ref, Result& res, const XSLibrary& xslib,
                   const Mesh* const mesh)
  : reference_solution{ref}, nVert{mesh->GetNV()}, nDim{mesh->SpaceDimension()},
    keff{res.keff}, xs{&xslib}
{
  // set node
  Vector vec_node;
  mesh->GetVertices(vec_node);

  // put node in a useful format
  node.reserve(nVert);
  for (int i{0}; i < nVert; ++i)
  {
    if (nDim == 3)
      node.push_back(Vertex{vec_node[0 * nVert + i], vec_node[1 * nVert + i],
                            vec_node[2 * nVert + i]});
    else
      node.push_back(Vertex{vec_node[0 * nVert + i], vec_node[1 * nVert + i]});
  }

  // set sol
  res.phi.GetNodalValues(sol, 1); // vdim=1
  if (reference_solution != fixed_src)
  {
    for (int i{0}; i < sol.Size(); ++i) sol[i] = abs(sol[i]);
  }

  // store a writable pointer
  phi = &res.phi;
  if (reference_solution != fixed_src)
  {
    for (int i{0}; i < phi->Size(); ++i)
    {
      phi->GetData()[i] = abs(phi->GetData()[i]);
    }
    const double local_phi_max{phi->Max()};
    double phi_max;
    MPI_Allreduce(&local_phi_max, &phi_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    *phi /= phi_max;
    sol /= phi_max;
  }

  // set fun to appropriate solution function
  switch (ref)
  {
    case (fixed_src): fun = &Analytic::fixed_src_fun; break;
    case (oned_oneg):
      fun           = &Analytic::oned_oneg_fun;
      analytic_keff = &Analytic::oned_oneg_keff;
      break;
    case (twod_oneg): // FALL THROUGH
    case (square):
      fun           = &Analytic::twod_oneg_fun;
      analytic_keff = &Analytic::twod_oneg_keff;
      break;
    case (threed_oneg):
      fun           = &Analytic::threed_oneg_fun;
      analytic_keff = &Analytic::threed_oneg_keff;
      break;
    case (finite_cyl):
      fun           = &Analytic::finite_cyl_fun;
      analytic_keff = &Analytic::finite_cyl_keff;
      break;
    case (tworeg):
      fun           = &Analytic::tworeg_fun;
      analytic_keff = &Analytic::tworeg_keff;
      break;
    default: throw InvalidReferenceName{};
  }
}

void Analytic::analyze(ostream& os) const
{
  Vector err(sol.Size());
  for (int i{0}; i < err.Size(); ++i)
    err[i] = fun(this, node[i], *xs);
  err -= sol;

  // These are computed as they are in LUPINE and are here for compatibility.
  // This method does not account properly for differently sized elements.
  const double local_L1norm{err.Norml1()};
  const double local_infnorm{err.Normlinf()};
  double local_L2norm{err.Norml2()};
  local_L2norm *= local_L2norm;

  double L1norm;
  double L2norm;
  double infnorm;
  double rmsnorm;
  int total_nVert;

  MPI_Allreduce(&local_L1norm, &L1norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_L2norm, &L2norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&nVert, &total_nVert, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_infnorm, &infnorm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  L2norm = sqrt(L2norm);
  rmsnorm = L2norm / sqrt(nVert);

  // These are computed according to the FEM definitions of these and are truly
  // "norms"/integral values defined in the space.
  FunctionCoefficient exact_sol(oned_oneg_c_fun);
  const double L1error{phi->ComputeL1Error(exact_sol)};
  const double L2error{phi->ComputeL2Error(exact_sol)};
  const double inferror{phi->ComputeMaxError(exact_sol)};

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
  {
    os << "\n=== ANALYZE ===\n";
    os << "reference: " << reference_enum_to_str(reference_solution) << '\n';

    if (reference_solution != fixed_src)
    {
      const double kref{analytic_keff(this, *xs)};
      const auto previous_precision{os.precision()};
      os << setprecision(20);
      os << "keff = " << keff << '\n';
      os << "kref = " << kref << '\n';
      os << "keff_err_[pcm] = " << (kref - keff) * 1e5 << '\n';
      os << setprecision(previous_precision);
    }

    os << "(norms do not account for element sizes)\n";
    os << "L1norm  = " << L1norm  << '\n';
    os << "L2norm  = " << L2norm  << '\n';
    os << "rmsnorm = " << rmsnorm << '\n';
    os << "infnorm = " << infnorm << '\n';
    os << "(errors are true FEM errors)\n";
    os << "L1error  = " << L1error  << '\n';
    os << "L2error  = " << L2error  << '\n';
    os << "inferror = " << inferror << '\n';
    os << '\n';
  }
}

void Analytic::print(ostream& os) const
{
  for (int i{0}; i < nVert; ++i)
  {
    const double val{fun(this, node[i], *xs)};
    const double err{abs(sol[i] - val)};
    os << node[i](0) << ',' << node[i](1) << ',' << node[i](2) << ',' << sol[i]
       << ',' << val << ',' << err << '\n';
  }
}

Analytic::Reference Analytic::reference_str_to_enum(string str)
{
  if (str == "oned_oneg")
    return oned_oneg;
  if (str == "twod_oneg")
    return twod_oneg;
  if (str == "threed_oneg")
    return threed_oneg;
  if (str == "finite_cyl")
    return finite_cyl;
  if (str == "fixed_src")
    return fixed_src;
  if (str == "square")
    return square;
  if (str == "tworeg")
    return tworeg;
  return invalid;
}

string Analytic::reference_enum_to_str(Analytic::Reference ref)
{
  switch (ref)
  {
    case (oned_oneg): return "oned_oneg";
    case (twod_oneg): return "twod_oneg";
    case (threed_oneg): return "threed_oneg";
    case (finite_cyl): return "finite_cyl";
    case (fixed_src): return "fixed_src";
    case (square): return "square";
    case (tworeg): return "tworeg";
    default: return "invalid";
  }
}

double Analytic::fixed_src_fun(const Vertex& v, const XSLibrary& xs) const
{
  // TODO these need to come as input to the constructor
  const double sigma_tr{xs.rec[0].sigma_tr[0]};
  const double sigma_r{xs.rec[0].sigma_r[0]};
  const double q_fixed{xs.rec[0].source[0]};

  static constexpr double Lx_small{1.0};
  const double diffusion{1.0 / (3.0 * sigma_tr)};

  const double kappa{sqrt(sigma_r / diffusion)};
  const double x{v(0)}; // x is the zeroth coordinate

  return (q_fixed / diffusion)
         * (1.0 - cosh(kappa * x)
            + (cosh(kappa * Lx_small) - 1.0) / (sinh(kappa * Lx_small))
                * sinh(kappa * x));
}

double Analytic::oned_oneg_fun(const Vertex& v, const XSLibrary&) const
{
  static constexpr double Lx{100.0};
  const double x{v(0)}; // x is the zeroth coordinate
  return sin((M_PI / Lx) * x);
}

double Analytic::twod_oneg_fun(const Vertex& v, const XSLibrary&) const
{
  static constexpr double Lx{100.0};
  static constexpr double Ly{100.0};
  const double x{v(0)}; // x is the zeroth coordinate
  const double y{v(1)}; // y is the first coordinate
  return sin((M_PI / Lx) * x) * sin((M_PI / Ly) * y);
}

double Analytic::threed_oneg_fun(const Vertex& v, const XSLibrary&) const
{
  static constexpr double Lx{100.0};
  static constexpr double Ly{100.0};
  static constexpr double Lz{100.0};
  const double x{v(0)}; // x is the zeroth coordinate
  const double y{v(1)}; // y is the first coordinate
  const double z{v(2)}; // z is the first coordinate
  return sin((M_PI / Lx) * x) * sin((M_PI / Ly) * y) * sin((M_PI / Lz) * z);
}

double Analytic::finite_cyl_fun(const Vertex& v, const XSLibrary&) const
{
  static constexpr double R{50.0};
  static constexpr double H{100.0};

  const double r{sqrt(v(0) * v(0) + v(1) * v(1))}; // distance to origin
  const double z{v(2)};

  return cyl_bessel_j(0, (alpha0 / R) * r) * sin((M_PI / H) * z);
}

double Analytic::tworeg_fun(const Vertex& v, const XSLibrary& xs) const
{
  const vector<double> sigma_tr{xs.rec[0].sigma_tr[0], xs.rec[1].sigma_tr[0]};
  const vector<double> sigma_r{xs.rec[0].sigma_r[0], xs.rec[1].sigma_r[0]};
  const vector<double> nusf{xs.rec[0].nusf[1], xs.rec[1].nusf[0]};

  vector<double> diffusion(sigma_tr.size());
  for (size_t i{0}; i < sigma_tr.size(); ++i)
    diffusion[i] = 1.0 / (3.0 * sigma_tr[i]);

  static constexpr double L_F{8e1};
  static constexpr double L_R{1e2};
  static constexpr double BF{0.01716859736590981012}; // [1/cm] LF=80cm

  const double x{v(0)}; // x is the zeroth coordinate

  // indices are material indices, not group indices

  const double kappa{sqrt(sigma_r[1] / diffusion[1])};

  if (x <= L_F)
    return cos(BF * x);

  return cos(BF * L_F)
         * (cosh(kappa * (x - L_F))
            - sinh(kappa * (x - L_F)) / tanh(kappa * (L_R - L_F)));
}

double Analytic::oned_oneg_keff(const XSLibrary& xs) const
{
  const double transport{xs.rec[0].sigma_tr[0]};
  const double removal{xs.rec[0].sigma_r[0]};
  const double nusf{xs.rec[0].nusf[0]};

  static constexpr double Lx{100.0};

  const double diffusion{1.0 / (3.0 * transport)};

  return nusf / (diffusion * pow(M_PI / Lx, 2) + removal);
}

double Analytic::twod_oneg_keff(const XSLibrary& xs) const
{
  const double transport{xs.rec[0].sigma_tr[0]};
  const double removal{xs.rec[0].sigma_r[0]};
  const double nusf{xs.rec[0].nusf[0]};

  static constexpr double Lx{100.0};
  static constexpr double Ly{100.0};

  const double diffusion{1.0 / (3.0 * transport)};

  return nusf / (diffusion * (pow(M_PI / Lx, 2) + pow(M_PI / Ly, 2)) + removal);
}

double Analytic::threed_oneg_keff(const XSLibrary& xs) const
{
  const double transport{xs.rec[0].sigma_tr[0]};
  const double removal{xs.rec[0].sigma_r[0]};
  const double nusf{xs.rec[0].nusf[0]};

  static constexpr double Lx{100.0};
  static constexpr double Ly{100.0};
  static constexpr double Lz{100.0};

  const double diffusion{1.0 / (3.0 * transport)};

  return nusf
         / (diffusion
              * (pow(M_PI / Lx, 2) + pow(M_PI / Ly, 2) + pow(M_PI / Lz, 2))
            + removal);
}

double Analytic::finite_cyl_keff(const XSLibrary& xs) const
{
  const double transport{xs.rec[0].sigma_tr[0]};
  const double removal{xs.rec[0].sigma_r[0]};
  const double nusf{xs.rec[0].nusf[0]};

  static constexpr double R{50.0};
  static constexpr double H{100.0};

  const double diffusion{1.0 / (3.0 * transport)};

  return nusf / (diffusion * (pow(alpha0 / R, 2) + pow(M_PI / H, 2)) + removal);
}

double Analytic::tworeg_keff(const XSLibrary& xs) const
{
  const vector<double> sigma_tr{xs.rec[0].sigma_tr[0], xs.rec[1].sigma_tr[0]};
  const vector<double> sigma_r{xs.rec[0].sigma_r[0], xs.rec[1].sigma_r[0]};
  const vector<double> nusf{xs.rec[0].nusf[1], xs.rec[1].nusf[0]};

  vector<double> diffusion(sigma_tr.size());
  for (size_t i{0}; i < sigma_tr.size(); ++i)
    diffusion[i] = 1.0 / (3.0 * sigma_tr[i]);

  static constexpr double BF{0.01716859736590981012}; // [1/cm] LF=80cm

  // indices are material indices, not group indices
  return nusf[0] / (diffusion[0] * pow(BF, 2) + sigma_r[0]);
}

double oned_oneg_c_fun(const Vector& v)
{
  static constexpr double Lx{100.0};
  const double x{v[0]};
  return sin((M_PI/Lx) * x);
}
