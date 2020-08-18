#include "analytic.hpp"

#include <cmath>
#include <vector>
#define _USE_MATH_DEFINES

#include <iostream>
#include <string>

using namespace mfem;

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
using std::sin;
using std::sinh;
using std::tanh;

Analytic::Analytic(const Reference ref, Result& res, const XSLibrary& xslib,
                   const Mesh* const mesh)
  : reference_solution{ref}, nVert{mesh->GetNV()}, nDim{mesh->SpaceDimension()},
    keff{res.keff}, phi{&res.phi[0]}
{
  Analytic_solutions::xs_global = &xslib;

  if (reference_solution == oned_twog) phi2 = &res.phi[1];

  // set node
  Vector vec_node;
  mesh->GetVertices(vec_node);

  // put node in a useful format
  node.reserve(nVert);
  for (int i{0}; i < nVert; ++i)
  {
    node.push_back(Vector(nDim));
    for (int j{0}; j < nDim; ++j) node[i][j] = vec_node[j * nVert + i];
  }

  // store a writable pointer
  if (reference_solution != fixed_src) normalize_solution_flux();

  // set fun to appropriate solution function
  switch (ref)
  {
    case (fixed_src): fun = &Analytic_solutions::fixed_src_fun; break;
    case (oned_oneg):
      fun           = &Analytic_solutions::oned_oneg_fun;
      analytic_keff = &Analytic_solutions::oned_oneg_keff;
      break;
    case (twod_oneg): // FALL THROUGH
    case (square):
      fun           = &Analytic_solutions::twod_oneg_fun;
      analytic_keff = &Analytic_solutions::twod_oneg_keff;
      break;
    case (threed_oneg):
      fun           = &Analytic_solutions::threed_oneg_fun;
      analytic_keff = &Analytic_solutions::threed_oneg_keff;
      break;
    case (finite_cyl):
      fun           = &Analytic_solutions::finite_cyl_fun;
      analytic_keff = &Analytic_solutions::finite_cyl_keff;
      break;
    case (tworeg):
      fun           = &Analytic_solutions::tworeg_fun;
      analytic_keff = &Analytic_solutions::tworeg_keff;
      break;
    case (oned_twog):
      fun           = &Analytic_solutions::oned_oneg_fun; // same profile shape
      analytic_keff = &Analytic_solutions::oned_twog_keff;
      break;
    default:
      std::cerr << reference_enum_to_str(reference_solution) << std::endl;
      throw Invalid_reference_name{};
  }
}

void Analytic::analyze(ostream& os) const
{
  Vector err(phi->Size());
  for (int i{0}; i < err.Size(); ++i) err[i] = fun(node[i]);
  err -= *phi;

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
  MPI_Allreduce(&local_infnorm, &infnorm, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  L2norm  = sqrt(L2norm);
  rmsnorm = L2norm / sqrt(nVert);

  // These are computed according to the FEM definitions of these and are truly
  // "norms"/integral values defined in the space.
  FunctionCoefficient exact_sol(fun);
  const double L1error{phi->ComputeL1Error(exact_sol)};
  const double L2error{phi->ComputeL2Error(exact_sol)};
  const double inferror{phi->ComputeMaxError(exact_sol)};

  // If this is a two group problem, calculate the ratio.
  // Requries accumulating accross processes.
  double ratio_calc{0.0};
  if (phi2)
  {
    const double local_ratio_calc{phi2->Max()};
    MPI_Allreduce(&local_ratio_calc, &ratio_calc, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
  }

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
  {
    os << "\n=== ANALYZE ===\n";
    os << "reference: " << reference_enum_to_str(reference_solution) << '\n';

    if (reference_solution != fixed_src)
    {
      const double kref{analytic_keff()};
      const auto previous_precision{os.precision()};
      os << setprecision(20);
      os << "keff = " << keff << '\n';
      os << "kref = " << kref << '\n';
      os << "keff_err_[pcm] = " << (kref - keff) * 1e5 << '\n';
      os << setprecision(previous_precision);
    }

    if (reference_solution == oned_twog)
    {
      const auto previous_precision{os.precision()};
      const double ratio_ref{Analytic_solutions::oned_twog_ratio()};
      os << "RATIO:" << '\n';
      os << setprecision(20);
      os << "ratio_calc = " << ratio_calc << '\n';
      os << "ratio_ref  = " << ratio_ref << '\n';
      os << "ratio_err  = " << (ratio_ref - ratio_calc) << '\n';
      os << setprecision(previous_precision);
      os << "(errors only for fast group)\n";
    }

    os << "(norms do not account for element sizes)\n";
    os << "L1norm  = " << L1norm << '\n';
    os << "L2norm  = " << L2norm << '\n';
    os << "rmsnorm = " << rmsnorm << '\n';
    os << "infnorm = " << infnorm << '\n';
    os << "(errors are true FEM errors)\n";
    os << "L1error  = " << L1error << '\n';
    os << "L2error  = " << L2error << '\n';
    os << "inferror = " << inferror << '\n';
    os << '\n';
  }
}

void Analytic::print(ostream& os) const
{
  double* access{phi->GetData()};
  for (int i{0}; i < nVert; ++i, ++access)
  {
    const double val{fun(node[i])};
    const double err{abs(*access - val)};
    for (int j{0}; j < nDim; ++j) os << node[i][j] << ',';
    if (nDim == 2) os << 0.0 << ',';
    os << *access << ',' << val << ',' << err << '\n';
  }
}

Analytic::Reference Analytic::reference_str_to_enum(string str)
{
  if (str == "oned_oneg") return oned_oneg;
  if (str == "twod_oneg") return twod_oneg;
  if (str == "threed_oneg") return threed_oneg;
  if (str == "finite_cyl") return finite_cyl;
  if (str == "fixed_src") return fixed_src;
  if (str == "square") return square;
  if (str == "tworeg") return tworeg;
  if (str == "oned_twog") return oned_twog;
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
    case (oned_twog): return "oned_twog";
    default: return "invalid";
  }
}

void Analytic::normalize_solution_flux() const
{
  double* access{phi->GetData()};
  for (int i{0}; i < phi->Size(); ++i, ++access) *access = abs(*access);
  const double local_phi_max{phi->Max()};
  double phi_max;
  MPI_Allreduce(&local_phi_max, &phi_max, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  *phi /= phi_max;
  if (phi2) *phi2 /= phi_max;
}

namespace Analytic_solutions
{

  const XSLibrary* xs_global{nullptr};

  double fixed_src_fun(const Vector& v)
  {
    const double sigma_tr{xs_global->rec[0].sigma_tr[0]};
    const double sigma_r{xs_global->rec[0].sigma_r[0]};
    const double q_fixed{xs_global->rec[0].source[0]};

    static constexpr double Lx_small{1.0};
    const double diffusion{1.0 / (3.0 * sigma_tr)};

    const double kappa{sqrt(sigma_r / diffusion)};
    const double x{v[0]}; // x is the zeroth coordinate

    return (q_fixed / diffusion)
           * (1.0 - cosh(kappa * x)
              + (cosh(kappa * Lx_small) - 1.0) / (sinh(kappa * Lx_small))
                * sinh(kappa * x));
  }

  double oned_oneg_fun(const Vector& v)
  {
    const double x{v[0]};
    constexpr double Lx{100.0};
    return sin((M_PI / Lx) * x);
  }

  double twod_oneg_fun(const Vector& v)
  {
    static constexpr double Lx{100.0};
    static constexpr double Ly{100.0};
    const double x{v[0]}; // x is the zeroth coordinate
    const double y{v[1]}; // y is the first coordinate
    return sin((M_PI / Lx) * x) * sin((M_PI / Ly) * y);
  }

  double threed_oneg_fun(const Vector& v)
  {
    static constexpr double Lx{100.0};
    static constexpr double Ly{100.0};
    static constexpr double Lz{100.0};
    const double x{v[0]}; // x is the zeroth coordinate
    const double y{v[1]}; // y is the first coordinate
    const double z{v[2]}; // z is the first coordinate
    return sin((M_PI / Lx) * x) * sin((M_PI / Ly) * y) * sin((M_PI / Lz) * z);
  }

  // Note: this may not be the most efficient implementation but it works and
  // only the most minimal dependencies. The C++ implmentation of cyl_bessel_j0
  // requires C++17 which is not compatible with many compilers including nvcc
  // and IBM's xl compiler series.
  double bessel_j0(double x)
  {
#if __cplusplus >= 201703L
    return std::cyl_bessel_j(0, x);
#else
    constexpr double abs_tol{1e-12};
    constexpr int max_iter{100};

    double fct{1.0};
    double sum{0.0};

    for (int k{0}; k < max_iter; fct*=++k)
    {
      const double term{std::pow(-1,k) * std::pow(0.5*x, 2*k) / std::pow(fct,2)};
      sum += term;
      if (std::abs(term) <= abs_tol)
        return sum;
    }
    return sum;
#endif
  }

  double finite_cyl_fun(const Vector& v)
  {
    static constexpr double R{50.0};
    static constexpr double H{100.0};

    const double r{sqrt(v[0] * v[0] + v[1] * v[1])}; // distance to origin
    const double z{v[2]};

    return bessel_j0((alpha0 / R) * r) * sin((M_PI / H) * z);
  }

  double tworeg_fun(const Vector& v)
  {
    const XSRecord& fuel{xs_global->rec[0]};
    const XSRecord& reflector{xs_global->rec[1]};
    const vector<double> sigma_tr{fuel.sigma_tr[0], reflector.sigma_tr[0]};
    const vector<double> sigma_r{fuel.sigma_r[0], reflector.sigma_r[0]};
    const vector<double> nusf{fuel.nusf[0], reflector.nusf[0]};

    vector<double> diffusion(sigma_tr.size());
    for (size_t i{0}; i < diffusion.size(); ++i)
      diffusion[i] = 1.0 / (3.0 * sigma_tr[i]);

    static constexpr double L_F{8e1};
    static constexpr double L_R{1e2};
    static constexpr double BF{0.01716859736590981012}; // [1/cm] LF=80cm

    const double x{v[0]}; // x is the zeroth coordinate

    // indices are material indices, not group indices

    const double kappa{sqrt(sigma_r[1] / diffusion[1])};

    if (x <= L_F) return cos(BF * x);

    return cos(BF * L_F)
           * (cosh(kappa * (x - L_F))
              - sinh(kappa * (x - L_F)) / tanh(kappa * (L_R - L_F)));
  }

  double oned_oneg_keff()
  {
    const double transport{xs_global->rec[0].sigma_tr[0]};
    const double removal{xs_global->rec[0].sigma_r[0]};
    const double nusf{xs_global->rec[0].nusf[0]};

    static constexpr double Lx{100.0};

    const double diffusion{1.0 / (3.0 * transport)};

    return nusf / (diffusion * pow(M_PI / Lx, 2) + removal);
  }

  double twod_oneg_keff()
  {
    const double transport{xs_global->rec[0].sigma_tr[0]};
    const double removal{xs_global->rec[0].sigma_r[0]};
    const double nusf{xs_global->rec[0].nusf[0]};

    static constexpr double Lx{100.0};
    static constexpr double Ly{100.0};

    const double diffusion{1.0 / (3.0 * transport)};

    return nusf
           / (diffusion * (pow(M_PI / Lx, 2) + pow(M_PI / Ly, 2)) + removal);
  }
  double threed_oneg_keff()
  {
    const double transport{xs_global->rec[0].sigma_tr[0]};
    const double removal{xs_global->rec[0].sigma_r[0]};
    const double nusf{xs_global->rec[0].nusf[0]};

    static constexpr double Lx{100.0};
    static constexpr double Ly{100.0};
    static constexpr double Lz{100.0};

    const double diffusion{1.0 / (3.0 * transport)};

    return nusf
           / (diffusion
              * (pow(M_PI / Lx, 2) + pow(M_PI / Ly, 2) + pow(M_PI / Lz, 2))
              + removal);
  }

  double finite_cyl_keff()
  {
    const double transport{xs_global->rec[0].sigma_tr[0]};
    const double removal{xs_global->rec[0].sigma_r[0]};
    const double nusf{xs_global->rec[0].nusf[0]};

    static constexpr double R{50.0};
    static constexpr double H{100.0};

    const double diffusion{1.0 / (3.0 * transport)};

    return nusf
           / (diffusion * (pow(alpha0 / R, 2) + pow(M_PI / H, 2)) + removal);
  }

  double tworeg_keff()
  {
    const XSRecord& fuel{xs_global->rec[0]};
    const XSRecord& reflector{xs_global->rec[1]};
    const vector<double> sigma_tr{fuel.sigma_tr[0], reflector.sigma_tr[0]};
    const vector<double> sigma_r{fuel.sigma_r[0], reflector.sigma_r[0]};
    const vector<double> nusf{fuel.nusf[0], reflector.nusf[0]};

    vector<double> diffusion(sigma_tr.size());
    for (size_t i{0}; i < diffusion.size(); ++i)
      diffusion[i] = 1.0 / (3.0 * sigma_tr[i]);

    static constexpr double BF{0.01716859736590981012}; // [1/cm] LF=80cm

    // indices are material indices, not group indices
    return nusf[0] / (diffusion[0] * pow(BF, 2) + sigma_r[0]);
  }

  double oned_twog_keff()
  {
    static constexpr double Lx{100.0};

    const XSRecord& rec{xs_global->rec[0]};

    vector<double> diffusion(xs_global->ngroup());
    for (size_t g{0}; g < diffusion.size(); ++g)
      diffusion[g] = 1.0 / (3.0 * rec.sigma_tr[g]);

    const double bsq{pow(M_PI / Lx, 2)};
    const double k2k1{oned_twog_ratio()};
    return (rec.nusf[0] + rec.nusf[1] * k2k1)
           / (diffusion[0] * bsq + rec.sigma_r[0]);
  }

  double oned_twog_ratio()
  {
    static constexpr double Lx{100.0};
    const XSRecord& rec{xs_global->rec[0]};
    const double bsq{pow(M_PI / Lx, 2)};

    vector<double> diffusion(xs_global->ngroup());
    for (size_t g{0}; g < diffusion.size(); ++g)
      diffusion[g] = 1.0 / (3.0 * rec.sigma_tr[g]);

    return rec.sigma_s.Elem(1, 0) / (diffusion[1] * bsq + rec.sigma_r[1]);
  }
} // namespace Analytic_solutions
