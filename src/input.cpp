#include "input.hpp"

#include <iostream>
using std::cerr;

using std::ifstream;
using std::ostream;
using std::string;
using std::vector;

using namespace mfem;

Input::Input(std::string fname) : input_file_str{fname}
{
  input_file = input_file_str.c_str();
}

Input::Input(const char* fname) : input_file_str{fname}
{
  input_file = input_file_str.c_str();
}

void Input::open()
{
  ifs.open(input_file_str);
  if (!ifs) throw Invalid_file_name{input_file_str};
  stub = path_stub(input_file_str);
}

void Input::read_core()
{
  string card;

  ifs >> card;
  if (card != "[CORE]")
    throw Invalid_input_card{"Expected [CORE]. Got: " + card};

  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n');) continue;
    }
    else if (card == "mesh_file")
    {
      ifs >> mesh_file_str;
      mesh_file_str = stub + mesh_file_str;
      mesh_file     = mesh_file_str.c_str();
    }
    else if (card == "restart_mesh")
    {
      restart.mesh = read_bool(ifs);
    }
    else if (card == "use_full_assembly")
    { 
      // tensor assembly of ParLinearForm in solver_saaf_angle_power_iteration
      use_full_assembly = read_bool(ifs);
    }
    else if (card == "boundary_conditions")
    {
      read_vector(ifs, boundary_conditions);
    }
    else if (card == "order")
    {
      ifs >> order;
    }
    else if (card == "refine_serial")
    {
      ifs >> refine_serial;
    }
    else if (card == "refine_parallel")
    {
      ifs >> refine_parallel;
    }
    else if (card == "userxs")
    {
      ifs >> xs_file;
      xs_file   = stub + xs_file;
      xs_format = XSLibrary::userxs;
    }
    else if (card == "hdf5xs")
    {
      ifs >> hdf5xs_lib_name >> xs_file;
      ;
      xs_file   = stub + xs_file;
      xs_format = XSLibrary::hdf5xs;
    }
    else if (card == "reference")
    {
      ifs >> reference_str;
      ref       = Analytic::reference_str_to_enum(reference_str);
      reference = reference_str.c_str();
    }
    else if (card == "device")
    {
      ifs >> device_str;
      device = device_str.c_str();
    }
    else if (card == "static_condensation")
    {
      static_condensation = read_bool(ifs);
    }
    else if (card == "verbose")
    {
      verbose = read_bool(ifs);
    }
    else if (card == "[MATERIAL]")
    {
      // expecting to lead into [MATERIAL] block
      // cannot lead directly into [STATE] block
      reset_block(ifs, "[MATERIAL]");
      break;
    }
    else
    {
      cerr << "INVALID CARD\n";
      cerr << card << '\n';
      throw Invalid_input_card{card};
    }
  }
}

int Input::nstate()
{
  if (number_of_state != 0)
    return number_of_state;

  // mark where i start taking things out of the stream
  const long int position{ifs.tellg()};

  string card;
  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n');) continue;
    }
    else if (card == "[STATE]")
    {
      ++number_of_state;
    }
  }

  // return to where i started
  ifs.clear();
  ifs.seekg(position, ifs.beg);

  return number_of_state;
}

void Input::read_state()
{
  string card;

  ifs >> card;
  if (card != "[STATE]")
    throw Invalid_input_card{"Expected [STATE]. Got: " + card};

  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n');) continue;
    }
    else if (card == "max_outer")
    {
      ifs >> tol.max_outer;
    }
    else if (card == "k_tol")
    {
      ifs >> tol.k;
    }
    else if (card == "phi_tol")
    {
      ifs >> tol.phi;
    }
    else if (card == "max_inner")
    {
      ifs >> tol.max_inner;
    }
    else if (card == "inner_rel_tol")
    {
      ifs >> tol.inner_rel;
    }
    else if (card == "inner_abs_tol")
    {
      ifs >> tol.inner_abs;
    }
    else if (card == "temp_max_outer")
    {
      ifs >> tol.temp_max_outer;
    }
    else if (card == "temp_tol")
    {
      ifs >> tol.temp;
    }
    else if (card == "temp_max_inner")
    {
      ifs >> tol.temp_max_inner;
    }
    else if (card == "temp_inner_rel")
    {
      ifs >> tol.temp_inner_rel;
    }
    else if (card == "temp_inner_abs")
    {
      ifs >> tol.temp_inner_abs;
    }
    else if (card == "max_scatter")
    {
      ifs >> tol.max_scatter;
    }
    else if (card == "scatter_tol")
    {
      ifs >> tol.scatter;
    }
    else if (card == "use_dsa")
    {
      tol.use_dsa = read_bool(ifs);
    }
    else if (card == "solver")
    {
      ifs >> solver_str;
      sol    = solver_choice_str_to_enum(solver_str);
      solver = solver_str.c_str();
    }
    else if (card == "restart_out")
    {
      restart.out = read_bool(ifs);
    }
    else if (card == "restart_flux")
    {
      restart.flux = read_bool(ifs);
    }
    else if (card == "restart_runtime")
    {
      ifs >> restart.runtime;
    }
    else if (card == "flux_free_chi")
    {
      flux_free_chi = read_bool(ifs);
    }
    else if (card == "spn")
    {
      ifs >> spn;
      if (((spn + 1) % 2) != 0) throw Invalid_input_card{"SPN must be odd."};
    }
    else if (card == "quadrature")
    {
      ifs >> quadrature_str;
      quadrature_type = quadrature_choice_str_to_enum(quadrature_str);
      quadrature = quadrature_str.c_str();
    }
    else if (card == "quad_order")
    {
      ifs >> quad_order;
      if ((quad_order % 2) != 0) 
        throw Invalid_input_card{"Quadrature order must be even."};
    }
    else if (card == "quad_legendre_order")
    {
      ifs >> quad_legendre_order;
      if ((quad_order % 2) != 0) 
        throw Invalid_input_card{"Legendre quadrature order must be even."};
    }
    else if (card == "quad_chebyschev_order")
    {
      ifs >> quad_chebyschev_order;
      if ((quad_order % 2) != 0) 
        throw Invalid_input_card{"Chebyschev quadrature order must be even."};
    }
    else if (card == "rxpwr")
    {
      ifs >> rxpwr;
    }
    else if (card == "temp_boundary")
    {
      ifs >> temp_boundary;
    }
    else if (card == "temp_heat_pipe")
    {
      ifs >> temp_heat_pipe;
    }
    else if (card == "linear_solver")
    {
      ifs >> str_linear_solver;
    }
    else if (card == "[STATE]")
    {
      // prepare to head into another state block
      reset_block(ifs, "[STATE]");
      break;
    }
    else
    {
      cerr << "INVALID CARD\n";
      cerr << card << '\n';
      throw Invalid_input_card{card};
    }
  }
}

// The MFEM command-line argument parser can set the const char* values but this
// does not update the local strings or the enums themselves.
void Input::reconcile_enums()
{
  reference_str = reference;
  ref           = Analytic::reference_str_to_enum(reference_str);
  if ((reference_str != "invalid") && (ref == Analytic::invalid))
    throw Invalid_input_card{"Invalid reference name: " + reference_str};

  solver_str = solver;
  sol        = solver_choice_str_to_enum(solver_str);
}

void Input::echo(ostream& os) const
{
  os << "[CORE]\n";

  os << "mesh_file " << mesh_file << '\n';
  os << "boundary_conditions";
  for (int i{0}; i < boundary_conditions.Size(); ++i)
    os << ' ' << boundary_conditions[i];
  os << '\n';
  os << "order " << order << '\n';
  os << "refine_serial " << refine_serial << '\n';
  os << "refine_parallel " << refine_parallel << '\n';

  os << "xs_file " << xs_file << '\n';

  if (ref != Analytic::invalid) os << "reference " << reference << '\n';

  os << "device " << device << '\n';
  os << "static_condensation " << static_condensation << '\n';

  os << "\n[STATE]\n";
  os << "verbose " << verbose << '\n';
  os << "max_inner " << tol.max_inner << '\n';
  os << "inner_rel_tol " << tol.inner_rel << '\n';
  os << "inner_abs_tol " << tol.inner_abs << '\n';
  os << "max_outer " << tol.max_outer << '\n';
  os << "k_tol " << tol.k << '\n';
  os << "phi_tol " << tol.phi << '\n';
  os << "solver " << solver_choice_enum_to_str(sol) << '\n';
  os << "flux_free_chi " << flux_free_chi << '\n';
}

string Input::path_stub(string path)
{
  string stub;
  // assume absolute paths start with ~ or /
  if ((path[0] == '~') || (path[0] == '/')) return stub;
  // return up-to-and-including the final /
  const auto final_slash{path.find_last_of('/')};
  stub = path.substr(0, final_slash + 1);
  return stub;
}

void Input::reset_block(ifstream& ifs, const string& block_head)
{
  for (size_t i{0}; i < block_head.size(); ++i) ifs.unget();
}

bool Input::read_bool(ifstream& ifs)
{
  string str;
  ifs >> str;
  for (char& c : str) c = tolower(c);
  return ((str == "1") || (str == "t") || (str == "true"));
}

void Input::read_vector(ifstream& ifs, Array<int>& arr)
{
  for (int val; ifs >> val;) arr.Append(val);
  ifs.unget();
  ifs.clear();
}

string Input::concatenate_path(const string& fname) const
{
  return stub + fname;
}

OptionsParser custom_arguments(Input& inp, int argc, char* argv[])
{
  OptionsParser args(argc, argv);

  // general
  args.AddOption(&inp.input_file, "-i", "--input", "Input file.",
                 true); // required

  // CORE
  args.AddOption(&inp.mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&inp.boundary_conditions, "-bc", "--boundary-conditions",
                 "Boundary conditions associated with mesh.");
  args.AddOption(&inp.order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&inp.refine_serial, "-rs", "--refine-serial",
                 "Refinements to perform on serial mesh.");
  args.AddOption(&inp.refine_parallel, "-rp", "--refine-parallel",
                 "Refinements to perform on parallel mesh.");
  // note: xs file cannot be specified as hdf5xs takes two parameters
  args.AddOption(&inp.reference, "-ref", "--reference",
                 "Analytic reference solution.");
  args.AddOption(&inp.device, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&inp.static_condensation, "-sc", "--static-condensation",
                 "-no-sc", "--no-static-condensation",
                 "Enable static condensation.");
  args.AddOption(&inp.verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                 "Enable verbose output.");

  // GPU
  args.AddOption(&inp.ndevices, "-nd", "--ndevices",
                 "Number of GPU devices in pool that will be used in MPITeams.");

  // Note: STATE variables cannot be input from the command line as they will be
  // overwritten by reading the STATE block in the input file at for the first
  // state.

  return args;
}

string Input::input_file_stub()
{
  if (input_file_stub_str != "") return input_file_stub_str;

  input_file_stub_str = input_file_str;

  const size_t last_slash{input_file_stub_str.find_last_of("/")};
  if (last_slash != input_file_stub_str.length()) 
    input_file_stub_str = input_file_stub_str.substr(last_slash+1, input_file_stub_str.length());
  const size_t last_dot{input_file_stub_str.find_last_of(".")};
  input_file_stub_str = input_file_stub_str.substr(0, last_dot);

  return input_file_stub_str;
}
