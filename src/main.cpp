#include "analytic.hpp"
#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"

#include "solver.hpp"
#include "solver_amgx.h"
#include "conduction.hpp"

#include "timer.hpp"
#include "xslibrary.hpp"

#include "quadrature.hpp"
#include "level_symmetric_quadrature.hpp"
#include "legendre_chebyschev_quadrature.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace mfem;

void version(std::ostream& os); // forward declaration

int main(int argc, char* argv[])
{

  Timer time;

  // Initialize MPI
  int num_procs;
  int myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0) { version(cout); }

  // 1. Parse command-line options.
  // Do our own parsing to get the input file name.
  string input_file;
  for (int i{1}; i < argc; ++i)
  {
    if (((strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "--input") == 0))
        && (argc >= (i + 1)))
    {
      if (argv[i+1])
        input_file = argv[i+1];
      break;
    }
  }

  // read the input file
  Input inp{input_file};
  if (!input_file.empty())
  {
    inp.open();
    inp.read_core();
  }

  // parse command-line options using MFEM
  OptionsParser args{custom_arguments(inp, argc, argv)};
  args.Parse();

  if (!args.Good())
  {
    if (myid == 0) args.PrintUsage(cout);
    MPI_Finalize();
    return 1;
  }

  // need to reconcile enums to match their const char* counterparts
  inp.reconcile_enums();

  if ((myid == 0) && (inp.verbose)) args.PrintOptions(cout);

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.

  std::string device_str{inp.device};
  device_str += ",omp";
  Device device(device_str);
  if (myid == 0) device.Print();
#ifdef GPU_AWARE_MPI
  if (device.IsEnabled()) device.SetGPUAwareMPI();
#endif

  // core block is finished, you can read cross sections from file
  // xs.read_userxs(inp.xs_file);
  XSLibrary xs;
  switch (inp.xs_format)
  {
    case (XSLibrary::userxs): xs.read_userxs(inp.xs_file); break;
    case (XSLibrary::hdf5xs):
      xs.read_hdf5xs(inp.xs_file, inp.hdf5xs_lib_name);
      break;
    default: cerr << "Invalid xs format" << endl; return 1;
  }

  // core block is finished, you can read the material block
  Material_library mat_lib{xs};
  mat_lib.read_material(inp.ifs);

  time.start("Mesh processing");

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  Mesh mesh{inp.mesh_file, 1, 1};
  int dim = mesh.Dimension();

  // refine the mesh in serial
  for (int lvl{0}; lvl < inp.refine_serial; ++lvl) mesh.UniformRefinement();

  // create a parallel mesh and refine as necessary
  ParMesh pmesh{MPI_COMM_WORLD, mesh};
  for (int lvl{0}; lvl < inp.refine_parallel; ++lvl) pmesh.UniformRefinement();

  time.stop();

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection* fec{nullptr};
  if (inp.order > 0)
  { 
    fec = new H1_FECollection(inp.order, dim);
  }
  else if (pmesh.GetNodes())
  {
    fec = pmesh.GetNodes()->OwnFEC();
    if (myid == 0) cout << "Using isoparametric FEs: " << fec->Name() << endl;
  }
  else
  {
    fec = new H1_FECollection(inp.order = 1, dim);
  }

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_bdr(pmesh.bdr_attributes.Max());
  // begin by setting all bcs to essential
  ess_bdr = 1;

  // go through and set boundary conditions based on user input
  for (int i{0}; (i < inp.boundary_conditions.Size()) && (i < ess_bdr.Size());
       ++i)
    ess_bdr[i] = inp.boundary_conditions[i];

  // TODO more elegant...
  const auto position{inp.ifs.tellg()};
  inp.read_state();
  inp.ifs.clear();
  inp.ifs.seekg(position);

  const bool is_transport{((inp.sol == Solver_choice::epdo_power_iteration) ||
    (inp.sol == Solver_choice::sfent_power_iteration) ||
    (inp.sol == Solver_choice::saaf_power_iteration) ||
    (inp.sol == Solver_choice::saaf_angle_power_iteration) ||
    (inp.sol == Solver_choice::saaf_slepc_vector) ||
    (inp.sol == Solver_choice::saaf_true_slepc_vector) || 
    (inp.sol == Solver_choice::saaf_true_amgx_vector))};

  std::unique_ptr<Quadrature> quad;
  switch (inp.quadrature_type)
  {
    case (Quadrature_choice::level_symmetric):
      quad =
        std::make_unique<Level_symmetric_quadrature>(inp.quad_order,
        pmesh.SpaceDimension());
      break;
    case (Quadrature_choice::legendre_chebyschev):
      if ((inp.quad_legendre_order > 0) && (inp.quad_chebyschev_order > 0))
        quad =
          std::make_unique<Legendre_chebyschev_quadrature>(inp.quad_legendre_order,
          inp.quad_chebyschev_order, pmesh.SpaceDimension());
      else
        quad = std::make_unique<Legendre_chebyschev_quadrature>(inp.quad_order,
          pmesh.SpaceDimension());
      break;
    default:
      if (is_transport)
      {
        std::cout << "Invalid quadrature and quadrature is required for both EPDO and SFENT." << std::endl;
        return 1;
      }
  }

  int vdim{0};
  switch (inp.sol)
  {
    case (Solver_choice::slepc_vector):
    case (Solver_choice::amgx_vector):
      vdim = mat_lib.ngroup()*inp.spn_eq();
      break;
    case (Solver_choice::saaf_slepc_vector):
    case (Solver_choice::saaf_true_slepc_vector):
    case (Solver_choice::saaf_true_amgx_vector):
      vdim = mat_lib.ngroup()*quad->nd();
      break;
    case (Solver_choice::saaf_angle_power_iteration):
      vdim = quad->nd();
      break;
    default:
      vdim = 1;
      break;
  }

  ParFiniteElementSpace fespace{&pmesh, fec, vdim, Ordering::byVDIM};
  ParFiniteElementSpace scalar_flux_fespace{&pmesh, fec};

  long long dof{fespace.GlobalVSize()};
  switch (inp.sol)
  {
    case (Solver_choice::slepc_vector):
    case (Solver_choice::saaf_slepc_vector):
    case (Solver_choice::saaf_true_slepc_vector):
    case (Solver_choice::amgx_vector):
    case (Solver_choice::saaf_true_amgx_vector):
      break;
    case (Solver_choice::saaf_angle_power_iteration):
      dof *= mat_lib.ngroup();
      break;
    case (Solver_choice::saaf_power_iteration):
    case (Solver_choice::sfent_power_iteration):
    case (Solver_choice::epdo_power_iteration):
      dof *= mat_lib.ngroup()*quad->nd();
      break;
    default:
      dof *= mat_lib.ngroup()*inp.spn_eq();
      break;
  }

  if (myid == 0) cout << "Number of finite element unknowns: " << dof << endl;

  // TODO this should move outside of the [STATE] loop but we need to disallow
  // changing the solver type between states (maybe move to [CORE] block)
  int flux_length{0};
  switch (inp.sol)
  {
    case (Solver_choice::slepc_vector):
    case (Solver_choice::saaf_slepc_vector):
    case (Solver_choice::saaf_true_slepc_vector):
    case (Solver_choice::amgx_vector):
    case (Solver_choice::saaf_true_amgx_vector):
      flux_length = 1;
      break;
    default:
      flux_length = mat_lib.ngroup();
      break;
  }

  int moment_angle_length{0};
  switch (inp.sol)
  {
    case (Solver_choice::slepc_vector):
    case (Solver_choice::saaf_slepc_vector):
    case (Solver_choice::saaf_true_slepc_vector):
    case (Solver_choice::saaf_angle_power_iteration):
    case (Solver_choice::amgx_vector):
    case (Solver_choice::saaf_true_amgx_vector):
      moment_angle_length = 1;
      break;
    case (Solver_choice::saaf_power_iteration):
    case (Solver_choice::sfent_power_iteration):
    case (Solver_choice::epdo_power_iteration):
      moment_angle_length = quad->nd();
      break;
    default:
      moment_angle_length = inp.spn_eq();
      break;
  }

  Result res{&fespace, flux_length, moment_angle_length};
  Result scalar_flux{&scalar_flux_fespace, mat_lib.ngroup()};
  ParGridFunction power{&scalar_flux_fespace};
  ParGridFunction temperature{&scalar_flux_fespace};

  /* TODO SUM
  ofstream sum{inp.input_file_stub() + ".sum"};
  sum << "=== STATEPOINT SUMMARY ===\n" << "state, spn_order, keff\n";
  */

  for (int state{0}; state < inp.nstate(); ++state)
  {
    inp.read_state();
    // TODO probably need to abandon [STATE] vs [CORE] for now...
    //res.resize(mat_lib.ngroup(), inp.spn_eq()); // TODO this is invalid...

    // do the solving

    if (inp.ref == Analytic::fixed_src)
    {
      diffusion_fixed_src(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
    }
    else
    {
      switch (inp.sol)
      {
        case (Solver_choice::lobpcg):
          diffusion_hypre_lobpcg(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::power_iteration):
          spn_power_iteration(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::amgx_power_iteration):
          spn_amgx_power_iteration(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::slepc_vector):
          spn_slepc_vector(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res, scalar_flux);
          calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
          temperature_heat_conduction(inp, mat_lib, power, temperature);
          break;
        case (Solver_choice::amgx_vector):
          spn_amgx_vector(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::slepc_manual):
          spn_slepc_manual(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::epdo_power_iteration):
          epdo_power_iteration(quad.get(), time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res, scalar_flux);
          break;
        case (Solver_choice::sfent_power_iteration):
          sfent_power_iteration(quad.get(), time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_power_iteration):
          saaf_power_iteration(quad.get(), time, inp, mat_lib, &pmesh, &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_angle_power_iteration):
          saaf_angle_power_iteration(quad.get(), time, inp, mat_lib, &pmesh, &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_slepc_vector):
          saaf_slepc_vector(quad.get(), time, inp, mat_lib, &pmesh, &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_true_slepc_vector):
          saaf_true_slepc_vector(quad.get(), time, inp, mat_lib, res);
          calc_scalar_flux_slepc_vector(res, quad.get(), scalar_flux);
          calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
          temperature_heat_conduction(inp, mat_lib, power, temperature);
          break;
        case (Solver_choice::saaf_true_amgx_vector):
          saaf_true_amgx_vector(quad.get(), time, inp, mat_lib, res);
          calc_scalar_flux_slepc_vector(res, quad.get(), scalar_flux);
          calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
          /* TODO finish implementation
          temperature_heat_conduction(inp, mat_lib, power, temperature);
          */
          break;
        case (Solver_choice::conduction_radial):
        case (Solver_choice::conduction_linear):
          temperature_heat_conduction(inp, mat_lib, power, temperature);
          break;
        default:
          std::cerr << "Invalid solver choice.\n";
          MPI_Finalize();
          return 1;
      }
    }

    // output

    time.start("Output");

    const auto old_precision{cout.precision()};
    if ((myid == 0) && (inp.ref != Analytic::fixed_src))
      cout << "keff = " << setprecision(20) << res.keff
           << setprecision(old_precision) << '\n';

    if (inp.ref != Analytic::invalid)
    {
      Result* gf{nullptr};
      if (is_transport)
        gf = &scalar_flux;
      else
        gf = &res;
      Analytic anal(inp.ref, inp.sol, *gf, xs, &pmesh);
      anal.analyze(cout);
      ostringstream analysis_fname;
      analysis_fname << "out_" << setfill('0') << setw(6) << myid << ".csv";
      ofstream ofs{analysis_fname.str()};
      anal.print(ofs);
    }

    ostringstream state_name;
    state_name << "STATE" << setfill('0') << setw(3) << state;

    ParaViewDataCollection output(inp.input_file_stub() + "_" + state_name.str(), &pmesh);

    const size_t group_width{std::to_string(xs.ngroup()).length()};
    size_t moment_width;
    if (is_transport)
      moment_width = std::to_string(res.nmoment()).length();
    else
      moment_width = std::to_string(2*inp.spn_eq()).length();

    if ((inp.sol == Solver_choice::conduction_radial) ||
        (inp.sol == Solver_choice::conduction_linear))
    {
      output.RegisterField("temperature", &temperature);
    }

    if ((inp.sol == Solver_choice::slepc_vector) || 
        (inp.sol == Solver_choice::saaf_slepc_vector) || 
        (inp.sol == Solver_choice::saaf_true_slepc_vector) ||
        (inp.sol == Solver_choice::amgx_vector) || 
        (inp.sol == Solver_choice::saaf_true_amgx_vector))
    {
      output.RegisterField("phi", &res(0));
      if ((inp.sol == Solver_choice::saaf_true_slepc_vector) || 
          (inp.sol == Solver_choice::slepc_vector) |\
          (inp.sol == Solver_choice::saaf_true_amgx_vector))
      {
        for (int g{0}; g < scalar_flux.ngroup(); ++g)
        {
          std::ostringstream scalar_flux_record;
          scalar_flux_record << "scalar_flux_g" 
                             << std::setfill('0')
                             << std::setw(group_width)
                             << g;
          output.RegisterField(scalar_flux_record.str(), &scalar_flux(g));
        }
        output.RegisterField("power", &power);
        output.RegisterField("temperature", &temperature);
      }
    }
    else
    {
      for (int g{0}; g < res.ngroup(); ++g)
      {
        for (int n{0}; n < res.nmoment(); ++n)
        {
          ostringstream record_name;

          if (is_transport)
            record_name << "phi_g" << setfill('0') << setw(group_width) << g
                        << "_n" << setfill('0') << setw(moment_width) << n;
          else
            record_name << "phi_g" << setfill('0') << setw(group_width) << g
                        << "_n" << setfill('0') << setw(moment_width) << 2*n;

          output.RegisterField(record_name.str(), &res(g,n));
        }
        if (is_transport)
        {
          ostringstream scalar_flux_record;
          scalar_flux_record << "scalar_flux_g" << setfill('0') 
                             << setw(group_width) << g;
          output.RegisterField(scalar_flux_record.str(), &scalar_flux(g));
        }
      }
    }

    output.SetFormat(DataCollection::PARALLEL_FORMAT);
    output.Save();

    /* TODO SUM sum << state << ',' << inp.spn_order() << ',' << res.keff << '\n'; */

    time.stop();

  }

  if (myid == 0) time.Print(cout);

  // 15. Free the used memory.

  if (inp.order > 0) delete fec;

  MPI_Finalize();

  return 0;
}
