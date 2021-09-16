#include "analytic.hpp"
#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"

#include "solver.hpp"
#include "solver_amgx.h"
#include "solver_saaf_angle_power_iteration.hpp"
#include "solver_saaf_true_slepc_vector.hpp"

#include "conduction.hpp"

#include "timer.hpp"
#include "xslibrary.hpp"

#include "quadrature.hpp"
#include "level_symmetric_quadrature.hpp"
#include "legendre_chebyschev_quadrature.hpp"

#include "output.hpp"

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

  constexpr bool restart{false};

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
  std::unique_ptr<ParMesh> pmesh;
  if (restart)
  {
    const std::string fname{MakeParFilename(inp.input_file_stub() + "_restart", myid, ".mesh")};
    std::ifstream ifs{fname};
    MFEM_VERIFY(ifs.good(), "Restart mesh file " << fname << " not found.");
    pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, ifs);
  }
  else
  {
    Mesh mesh{inp.mesh_file, 1, 1};

    // refine the mesh in serial
    for (int lvl{0}; lvl < inp.refine_serial; ++lvl) mesh.UniformRefinement();

    // create a parallel mesh and refine as necessary
    pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh);
    for (int lvl{0}; lvl < inp.refine_parallel; ++lvl) pmesh->UniformRefinement();

    const std::string fname{MakeParFilename(inp.input_file_stub() + "_restart", myid, ".mesh")};
    std::ofstream ofs{fname};
    pmesh->ParPrint(ofs);
  }

  const int dim = pmesh->SpaceDimension();

  time.stop();

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection* fec{nullptr};
  // only worry about FEC if this is not a restart problem
  // for restart problems, FEC is created and owned by PGF
  if (!restart)
  {
    if (inp.order > 0)
    { 
      fec = new H1_FECollection(inp.order, dim);
    }
    else if (pmesh->GetNodes())
    {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0) cout << "Using isoparametric FEs: " << fec->Name() << endl;
    }
    else
    {
      fec = new H1_FECollection(inp.order = 1, dim);
    }
  }

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
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
    (inp.sol == Solver_choice::saaf_angle_nda_slepc) ||
    (inp.sol == Solver_choice::saaf_angle_jacobi_power_iteration) ||
    (inp.sol == Solver_choice::saaf_slepc_vector) ||
    (inp.sol == Solver_choice::saaf_true_slepc_vector) || 
    (inp.sol == Solver_choice::saaf_split_slepc_vector) || 
    (inp.sol == Solver_choice::saaf_split_power_iteration) || 
    (inp.sol == Solver_choice::saaf_self_power_iteration) || 
    (inp.sol == Solver_choice::saaf_true_slepc_diffusion_vector) || 
    (inp.sol == Solver_choice::saaf_true_slepc_transport_vector) || 
    (inp.sol == Solver_choice::saaf_true_amgx_vector))};

  std::unique_ptr<Quadrature> quad;
  switch (inp.quadrature_type)
  {
    case (Quadrature_choice::level_symmetric):
      quad =
        std::make_unique<Level_symmetric_quadrature>(inp.quad_order,
        pmesh->SpaceDimension());
      break;
    case (Quadrature_choice::legendre_chebyschev):
      if ((inp.quad_legendre_order > 0) && (inp.quad_chebyschev_order > 0))
        quad =
          std::make_unique<Legendre_chebyschev_quadrature>(inp.quad_legendre_order,
          inp.quad_chebyschev_order, pmesh->SpaceDimension());
      else
        quad = std::make_unique<Legendre_chebyschev_quadrature>(inp.quad_order,
          pmesh->SpaceDimension());
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
    case (Solver_choice::saaf_split_slepc_vector):
    case (Solver_choice::saaf_split_power_iteration):
    case (Solver_choice::saaf_self_power_iteration):
    case (Solver_choice::saaf_true_slepc_diffusion_vector):
    case (Solver_choice::saaf_true_slepc_transport_vector):
    case (Solver_choice::saaf_true_amgx_vector):
    case (Solver_choice::saaf_angle_nda_slepc):
    case (Solver_choice::saaf_angle_jacobi_power_iteration):
      vdim = mat_lib.ngroup()*quad->nd();
      break;
    case (Solver_choice::saaf_angle_power_iteration):
      vdim = quad->nd();
      break;
    default:
      vdim = 1;
      break;
  }

  int scalar_flux_vdim{1};
  switch (inp.sol)
  {
    case (Solver_choice::saaf_true_slepc_diffusion_vector):
    case (Solver_choice::saaf_angle_nda_slepc):
    case (Solver_choice::saaf_angle_jacobi_power_iteration):
      scalar_flux_vdim = mat_lib.ngroup();
      break;
    default:
      break;
  }

  ParFiniteElementSpace fespace{pmesh.get(), fec, vdim, Ordering::byVDIM};
  ParFiniteElementSpace scalar_flux_fespace{pmesh.get(), fec, scalar_flux_vdim, Ordering::byVDIM};

  long long dof{fespace.GlobalVSize()};
  switch (inp.sol)
  {
    case (Solver_choice::slepc_vector):
    case (Solver_choice::amgx_vector):
    case (Solver_choice::saaf_slepc_vector):
    case (Solver_choice::saaf_true_slepc_vector):
    case (Solver_choice::saaf_split_slepc_vector):
    case (Solver_choice::saaf_split_power_iteration):
    case (Solver_choice::saaf_self_power_iteration):
    case (Solver_choice::saaf_true_slepc_diffusion_vector):
    case (Solver_choice::saaf_true_slepc_transport_vector):
    case (Solver_choice::saaf_true_amgx_vector):
    case (Solver_choice::saaf_angle_nda_slepc):
    case (Solver_choice::saaf_angle_jacobi_power_iteration):
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
    case (Solver_choice::saaf_split_slepc_vector):
    case (Solver_choice::saaf_split_power_iteration):
    case (Solver_choice::saaf_self_power_iteration):
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
    case (Solver_choice::saaf_split_slepc_vector):
    case (Solver_choice::saaf_split_power_iteration):
    case (Solver_choice::saaf_self_power_iteration):
    case (Solver_choice::saaf_angle_power_iteration):
    case (Solver_choice::saaf_angle_nda_slepc):
    case (Solver_choice::saaf_angle_jacobi_power_iteration):
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

  int scalar_flux_length{mat_lib.ngroup()};
  switch (inp.sol)
  {
    case (Solver_choice::saaf_true_slepc_diffusion_vector):
    case (Solver_choice::saaf_angle_nda_slepc):
    case (Solver_choice::saaf_angle_jacobi_power_iteration):
      scalar_flux_length = 1;
      break;
    default:
      break;
  }

  Result res{&fespace, flux_length, moment_angle_length};
  Result scalar_flux{&scalar_flux_fespace, scalar_flux_length};

  ParFiniteElementSpace true_scalar_fespace{pmesh.get(), fec};
  ParGridFunction power{&true_scalar_fespace};
  ParGridFunction temperature{&true_scalar_fespace};

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

    mesh_stats(pmesh.get(), std::cout, &mat_lib); // volume correct xs

    if (inp.ref == Analytic::fixed_src)
    {
      diffusion_fixed_src(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res);
    }
    else
    {
      switch (inp.sol)
      {
        case (Solver_choice::lobpcg):
          diffusion_hypre_lobpcg(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res);
          break;
        case (Solver_choice::power_iteration):
          spn_power_iteration(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res);
          break;
        case (Solver_choice::amgx_power_iteration):
          spn_amgx_power_iteration(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res);
          break;
        case (Solver_choice::slepc_vector):
          spn_slepc_vector(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res, scalar_flux);
          calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
          //temperature_heat_conduction(inp, mat_lib, power, temperature);
          break;
        case (Solver_choice::amgx_vector):
          spn_amgx_vector(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res);
          break;
        case (Solver_choice::slepc_manual):
          spn_slepc_manual(time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res);
          break;
        case (Solver_choice::epdo_power_iteration):
          epdo_power_iteration(quad.get(), time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res, scalar_flux);
          break;
        case (Solver_choice::sfent_power_iteration):
          sfent_power_iteration(quad.get(), time, inp, mat_lib, ess_bdr, pmesh.get(), &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_power_iteration):
          saaf_power_iteration(quad.get(), time, inp, mat_lib, pmesh.get(), &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_angle_power_iteration):
          saaf_angle_power_iteration(quad.get(), time, inp, mat_lib, res, scalar_flux, scalar_flux.keff, power, temperature);
          res.keff = scalar_flux.keff;
          break;
        case (Solver_choice::saaf_angle_jacobi_power_iteration):
          saaf_angle_jacobi_power_iteration(quad.get(), time, inp, mat_lib, res, scalar_flux);
          break;
        case (Solver_choice::saaf_angle_nda_slepc):
          saaf_angle_nda_slepc(quad.get(), time, inp, mat_lib, res, scalar_flux);
          break;
        case (Solver_choice::saaf_slepc_vector):
          saaf_slepc_vector(quad.get(), time, inp, mat_lib, pmesh.get(), &fespace, res, scalar_flux);
          break;
        case (Solver_choice::saaf_true_slepc_vector):
          saaf_true_slepc_vector(quad.get(), time, inp, mat_lib, res);
          calc_scalar_flux_slepc_vector(res, quad.get(), mat_lib, scalar_flux);
          calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
          //temperature_heat_conduction(inp, mat_lib, power, temperature);
          break;
        case (Solver_choice::saaf_true_slepc_diffusion_vector):
          saaf_true_slepc_diffusion_vector(quad.get(), time, inp, mat_lib, res, scalar_flux);
          break;
        case (Solver_choice::saaf_true_slepc_transport_vector):
          saaf_true_slepc_transport_vector(quad.get(), time, inp, mat_lib, res);
          calc_scalar_flux_slepc_vector(res, quad.get(), mat_lib, scalar_flux);
          break;
        case (Solver_choice::saaf_true_amgx_vector):
          saaf_true_amgx_vector(quad.get(), time, inp, mat_lib, res);
          calc_scalar_flux_slepc_vector(res, quad.get(), mat_lib, scalar_flux);
          calc_power(mat_lib, scalar_flux, inp.rxpwr, power);
          /* TODO finish implementation
          temperature_heat_conduction(inp, mat_lib, power, temperature);
          */
          break;
        case (Solver_choice::saaf_split_slepc_vector):
          saaf_split_slepc_vector(quad.get(), time, inp, mat_lib, res, scalar_flux);
          calc_scalar_flux_slepc_vector(res, quad.get(), mat_lib, scalar_flux);
          break;
        case (Solver_choice::saaf_split_power_iteration):
          saaf_split_power_iteration(quad.get(), time, inp, mat_lib, res, scalar_flux);
          calc_scalar_flux_slepc_vector(res, quad.get(), mat_lib, scalar_flux);
          break;
        case (Solver_choice::saaf_self_power_iteration):
          saaf_self_power_iteration(quad.get(), time, inp, mat_lib, res, scalar_flux);
          calc_scalar_flux_slepc_vector(res, quad.get(), mat_lib, scalar_flux);
          break;
        case (Solver_choice::dg_spn_power_iteration):
          dg_spn_power_iteration(inp, mat_lib, pmesh.get());
          break;
        case (Solver_choice::dg_transport_power_iteration):
          dg_transport_power_iteration(inp, mat_lib, quad.get(), pmesh.get());
          break;
        case (Solver_choice::spn_partial_assembly_power_iteration):
          spn_partial_assembly_power_iteration(inp, mat_lib, ess_bdr, res);
          break;
        case (Solver_choice::conduction_radial):
        case (Solver_choice::conduction_linear):
          temperature_heat_conduction(inp, mat_lib, power, temperature);
          break;
        case (Solver_choice::none):
          break; // do absolutely nothing
        default:
          std::cerr << "Invalid solver choice.\n";
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
      Analytic anal(inp.ref, inp.sol, *gf, xs, pmesh.get());
      anal.analyze(cout);
      ostringstream analysis_fname;
      analysis_fname << "out_" << setfill('0') << setw(6) << myid << ".csv";
      ofstream ofs{analysis_fname.str()};
      anal.print(ofs);
    }

    ostringstream state_name;
    state_name << "STATE" << setfill('0') << setw(3) << state;

    ParaViewDataCollection output(inp.input_file_stub() + "_" + state_name.str(), pmesh.get());

    output.RegisterField("temperature", &temperature);
    output.RegisterField("power", &power);

    const size_t group_width{std::to_string(xs.ngroup()).length()};

    if (is_transport)
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

    }
    else
    {

      if (inp.sol == Solver_choice::slepc_vector)
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
      }
      else
      {
        for (int g{0}; g < scalar_flux.ngroup(); ++g)
        {
          std::ostringstream scalar_flux_record;
          scalar_flux_record << "scalar_flux_g"
                             << std::setfill('0')
                             << std::setw(group_width)
                             << g;
          output.RegisterField(scalar_flux_record.str(), &res(g));
        }
      }

    }

    output.SetFormat(DataCollection::PARALLEL_FORMAT);
    output.Save();

    /* TODO SUM sum << state << ',' << inp.spn_order() << ',' << res.keff << '\n'; */

    time.stop();

  }

  if (mat_lib.ngroup() == 24)
  {
    // assume 24-group SCALE-based energy group structure
    // 1) E < 0.625 eV (g < 3)
    // 2) 0.625 eV < E < 6 keV (3 < g < 17)
    // 3) E > 6 keV (g > 17)
    coarse_energy_plot(pmesh.get(), scalar_flux, {3, 17},
      inp.input_file_stub()+"_coarse_energy");
  }

  mesh_stats(pmesh.get());

  //total_xs_plot(&pmesh, mat_lib, inp.input_file_stub()+"_total_xs");

  if (myid == 0) time.Print(cout);

  // 15. Free the used memory.

  if (inp.order > 0) delete fec;

  MPI_Finalize();

  return 0;
}
