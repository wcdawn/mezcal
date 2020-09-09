#include "analytic.hpp"
#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "solver.hpp"
#include "timer.hpp"
#include "xslibrary.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace mfem;

#include <cmath>
using std::abs;
using std::sin;

void version(std::ostream& os); // forward declaration

int main(int argc, char* argv[])
{
  /*
   * Features to add:
   * - SPN.
   * - SLEPc & Trilinos solvers.
   * - Loop over states.
   */

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

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(inp.device);
  if (myid == 0) device.Print();

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

  // TODO this needs to allow for differing values of spn_eq with different states
  const int vdim{(inp.sol == Solver_choice::slepc_vector) ? mat_lib.ngroup()*inp.spn_eq() : 1};
  ParFiniteElementSpace fespace{&pmesh, fec, vdim, Ordering::byVDIM};
  const int dof{fespace.TrueVSize() * num_procs * ((inp.sol == Solver_choice::slepc_vector) ? 1 : mat_lib.ngroup())};
  if (myid == 0)
    cout << "Number of finite element unknowns: " << dof << endl;

  // TODO this should move outside of the [STATE] loop but we need to disallow
  // changing the solver type between states (maybe move to [CORE] block)
  const int flux_length{(inp.sol == Solver_choice::slepc_vector) ? 1 : xs.ngroup()};
  Result res{&fespace, flux_length, inp.spn_eq()};

  ofstream sum{inp.input_file_stub() + ".sum"};
  sum << "=== STATEPOINT SUMMARY ===\n" << "state, spn_order, keff\n";

  for (int state{0}; state < inp.nstate(); ++state)
  {
    inp.read_state();
    res.resize(mat_lib.ngroup(), inp.spn_eq());

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
          diffusion_power_iteration(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::slepc_vector):
          spn_slepc_vector(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        case (Solver_choice::slepc_manual):
          diffusion_slepc_manual(time, inp, mat_lib, ess_bdr, &pmesh, &fespace, res);
          break;
        default:
          cerr << "Invalid solver choice.\n";
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
      Analytic anal(inp.ref, inp.sol, res, xs, &pmesh);
      anal.analyze(cout);
      ostringstream analysis_fname;
      analysis_fname << "out_" << setfill('0') << setw(6) << myid << ".csv";
      ofstream ofs{analysis_fname.str()};
      anal.print(ofs);
    }

    ostringstream state_name;
    state_name << "STATE" << setfill('0') << setw(3) << state;

    ParaViewDataCollection output(inp.input_file_stub() + "_" + state_name.str(), &pmesh);

    const auto group_width{std::to_string(xs.ngroup()).length()};
    const auto moment_width{std::to_string(2*inp.spn_eq()).length()};

    if (inp.sol == Solver_choice::slepc_vector)
    {
      // TODO fix this so that this can be more descriptive
      // Will probably require constructing a new FiniteElementSpace, a new
      // Vector, and a new GridFunction
      output.RegisterField("phi", &res(0));
    }
    else
    {
      for (int n{0}; n < inp.spn_eq(); ++n)
      {
        for (int g{0}; g < xs.ngroup(); ++g)
        {
          ostringstream record_name;
          record_name << "phi_g" << setfill('0') << setw(group_width) << g
                      << "_n" << setfill('0') << setw(moment_width) << 2*n;
          output.RegisterField(record_name.str(), &res(g,n));
        }
      }
    }

    output.SetFormat(DataCollection::PARALLEL_FORMAT);
    output.Save();

    sum << state << ',' << inp.spn_order() << ',' << res.keff << '\n';

    time.stop();


  }

  if (myid == 0) time.Print(cout);

  // 15. Free the used memory.

  if (inp.order > 0) delete fec;

  MPI_Finalize();

  return 0;
}
