#include "analytic.hpp"
#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "solver.hpp"
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
double oned_oneg_c_fun(const Vector& v);

int main(int argc, char* argv[])
{
  /*
   * Features to add:
   * - Timing for solver and other routines.
   * - SPN.
   * - SLEPc & Trilinos solvers.
   */

  // "input" from a file

  // mesh
  string fname_xs;
  string fname_inp;

  // cross sections
  XSLibrary xs;

  // reference problem
  // Initialize MPI
  int num_procs;
  int myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 1. Parse command-line options.
  // Do our own parsing to get the input file name.
  string input_file;
  for (int i{1}; i < argc; ++i)
  {
    if (((strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "--input") == 0))
        && (argc >= (i + 1)))
    {
      input_file = argv[i + 1];
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

  // parse again and overwrite
  args.Parse();

  if (!args.Good())
  {
    if (myid == 0)
      args.PrintUsage(cout);
    MPI_Finalize();
    return 1;
  }

  // need to reconcile enums to match their const char* counterparts
  inp.reconcile_enums();

  if ((myid == 0) && (inp.verbose))
    args.PrintOptions(cout);

  // core block is finished, you can read cross sections from file
  // xs.read_userxs(inp.xs_file);
  switch (inp.xs_format)
  {
    case (XSLibrary::userxs):
      xs.read_userxs(inp.xs_file);
      break;
    case (XSLibrary::hdf5xs):
      xs.read_hdf5xs(inp.xs_file, inp.hdf5xs_lib_name);
      break;
    default:
      cerr << "Invalid xs format" << endl;
      return 1;
  }

  // core block is finished, you can read the material block
  Material_library mat_lib{xs};
  mat_lib.read_material(inp.ifs);

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(inp.device);
  if (myid == 0)
    device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  auto mesh{new Mesh(inp.mesh_file, 1, 1)};
  int dim = mesh->Dimension();

  // refine the mesh in serial
  for (int lvl{0}; lvl < inp.refine_serial; ++lvl) mesh->UniformRefinement();

  // create a parallel mesh and refine as necessary
  auto pmesh{new ParMesh(MPI_COMM_WORLD, *mesh)};
  delete mesh;
  for (int lvl{0}; lvl < inp.refine_parallel; ++lvl) pmesh->UniformRefinement();

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection* fec;
  if (inp.order > 0)
  {
    fec = new H1_FECollection(inp.order, dim);
  }
  else if (pmesh->GetNodes())
  {
    fec = pmesh->GetNodes()->OwnFEC();
    if (myid == 0)
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
  }
  else
  {
    fec = new H1_FECollection(inp.order = 1, dim);
  }
  auto fespace = new ParFiniteElementSpace(pmesh, fec);
  const HYPRE_Int size{fespace->GlobalTrueVSize()};
  if (myid == 0)
    cout << "Number of finite element unknowns: " << size * xs.ngroup() << endl;

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

  // do the solving
  Result res{fespace, xs.ngroup()};
  // TODO loop over states
  inp.read_state();

  if (inp.ref == Analytic::fixed_src)
  {
    res = diffusion_fixed_src(inp, mat_lib, ess_bdr, pmesh, fespace);
  }
  else
  {
    switch (inp.sol)
    {
      case (Input::lobpcg):
        res = diffusion_lobpcg(inp, mat_lib, ess_bdr, pmesh, fespace);
        break;
      case (Input::power_iteration):
        res = diffusion_power_iteration(inp, mat_lib, ess_bdr, pmesh, fespace);
        break;
      default:
        cerr << "Invalid solver choice.\n";
        MPI_Finalize();
        return 1;
    }
  }

  const auto old_precision{cout.precision()};
  if ((myid == 0) && (inp.ref != Analytic::fixed_src))
    cout << "keff = " << setprecision(20) << res.keff
         << setprecision(old_precision) << '\n';

  if (inp.ref != Analytic::invalid)
  {
    Analytic anal(inp.ref, res, xs, pmesh);
    anal.analyze(cout);
    ostringstream analysis_fname;
    analysis_fname << "out_" << setfill('0') << setw(6) << myid << ".csv";
    ofstream ofs{analysis_fname.str()};
    anal.print(ofs);
  }

  ParaViewDataCollection output("output_dir", pmesh);
  const auto group_width{std::to_string(xs.ngroup()).length()};
  for (int g{0}; g < xs.ngroup(); ++g)
  {
    ostringstream record_name;
    record_name << "phi_g" << setfill('0') << setw(group_width) << g;
    output.RegisterField(record_name.str(), &res.phi[g]);
  }
  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  // 15. Free the used memory.

  delete fespace;
  if (inp.order > 0)
  {
    delete fec;
  }
  delete pmesh;

  MPI_Finalize();

  return 0;
}
