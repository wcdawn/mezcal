#include "analytic.hpp"
#include "input.hpp"
#include "mfem.hpp"
#include "solver.hpp"
#include "xslibrary.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace mfem;

int main(int argc, char* argv[])
{
  // "input" from a file

  // mesh
  string fname_xs;
  string fname_inp;

  // cross sections
  XSLibrary xs;
  vector<double> q_fixed;

  // boundary conditions
  vector<int> boundary_conditions; // 0 is essential, 1 is non-essential

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
    inp.read_state();
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

  const int mfem_print_level{inp.verbose ? 1 : 0};
  // const int hypre_print_level{inp.verbose ? 1 : 0};

  if ((myid == 0) && mfem_print_level)
    args.PrintOptions(cout);

  // input is finalized, you can read cross sections
  xs.read_userxs(inp.xs_file);

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
    cout << "Number of finite element unknowns: " << size << endl;

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
  // begin by setting all bcs to essential
  ess_bdr = 0;

  // boundary conditions are baked into the matrix
  // go through and set boundary conditions based on user input
  for (int i{0}; (i < inp.boundary_conditions.Size()) && (i < ess_bdr.Size());
       ++i)
    ess_bdr[i] = inp.boundary_conditions[i];

  // do the solving
  Result res{fespace};
  if (inp.ref == Analytic::fixed_src)
  {
    res = diffusion_fixed_src(inp, xs, ess_bdr, pmesh, fespace);
  }
  else
  {
    res = diffusion_lobpcg(inp, xs, ess_bdr, pmesh, fespace);
  }

  const auto old_precision{cout.precision()};
  if (myid == 0)
    cout << "keff = " << setprecision(20) << res.keff
         << setprecision(old_precision) << '\n';

  // TODO this is wrong for parallel!
  if ((myid == 0) && (inp.ref != Analytic::invalid))
  {
    Analytic anal(inp.ref, pmesh, res.phi, res.keff);
    anal.analyze(cout);
    ofstream ofs{"out.csv"};
    anal.print(ofs);
  }

  // TODO this is wrong for parallel!
  // write vtk
  if (myid == 0)
  {
    ofstream ofs{"out.vtk"};
    pmesh->PrintVTK(ofs, 0);
    res.phi.SaveVTK(ofs, "solution", 0);
    ofstream ofs_mesh{"out.mesh"};
    pmesh->Print(ofs_mesh);
  }

  // 15. Free the used memory.

  /*
  delete A;
  delete M;
  */

  delete fespace;
  if (inp.order > 0)
  {
    delete fec;
  }
  delete pmesh;

  MPI_Finalize();

  return 0;
}
