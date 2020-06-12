#include "analytic.hpp"
#include "xslibrary.hpp"
#include "input.hpp"
#include "mfem.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

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
    if (((strcmp(argv[i], "-i") == 0)  || (strcmp(argv[i], "--input") == 0)) 
      && (argc >= (i+1)))
    {
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
    inp.read_state();
  }

  // parse command-line options using MFEM
  OptionsParser args{custom_arguments(inp, argc, argv)};

  // parse again and overwrite
  args.Parse();

  if (!args.Good())
  {
    if (myid == 0) args.PrintUsage(cout);
    MPI_Finalize();
    return 1;
  }

  const int mfem_print_level{inp.verbose ? 1 : 0};
  const int hypre_print_level{inp.verbose ? 1 : 0};

  if ((myid == 0) && mfem_print_level) args.PrintOptions(cout);

  // input is finalized, you can read cross sections
  xs.read_userxs(inp.xs_file);

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(inp.device);
  if (myid == 0) device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  auto mesh{new Mesh(inp.mesh_file, 1, 1)};
  int dim = mesh->Dimension();

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 50,000
  //    elements.
  for (int lvl{0}; lvl < inp.refine_serial; ++lvl) mesh->UniformRefinement();

  // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
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
  Array<int> ess_tdof_list;
  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
  // begin by setting all bcs to essential
  ess_bdr = 0;

  // boundary conditions are baked into the matrix
  // go through and set boundary conditions based on user input
  for (int i{0}; 
      (i < inp.boundary_conditions.Size()) && (i < ess_bdr.Size()); ++i)
    ess_bdr[i] = inp.boundary_conditions[i];

  // 8. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  ParGridFunction x(fespace);
  x = 0.0;

  // 9. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    domain integrator.

  // assemble leakage coefficient
  Vector leakage(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < leakage.Size(); ++i)
    leakage[i] = 1.0 / (3.0 * xs.rec[i].sigma_tr[0]);
  PWConstCoefficient leakage_func(leakage);

  // assemble removal coefficient
  Vector removal(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = xs.rec[i].sigma_r[0];
  PWConstCoefficient removal_func(removal);

  // assemble fission coefficient
  Vector fission(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i)
    fission[i] = xs.rec[i].nusf[0];
  PWConstCoefficient fission_func(fission);

  auto a{new ParBilinearForm(fespace)};
  auto m{new ParBilinearForm(fespace)};

  if (inp.partial_assembly)
  {
    a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }

  a->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a->AddDomainIntegrator(new MassIntegrator(removal_func));
  m->AddDomainIntegrator(new MassIntegrator(fission_func));

  // 10. Assemble the bilinear form and the corresponding linear system,
  //     applying any necessary transformations such as: eliminating boundary
  //     conditions, applying conforming constraints for non-conforming AMR,
  //     static condensation, etc.
  if (inp.static_condensation)
  {
    a->EnableStaticCondensation();
    m->EnableStaticCondensation();
  }

  a->Assemble();
  a->EliminateEssentialBCDiag(ess_bdr, 1.0);
  a->Finalize();

  m->Assemble();
  // shift the eigenvalue corresponding to eliminated dofs to a large value
  m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  m->Finalize();

  HypreParMatrix* A = a->ParallelAssemble();
  HypreParMatrix* M = m->ParallelAssemble();

  delete a;
  delete m;

  // 11. Setup the preconditioner.
  Solver* prec{nullptr};
  auto amg{new HypreBoomerAMG(*A)};
  amg->SetPrintLevel(hypre_print_level);
  prec = amg;

  auto lobpcg{new HypreLOBPCG(MPI_COMM_WORLD)};
  lobpcg->SetNumModes(1); // solve for nev eigenmodes
  // Random seed is set used to initialize the initial eigenvalues for the lobpcg
  // algorithm. This can significantly affect convergence time but convergence
  // time remains bounded.
  // No default provided by MFEM but this is used in one of the tutorials.
  lobpcg->SetRandomSeed(730);
  lobpcg->SetPreconditioner(*prec);
  lobpcg->SetMaxIter(inp.tol.max_outer);
  lobpcg->SetTol(inp.tol.k * inp.tol.phi);
  lobpcg->SetPrecondUsageMode(1);
  lobpcg->SetPrintLevel(hypre_print_level);
  lobpcg->SetMassMatrix(*M);
  lobpcg->SetOperator(*A);

  // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
  //    parallel grid function to represent each of the eigenmodes returned by
  //    the solver.
  lobpcg->Solve();

  // store solution
  Array<double> eigenvalues;
  lobpcg->GetEigenvalues(eigenvalues);
  const double keff{1.0 / eigenvalues[0]};
  x = lobpcg->GetEigenvector(0); // this is the local solution on this process

  // free memory from solver
  delete lobpcg;
  delete prec; // don't also delete amg or that would be double delete

  const auto old_precision{cout.precision()};
  cout << "keff = " << setprecision(20) << keff << setprecision(old_precision) << '\n';

  // TODO this is wrong for parallel!
  if ((myid == 0) && (inp.ref != Analytic::invalid))
  {
    Analytic anal(inp.ref, pmesh, x, keff);
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
    x.SaveVTK(ofs, "solution", 0);
    ofstream ofs_mesh{"out.mesh"};
    pmesh->Print(ofs_mesh);
  }

  // 15. Free the used memory.
  delete A;
  delete M;
  delete fespace;
  if (inp.order > 0) { delete fec; }
  delete pmesh;

  MPI_Finalize();


  return 0;
}
