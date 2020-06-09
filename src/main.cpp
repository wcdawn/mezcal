#include "analytic.hpp"
#include "mfem.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace mfem;

////////////////////////////////////////////////////////////////////////////////
// SOLVER OPTIONS
////////////////////////////////////////////////////////////////////////////////

// < 0 prints no information
// 0 prints default based on what's "best",
// 1 prints every inner iteration & average reduction factor,
// 2 prints the total number of iterations & average reduction factor
// 3 prints first, last, and average reduction factor,
// >= 4 prints only average reduction factor
constexpr int print_level{0}; // default 0

// <= 0 prints nothing
// 1 prints a lot
// idk about 2
// 3 prints a lot and more
constexpr int hypre_print_level{1};

// maximum number of inner iterations
constexpr int max_inner{400}; // default 1000

// actual tolerance of r0 in the solver is max(nom*rel_tol*rel_tol,
// abs_tol*abs_tol) if abs_tol=0, abs_tol is ignored

// relative tolerance of inner Krylov solver
constexpr double inner_rel_tol{1e-12}; // default 1e-12

// absolute tolerance of inner Krylov solver
constexpr double inner_abs_tol{1e-24}; // default 1e-24 // examples use 0.0 too

int main(int argc, char* argv[])
{

  // "input" from a file

  // mesh
  string fname_mesh;

  // boundary conditions
  vector<int> boundary_conditions; // 0 is essential, 1 is non-essential

  // reference problem
  constexpr Analytic::Reference reference{Analytic::tworeg};

  // cross sections
  vector<double> sigma_tr;
  vector<double> sigma_r;
  vector<double> nusf;
  vector<double> q_fixed;

  // actually fill in data
  switch (reference)
  {
    case (Analytic::fixed_src):
      fname_mesh = "../analytic/fixed_src/fixed_src_quad.msh";
      boundary_conditions = {0, 1, 0, 1};
      sigma_tr.push_back(1.0 / 3.0);
      sigma_r.push_back(1.0);
      q_fixed.push_back(1.0);
      break;
    case (Analytic::oned_oneg):
      fname_mesh = "../analytic/1d1g/1d1g_tri.msh";
      boundary_conditions = {0, 1, 0, 1};
      sigma_tr.push_back(3.619027e-2);
      sigma_r.push_back(1.532090e-1);
      nusf.push_back(1.566769e-1);
      break;
    case (Analytic::twod_oneg):
      fname_mesh = "../analytic/2d1g/2d1g.msh";
      boundary_conditions = {1, 1, 1, 1};
      sigma_tr.push_back(3.619027e-2);
      sigma_r.push_back(1.532090e-1);
      nusf.push_back(1.566769e-1);
      break;
    case (Analytic::square):
      fname_mesh = "../analytic/square/square.mesh";
      boundary_conditions = {1, 1, 1, 1};
      sigma_tr.push_back(3.619027e-2);
      sigma_r.push_back(1.532090e-1);
      nusf.push_back(1.566769e-1);
      break;
    case (Analytic::finite_cyl):
      fname_mesh = "../analytic/finite_cyl/finite_cyl_tet.msh";
      boundary_conditions = {1, 1, 1, 1};
      sigma_tr.push_back(1.0 / 3.0);
      sigma_r.push_back(1.0);
      nusf.push_back(1.0);
      break;
    case (Analytic::tworeg):
      fname_mesh = "../analytic/tworeg/tworeg_80.msh";
      // MAT1
      sigma_tr.push_back(1.0 / (3.0 * 1.2));
      sigma_r.push_back(0.02);
      nusf.push_back(0.02);
      // MAT2
      sigma_tr.push_back(1.0 / (3.0 * 0.7));
      sigma_r.push_back(0.015);
      nusf.push_back(0.0);
  }

  // Initialize MPI
  int num_procs;
  int myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 1. Parse command-line options.
  // TODO add verbose option
  const char* mesh_file     = fname_mesh.c_str();
  int order                 = 1;
  bool static_cond          = false;
  bool pa                   = false;
  const char* device_config = "cpu";
  bool visualization        = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                 "--no-static-condensation", "Enable static condensation.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                 "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");

  args.Parse();
  if (!args.Good())
  {
    if (myid == 0)
      args.PrintUsage(cout);
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) args.PrintOptions(cout);

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  if (myid == 0) device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  auto mesh{new Mesh(mesh_file, 1, 1)};
  int dim = mesh->Dimension();

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 50,000
  //    elements.
  {
    // const int ref_levels{
    //   static_cast<int>(floor(log(5e4 / mesh->GetNE()) / log(2.) / dim))};
    constexpr int ref_levels{1};
    for (int lvl{0}; lvl < ref_levels; ++lvl) mesh->UniformRefinement();
  }

  // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  auto pmesh{new ParMesh(MPI_COMM_WORLD, *mesh)};
  delete mesh;
  {
    constexpr int par_ref_levels{0};
    for (int lvl{0}; lvl < par_ref_levels; ++lvl) pmesh->UniformRefinement();
  }

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection* fec;
  if (order > 0)
  {
    fec = new H1_FECollection(order, dim);
  }
  else if (pmesh->GetNodes())
  {
    fec = pmesh->GetNodes()->OwnFEC();
    if (myid == 0)
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
  }
  else
  {
    fec = new H1_FECollection(order = 1, dim);
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
  cout << "SIZE: " << pmesh->bdr_attributes.Size() << endl;
  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
  // begin by setting all bcs to essential
  ess_bdr = 0;

  // boundary conditions are baked into the matrix
  // go through and set boundary conditions based on user input
  for (int i{0}; (i < boundary_conditions.size()) && (i < ess_bdr.Size()); ++i)
    ess_bdr[i] = boundary_conditions[i];

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
    leakage[i] = 1.0 / (3.0 * sigma_tr[i]);
  PWConstCoefficient leakage_func(leakage);

  // assemble removal coefficient
  Vector removal(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i)
    removal[i] = sigma_r[i];
  PWConstCoefficient removal_func(removal);

  // assemble fission coefficient
  Vector fission(pmesh->attributes.Max()); // length is number of materials
  for (int i{0}; i < removal.Size(); ++i)
    fission[i] = nusf[i];
  PWConstCoefficient fission_func(fission);

  auto a{new ParBilinearForm(fespace)};
  auto m{new ParBilinearForm(fespace)};

  if (pa)
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
  if (static_cond)
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
  lobpcg->SetNumModes(1); // solve for nev eigenmodes // size will solve for *all* eigenmodes
  // Random seed is set used to initialize the initial eigenvalues for the lopcg
  // algorithm. This can significantly affect convergence time but convergence
  // time remains bounded.
  // No default provided by MFEM but this is used in one of the tutorials.
  lobpcg->SetRandomSeed(1);
  lobpcg->SetPreconditioner(*prec);
  lobpcg->SetMaxIter(max_inner);
  lobpcg->SetTol(inner_abs_tol);
  lobpcg->SetPrecondUsageMode(1);
  lobpcg->SetPrintLevel(hypre_print_level);
  lobpcg->SetMassMatrix(*M);
  lobpcg->SetOperator(*A);

  // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
  //    parallel grid function to represent each of the eigenmodes returned by
  //    the solver.
  Array<double> eigenvalues;
  lobpcg->Solve();
  lobpcg->GetEigenvalues(eigenvalues);
  x = lobpcg->GetEigenvector(0); // this is the local solution on this process

  // TODO this is wrong for parallel!
  if (myid == 0)
  {
    const double keff{1.0 / eigenvalues[0]};
    Analytic anal(reference, pmesh, x, keff);
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
  }


  // 14. Send the solution by socket to a GLVis server.
  if (visualization)
  {
    constexpr char vishost[] = "localhost";
    constexpr int visport    = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock.precision(8);
    sol_sock << "solution\n" << *pmesh << x << flush;
  }

  // 15. Free the used memory.
  delete A;
  delete M;
  delete fespace;
  if (order > 0) { delete fec; }
  delete pmesh;

  MPI_Finalize();

  return 0;
}
