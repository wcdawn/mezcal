#include "mfem.hpp"

#include "analytic.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

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
constexpr int print_level{2}; // default 0

// <= 0 prints nothing
// 1 prints a lot
// idk about 2
// 3 prints a lot and more
constexpr int hypre_print_level{0};

// maximum number of inner iterations
constexpr int max_inner{400};  // default 1000

// actual tolerance of r0 in the solver is max(nom*rel_tol*rel_tol, abs_tol*abs_tol)
// if abs_tol=0, abs_tol is ignored

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
  constexpr Analytic::Reference reference{Analytic::fixed_src};

  // cross sections
  double sigma_tr{0.0};
  double sigma_r{0.0};
  double nusf{0.0};
  double q_fixed{0.0};

  // actually fill in data
  switch (reference)
  {
    case (Analytic::fixed_src):
      fname_mesh = "../analytic/fixed_src/fixed_src_quad.msh";
      boundary_conditions = {0, 1, 0, 1};
      sigma_tr = 1.0 / 3.0;
      sigma_r  = 1.0;
      q_fixed  = 1.0;
      break;
    case (Analytic::oned_oneg):
      fname_mesh = "../analytic/1d1g/1d1g_tri.msh";
      boundary_conditions = {0, 1, 0, 1};
      sigma_tr = 3.619027e-2;
      sigma_r  = 1.532090e-1;
      nusf     = 1.566769e-1;
      break;
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
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good())
  {
    if (myid == 0) args.PrintUsage(cout);
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
    int ref_levels = 
      static_cast<int>(floor(log(5e4 / mesh->GetNE()) / log(2.) / dim));
    for (int l{0}; l < ref_levels; ++l) { mesh->UniformRefinement(); }
  }

  // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  auto pmesh{new ParMesh(MPI_COMM_WORLD, *mesh)};
  delete mesh;
  {
    int par_ref_levels = 2;
    for (int l = 0; l < par_ref_levels; ++l) { pmesh->UniformRefinement(); }
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
    if (myid == 0) cout << "Using isoparametric FEs: " << fec->Name() << endl;
  }
  else
  {
    fec = new H1_FECollection(order = 1, dim);
  }
  auto fespace = new ParFiniteElementSpace(pmesh, fec);
  HYPRE_Int size{fespace->GlobalTrueVSize()};
  if (myid == 0) cout << "Number of finite element unknowns: " << size << endl;

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
  // begin by setting all bcs to essential
  ess_bdr = 0;
  // go through and set boundary conditions based on user input
  for (int i{0}; (i < boundary_conditions.size()) && (i < ess_bdr.Size()); ++i)
    ess_bdr[i] = boundary_conditions[i];
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // 7. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  auto b{new ParLinearForm(fespace)};
  ConstantCoefficient source{q_fixed};
  b->AddDomainIntegrator(new DomainLFIntegrator(source));
  b->Assemble();

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
  leakage = 1.0 / (3.0 * sigma_tr);
  PWConstCoefficient leakage_func(leakage);

  // assemble removal coefficient
  Vector removal(pmesh->attributes.Max()); // length is number of materials
  removal = sigma_r;
  PWConstCoefficient removal_func(removal);

  auto a{new ParBilinearForm(fespace)};
  if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
  a->AddDomainIntegrator(new DiffusionIntegrator(leakage_func));
  a->AddDomainIntegrator(new MassIntegrator(removal_func));

  // 10. Assemble the bilinear form and the corresponding linear system,
  //     applying any necessary transformations such as: eliminating boundary
  //     conditions, applying conforming constraints for non-conforming AMR,
  //     static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  OperatorPtr A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

  // 11. Solve the linear system A X = B.
  Solver* prec{nullptr};
   if (pa)
   {
      if (UsesTensorBasis(*fespace))
      {
         prec = new OperatorJacobiSmoother(*a, ess_tdof_list);
      }
   }
   else
   {
     auto prec_move{new HypreBoomerAMG};
     prec_move->SetPrintLevel(0);
     prec = prec_move;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(inner_rel_tol);
   cg.SetAbsTol(inner_abs_tol);
   cg.SetMaxIter(max_inner);
   cg.SetPrintLevel(print_level);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

  // 12. Recover the solution as a finite element grid function. This is the
  //     local finite element solution on each processor.
  a->RecoverFEMSolution(X, *b, x);

  // 13. Save the refined mesh and the solution. This output can be viewed later
  //     using GLVis: "glvis -np <np> -m refined.mesh -g sol.gf".
  {
    ostringstream mesh_name;
    ostringstream sol_name;
    mesh_name << "mesh." << setfill('0') << setw(6) << myid;
    sol_name << "sol." << setfill('0') << setw(6) << myid;

    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(8);
    pmesh->Print(mesh_ofs);

    ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(8);
    x.Save(sol_ofs);
  }

  // TODO this is wrong for parallel!
  if (myid == 0)
  {
    constexpr double keff{1.0};
    Analytic anal(reference, pmesh, x, keff);
    anal.analyze(cout);
    ofstream ofs{"out.csv"};
    anal.print(ofs);
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
  delete a;
  delete b;
  delete fespace;
  if (order > 0) { delete fec; }
  delete pmesh;

  MPI_Finalize();

  return 0;
}
