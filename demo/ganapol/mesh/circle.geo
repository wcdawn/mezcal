// Run this with the following command line.
// gmsh circle.geo -save -format msh22

//h = 0.003125; // characteristic mesh size

refinement=5;
h = 0.1*0.5^refinement;

// critical radius for an infinite cylinder with c=1.5
// some sources include an extra "5" digit at the end but this is disputed
radius = 1.1783408;

Point(1) = {0.0, 0.0, 0.0, h};     // Center
Point(2) = {radius, 0.0, 0.0, h};  // Right
Point(3) = {0.0, radius, 0.0, h};  // Top
Point(4) = {-radius, 0.0, 0.0, h}; // Left
Point(5) = {0.0, -radius, 0.0, h}; // Bottom

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Curve Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};
Physical Surface(1) = {1}; // Bottom

// use quads
Mesh.RecombineAll = 1;
//Mesh.Algorithm = 8;

////Mesh 2;

// use quads
//// RecombineMesh;
//// Mesh.SubdivisionAlgorithm = 1;
//// //RefineMesh;
