#ifndef DELAUNAY
#define DELAUNAY

#include <vector>
#include <array>
#include <unordered_map>

class Point
{

public:

  Point() {}
  Point(double i, double j) : x{i}, y{j} {}
  Point(const std::array<double,2>& arr) : x{arr[0]}, y{arr[1]} {}
  Point(const std::vector<double>& vec) : x{vec[0]}, y{vec[1]} {}

  double x{0.0}, y{0.0};

  Point operator +(const Point& pt) const { return {x+pt.x, y+pt.y}; }
  Point operator -(const Point& pt) const { return {x-pt.x, y-pt.y}; }

  Point operator *(double factor) const { return {x*factor, y*factor}; }
  Point operator /(double factor) const { return {x/factor, y/factor}; }

  void operator +=(const Point& pt) { x+=pt.x; y+=pt.y; }
  void operator -=(const Point& pt) { x-=pt.x; y-=pt.y; }

  void operator *=(double factor) { x*=factor; y*=factor; }
  void operator /=(double factor) { x/=factor; y/=factor; }

};

class Edge
{

public:

  Edge(int i, int j) : edg{i,j} {}
  Edge(const Edge& e) : edg{e.edg} {}

  // An edge is the same regardless of the order of the indices. Therefore,
  // return "true" regardless of the order of the edges.
  bool operator==(const Edge& e) const
  { 
    return (((edg[0] == e[0]) && (edg[1] == e[1])) 
      || ((edg[0] == e[1]) && (edg[1] == e[0])));
  }
  
  // Do not allow a "non-const" option.
  // Must be initialized by constructor.
  int operator[](int i) const { return edg[i]; }

private:

  std::array<int,2> edg{0,0};

};

class Triangle
{

public:

  Triangle(int i, int j, int k) : tri{i,j,k} {}

  int operator[](int idx) const { return tri[idx]; }
  int& operator[](int idx) { return tri[idx]; }

  bool operator==(const Triangle& t) const
  {
    for (int i{0}; i < 3; ++i)
    {
      if (tri[i] != t[i]) return false;
    }
    return true;
  }

  bool operator!=(const Triangle& t) const
  {
    bool ans{false};
    for (int i{0}; i < 3; ++i)
    {
      ans = ans || (tri[i] != t[i]);
    }
    return ans;
  }

  Edge edge(int idx) const { return {tri[idx], tri[(idx+1)%3]}; }

private:

  std::array<int,3> tri{0,0,0};

};

class TriangularMesh
{
  
public:

  TriangularMesh() {}
  TriangularMesh(const std::vector<Point>& pt) : point_list{pt}, n_dir{static_cast<int>(pt.size())} {}
  TriangularMesh(const std::vector<Point>& pt, const std::vector<Triangle>& el)
    : point_list{pt}, element_list{el}, n_dir{static_cast<int>(pt.size())} {}

  void associate_point_list(const std::vector<Point>& pts)
  {
    point_list = pts;
    n_dir = static_cast<int>(pts.size());
  }
  void associate_element_list(const std::vector<Triangle>& el)
  { 
    element_list = el;
  }

  int nd() const { return n_dir; };

  const std::vector<Point>& get_point_list() const { return point_list; }
  std::vector<Point>& get_point_list() { return point_list; }
  const std::vector<Triangle>& get_element_list() const { return element_list; }
  std::vector<Triangle>& get_element_list() { return element_list; }
  const std::vector<int>& get_lookup_list() const { return lookup_list; }
  std::vector<int>& get_lookup_list() { return lookup_list; }

  int lookup_idx(int idx) const
  {
    return (idx >= n_dir) ? lookup_list[idx-n_dir] : idx;
  }

private:

  std::vector<Point> point_list;
  std::vector<Triangle> element_list;
  std::vector<int> lookup_list;
  int n_dir{0};

};

double dist(const Point& x0, const Point& x1);
Point calc_centroid(const std::vector<Point>& points);
double calc_max_distance(const Point& centroid, const std::vector<Point> points);
void append_super_triangle(std::vector<Point>& points, double fudge=1.1);
bool circumcircle_valid(const Point& a, const Point& b, const Point& c, const Point& d);

std::vector<Edge> identify_boundary(const TriangularMesh& msh);
void cleanup_elements(TriangularMesh& msh);
void cleanup_points(TriangularMesh& msh);

void delaunay(TriangularMesh& msh);
void add_reflected_points_half(TriangularMesh& msh, double tol=0.2); // tolerance set for S2
void add_reflected_points_full(TriangularMesh& msh, double tol=0.2); // tolerance set for S2

double triangle_area(const std::array<Point,3>& corn);
Triangle point_locator(const Point& point, const TriangularMesh& msh, double rel_tol=1e-8);

double psi(double xi, double eta, int idx);
std::unordered_multimap<int,double> interpolate_weight(const Point& point, const TriangularMesh& msh);

#endif
