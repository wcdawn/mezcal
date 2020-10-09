#include "delaunay.hpp"

#include "mfem.hpp"

#include <list>
#include <cmath>
#include <cstddef>

double dist(const Point& x0, const Point& x1)
{
  return std::sqrt(std::pow(x0.x - x1.x, 2) + std::pow(x0.y - x1.y, 2));
}

Point calc_centroid(const std::vector<Point>& points)
{
  Point centroid{0.0, 0.0};
  for (const Point& p : points)
    centroid += p;
  centroid /= points.size();
  return centroid;
}

double calc_max_distance(const Point& centroid, const std::vector<Point> points)
{
  double max_dist{0.0};
  for (const Point& p : points)
    max_dist = std::max(max_dist, dist(centroid, p));
  return max_dist;
}

void append_super_triangle(std::vector<Point>& points, double fudge)
{
  const Point centroid{calc_centroid(points)};
  const double max_dist{calc_max_distance(centroid, points) * fudge};

  const double h2{2.0*max_dist};
  const double a2{4.0*max_dist/std::sqrt(3.0)};
  points.emplace_back(centroid + Point{0.0,h2});
  points.emplace_back(centroid + Point{-a2,-h2});
  points.emplace_back(centroid + Point{a2,-h2});
}

bool circumcircle_valid(const Point& a, const Point& b, const Point& c, const Point& d)
{
  mfem::DenseMatrix mat(4);

  mat(0,0) = a.x;
  mat(0,1) = a.y;
  mat(0,2) = a.x*a.x + a.y*a.y;
  mat(0,3) = 1.0;

  mat(1,0) = b.x;
  mat(1,1) = b.y;
  mat(1,2) = b.x*b.x + b.y*b.y;
  mat(1,3) = 1.0;

  mat(2,0) = c.x;
  mat(2,1) = c.y;
  mat(2,2) = c.x*c.x + c.y*c.y;
  mat(2,3) = 1.0;

  mat(3,0) = d.x;
  mat(3,1) = d.y;
  mat(3,2) = d.x*d.x + d.y*d.y;
  mat(3,3) = 1.0;

  return !(mat.Det() > -std::sqrt(std::numeric_limits<double>::epsilon()));
}

void delaunay(TriangularMesh& msh)
{
  std::vector<Point> points{msh.get_point_list()}; // note the copy
  const int N{static_cast<int>(points.size())};
  append_super_triangle(points);

  std::vector<Triangle>& element{msh.get_element_list()};

  element.emplace_back(N+0, N+1, N+2);

  for (int i{0}; i < N; ++i)
  {

    std::list<Triangle> bad_triangles;
    for (const Triangle& tri : element)
    {
      if (!circumcircle_valid(points[tri[0]], points[tri[1]], points[tri[2]], points[i]))
        bad_triangles.emplace_back(tri);
    }

    std::list<Edge> polygon;
    for (const Triangle& bad_tri : bad_triangles)
    {
      for (int j{0}; j < 3; ++j)
      {
        const Edge edg{bad_tri.edge(j)};
        bool found{false};
        for (const Triangle& compare_tri : bad_triangles)
        {
          if (bad_tri != compare_tri)
          {
            for (int k{0}; k < 3; ++k)
            {
              const Edge compare_edg{compare_tri.edge(k)};
              if (compare_edg == edg)
              {
                found = true;
                break;
              }
            }
          }
          if (found)
            break;
        }
        if (!found)
          polygon.push_back(edg);
      }
    }

    for (const Triangle& bad_tri : bad_triangles)
    {
      for (std::size_t idx{0}; idx < element.size();)
      {
        if (bad_tri == element[idx])
        {
          element.erase(element.begin()+idx);
        }
        else
        {
          ++idx;
        }
      }
    }

    for (const Edge& edg : polygon)
    {
      const Triangle new_tri{edg[0], edg[1], static_cast<int>(i)};
      element.push_back(new_tri);
    }

  }

  for (std::size_t idx{0}; idx < element.size();)
  {
    const Triangle& tri{element[idx]};
    if ((tri[0] >= N) || (tri[1] >= N) || (tri[2] >= N))
    {
      element.erase(element.begin()+idx);
    }
    else
    {
      ++idx;
    }
  }

  // delete the super triangle as well
  // this is done automatically because the operations here are done on a copy
  // of the original array
}

std::vector<Edge> identify_boundary(const TriangularMesh& msh)
{
  const std::vector<Triangle>& element{msh.get_element_list()};
  std::vector<Edge> boundary;

  for (size_t i{0}; i < element.size(); ++i)
  {
    for (int j{0}; j < 3; ++j)
    {
      const Edge this_edge{element[i].edge(j)};
      auto it{std::find(boundary.begin(), boundary.end(), this_edge)};
      if (it != boundary.end())
      {
        boundary.erase(it);
      }
      else
      {
        // record was *not* found
        boundary.push_back(this_edge);
      }
    }
  }
  return boundary;
}

void add_reflected_points_half(TriangularMesh& msh, double tol)
{
  // this uses rotational symmetry for the phi direction and reflective
  // symmetry for the theta direction

  std::vector<Point>& pts{msh.get_point_list()};
  std::vector<int>& lookup{msh.get_lookup_list()};

  const std::vector<Edge> edge_list{identify_boundary(msh)};

  std::set<int> boundary_pts;
  for (const Edge& edg : edge_list)
  {
    boundary_pts.insert(edg[0]);
    boundary_pts.insert(edg[1]);
  }

  constexpr double max_lft{2.0*M_PI};
  constexpr double max_rgt{0.0};
  constexpr double max_up{0.5*M_PI};
  constexpr double max_dn{0.0};

  for (int idx : boundary_pts)
  {
    const double dist_lft{max_lft - pts[idx].x};
    const double dist_rgt{pts[idx].x - max_rgt};
    const double dist_up{max_up - pts[idx].y};
    const double dist_dn{pts[idx].y - max_dn};

    if ((dist_lft - 0.25*M_PI) < tol)
    {
      pts.emplace_back(max_rgt-dist_lft, pts[idx].y);
      lookup.push_back(idx);
    }

    if ((dist_rgt - 0.25*M_PI) < tol)
    {
      pts.emplace_back(max_lft+dist_rgt, pts[idx].y);
      lookup.push_back(idx);
    }

    if ((dist_up - 0.25*M_PI) < tol)
    {
      pts.emplace_back(pts[idx].x, max_up+dist_up);
      lookup.push_back(idx);
    }

    if ((dist_dn - 0.25*M_PI) < tol)
    {
      pts.emplace_back(pts[idx].x, max_dn-dist_dn);
      lookup.push_back(idx);
    }

  }

}

void add_reflected_points_full(TriangularMesh& msh, double tol)
{
  // this uses rotational symmetry instead of reflective symmetry for all
  // boundaries

  std::vector<Point>& pts{msh.get_point_list()};
  std::vector<int>& lookup{msh.get_lookup_list()};

  const std::vector<Edge> edge_list{identify_boundary(msh)};

  std::set<int> boundary_pts;
  for (const Edge& edg : edge_list)
  {
    boundary_pts.insert(edg[0]);
    boundary_pts.insert(edg[1]);
  }

  constexpr double max_lft{2.0*M_PI};
  constexpr double max_rgt{0.0};
  constexpr double max_up{0.5*M_PI};
  constexpr double max_dn{-0.5*M_PI};

  for (int idx : boundary_pts)
  {
    const double dist_lft{max_lft - pts[idx].x};
    const double dist_rgt{pts[idx].x - max_rgt};
    const double dist_up{max_up - pts[idx].y};
    const double dist_dn{pts[idx].y - max_dn};

    if ((dist_lft - 0.25*M_PI) < tol)
    {
      pts.emplace_back(max_rgt-dist_lft, pts[idx].y);
      lookup.push_back(idx);
    }

    if ((dist_rgt - 0.25*M_PI) < tol)
    {
      pts.emplace_back(max_lft+dist_rgt, pts[idx].y);
      lookup.push_back(idx);
    }

    if ((dist_up - 0.25*M_PI) < tol)
    {
      pts.emplace_back(pts[idx].x, max_dn-dist_up);
      lookup.push_back(idx);
    }

    if ((dist_dn - 0.25*M_PI) < tol)
    {
      pts.emplace_back(pts[idx].x, max_up+dist_dn);
      lookup.push_back(idx);
    }

  }

}

void cleanup_elements(TriangularMesh& msh)
{
  std::vector<Triangle>& element{msh.get_element_list()};
  const int n_dir{msh.nd()};
  std::size_t idx{0};
  while (idx < element.size())
  {
    if ((element[idx][0] >= n_dir) && (element[idx][1] >= n_dir) && (element[idx][2] >= n_dir))
      element.erase(element.begin()+idx);
    else
      ++idx;
  }
}

void cleanup_points(TriangularMesh& msh)
{
  std::vector<Point>& pts{msh.get_point_list()};
  std::vector<Triangle>& element{msh.get_element_list()};
  std::vector<int>& lookup{msh.get_lookup_list()};

  int idx{msh.nd()};
  while (idx < static_cast<int>(pts.size()))
  {
    bool found{false};
    for (const Triangle& tri : element)
    {
      found = (tri[0] == idx) || (tri[1] == idx) || (tri[2] == idx);
      if (found)
        break;
    }
    if (!found)
    {
      pts.erase(pts.begin()+idx);
      lookup.erase(lookup.begin()+idx-msh.nd());
      for (Triangle& tri : element)
      {
        for (int i{0}; i < 3; ++i)
          if (tri[i] > idx)
            tri[i] -= 1;
      }
    }
    else
    {
      ++idx;
    }
  }
}
