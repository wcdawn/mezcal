#include "xslibrary.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>

using namespace mfem;

using std::cerr;
using std::cout;
using std::endl;

using std::ostream;
using std::ifstream;
using std::string;
using std::vector;

using std::max_element;
using std::accumulate;

void XSRecord::set_ngroup(int ngroup)
{
  G = ngroup;
  sigma_tr.resize(G);
  sigma_t.resize(G);
  sigma_r.resize(G);
  nusf.resize(G);
  chi.resize(G);
  source.resize(G);
  is_upscatter.resize(G);
}

void XSRecord::calculate_removal_xs()
{
  for (int g{0}; g < G; ++g)
    sigma_r[g] = sigma_t[g] - sigma_s.Elem(g, g);
}

void XSRecord::self_scatter_rebalance()
{
  for (int g{0}; g < G; ++g)
  {
    if (sigma_s.Elem(g,g) == 0.0)
    {
      sigma_s.Set(g, g, sigma_tr[g] - sigma_t[g]);
      sigma_t[g] = sigma_tr[g];
    }
  }
}

void XSRecord::evaluate_upscatter()
{
  for (int g{0}; g < G; ++g)
  {
    const int row_size{sigma_s.RowSize(g)};
    const int* row_columns{sigma_s.GetRowColumns(g)};
    const auto max_upscatter_group{
      std::max_element(row_columns, row_columns + row_size)};
    if (*max_upscatter_group > g) 
      is_upscatter[g] = true;
    else
      is_upscatter[g] = false;
  }
}

void XSLibrary::read_userxs(const char* fname)
{
  const string fname_str{fname};
  read_userxs(fname_str);
}

void XSLibrary::read_userxs(const std::string& fname)
{
  ifstream ifs{fname};
  if (!ifs)
    throw Invalid_file_name{fname};
  string card;
  int mat_count{-1};
  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n');) continue;
    }
    else if (card == "ngroup")
    {
      ifs >> G;
      if (N > 0)
        initialize_records();
    }
    else if (card == "niso")
    {
      ifs >> N;
      if (G > 0)
        initialize_records();
    }
    else if (card == "name")
    {
      ++mat_count;
      ifs >> rec[mat_count].name;
    }
    else if (card == "nden")
    {
      ifs >> rec[mat_count].nden;
    }
    else if (card == "zaid")
    {
      ifs >> rec[mat_count].zaid;
    }
    else if (card == "transport")
    {
      read_vector(ifs, rec[mat_count].sigma_tr);
    }
    else if (card == "total")
    {
      read_vector(ifs, rec[mat_count].sigma_t);
    }
    else if (card == "nusf")
    {
      rec[mat_count].is_fiss = true;
      read_vector(ifs, rec[mat_count].nusf);
    }
    else if (card == "chi")
    {
      rec[mat_count].is_fiss = true;
      read_vector(ifs, rec[mat_count].chi);
    }
    else if (card == "source")
    {
      read_vector(ifs, rec[mat_count].source);
    }
    else if (card == "scatter")
    {
      // initialize a matrix with zero entries for self-scattering
      SparseMatrix tmp_sparse(G, G);
      for (int g{0}; g < G; ++g)
        tmp_sparse.Set(g, g, 0.0);

      double val;
      for (int i{0}; i < G; ++i)
      {
        for (int j{0}; j < G; ++j)
        {
          ifs >> val;
          // abs to potentially allow for higher-order scattering in the future
          if (abs(val) > 0.0)
            tmp_sparse.Set(i, j, val);
        }
      }
      // 0 - do not skip zero entries (necessary for diagonal)
      // true - fix empty rows (if a row is empty, store a zero on the diagonal)
      tmp_sparse.Finalize(0, true);
      tmp_sparse.SortColumnIndices();
      rec[mat_count].sigma_s = tmp_sparse; // hopefully a move operation
    }
    else
    {
      cerr << card << '\n';
      throw Invalid_input_card{card};
    }
  }

  for (XSRecord& record : rec)
  {
    record.calculate_removal_xs();
    record.self_scatter_rebalance();
    record.evaluate_upscatter();
  }

}

void XSLibrary::Print(ostream& os)
{
  os << "ngoup " << G << '\n';
  os << "niso " << N << '\n';
  for (int i{0}; i < N; ++i)
  {
    const XSRecord& record{rec[i]};
    os << '\n';
    os << "name " << record.name << '\n';
    if (record.zaid > 0)
      os << "zaid " << record.zaid << '\n';
    if (record.nden > 0.0) 
      os << "nden " << record.nden << '\n';
    os << "transport\n";
    write_vector(os, record.sigma_tr);
    os << "total\n";
    write_vector(os, record.sigma_r);

    os << "scatter\n";
    for (int g{0}; g < G; ++g)
    {
      for (int gprime{0}; gprime < G; ++gprime)
      {
        // this will write zeros as appropriate
        // note that Elem does not return zeros but if passed to a stream, it
        // will be output as a zero
        os << record.sigma_s.Elem(g, gprime) << ' ';
      }
      os << '\n';
    }

    if (record.is_fiss)
    {
      os << "nusf\n";
      write_vector(os, record.nusf);
      os << "chi\n";
      write_vector(os, record.chi);
    }
  }
}

void XSLibrary::initialize_records()
{
  rec.reserve(N);
  for (int i{0}; i < N; ++i)
    rec.push_back(XSRecord(G));
}
