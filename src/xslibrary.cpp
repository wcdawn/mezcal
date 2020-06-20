#include "xslibrary.hpp"

#include <fstream>
#include <string>
#include <vector>

using namespace mfem;

using std::cerr;
using std::ostream;
using std::ifstream;
using std::string;
using std::vector;

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
      continue;
    if (card == "ngroup")
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
      // By initializing with zeros on the diagonal, we ensure that the
      // diagonal (self-scattering) entries will exist in the matrix.
      Vector zero_diag(G);
      zero_diag = 0.0;
      SparseMatrix tmp_sparse(zero_diag);
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
      tmp_sparse.Finalize();
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
  }

}

void XSLibrary::Print(ostream& os)
{
  os << "ngoup " << G << '\n';
  os << "niso " << N << '\n';
  for (int i{0}; i < N; ++i)
  {
    const XSRecord* const record{&rec[i]};
    os << '\n';
    os << "name " << record->name << '\n';
    if (record->zaid > 0)
      os << "zaid " << record->zaid << '\n';
    if (record->nden > 0.0)
      os << "nden " << record->nden << '\n';
    os << "transport\n";
    write_vector(os, record->sigma_tr);
    os << "total\n";
    write_vector(os, record->sigma_r);

    os << "scatter\n";
    for (int g{0}; g < G; ++g)
    {
      for (int gprime{0}; gprime < G; ++gprime)
      {
        // this will write zeros as appropriate
        // note that Elem does not return zeros but if passed to a stream, it
        // will be output as a zero
        os << record->sigma_s.Elem(g,gprime) << ' ';
      }
      os << '\n';
    }

    if (record->is_fiss)
    {
      os << "nusf\n";
      write_vector(os, record->nusf);
      os << "chi\n";
      write_vector(os, record->chi);
    }
  }
}

void XSRecord::calculate_removal_xs()
{
  for (int g{0}; g < G; ++g)
    sigma_r[g] = sigma_t[g] - sigma_s.SearchRow(g, g);
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

void XSLibrary::initialize_records()
{
  for (int i{0}; i < N; ++i) rec.push_back(XSRecord(G));
}
