#include "xslibrary.hpp"

#include <fstream>
#include <string>
#include <vector>

using std::ifstream;
using std::string;
using std::vector;

void XSLibrary::read_userxs(const char* fname)
{
  const string fname_str{fname};
  read_userxs(fname_str);
}

void XSLibrary::read_userxs(std::string fname)
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
    else if (card == "transport")
    {
      read_vector(ifs, rec[mat_count].sigma_tr);
    }
    else if (card == "total")
    {
      read_vector(ifs, rec[mat_count].sigma_r); // TODO fix this
    }
    else if (card == "nusf")
    {
      read_vector(ifs, rec[mat_count].nusf);
    }
    else if (card == "chi")
    {
      read_vector(ifs, rec[mat_count].chi);
    }
    else if (card == "source")
    {
      read_vector(ifs, rec[mat_count].source);
    }
    else
    {
      throw Invalid_input_card{card};
    }
  }
}

void XSLibrary::initialize_records()
{
  for (int i{0}; i < N; ++i) rec.push_back(XSRecord(G));
}

void XSLibrary::read_vector(ifstream& ifs, vector<double>& vec)
{
  for (double& val : vec) ifs >> val;
}
