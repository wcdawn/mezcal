#include "xslibrary.hpp"
#include "make_unique.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

using namespace mfem;

using std::cerr;

using std::ifstream;
using std::ostream;
using std::string;
using std::vector;

using std::accumulate;
using std::max_element;

using std::make_unique;

using std::setfill;
using std::setw;
using std::stringstream;

void XSRecord::set_ngroup(int ngroup, int nmoment)
{
  G = ngroup;
  N = nmoment;
  sigma_tr.resize(G);
  sigma_t.resize(G);
  sigma_r.SetSize(G, nmoment);
  nusf.resize(G);
  kappasf.resize(G);
  chi.resize(G);
  source.resize(G);
  is_upscatter.resize(G);
}

void XSRecord::calculate_removal_xs()
{
  for (int n{0}; n < N; ++n)
  {
    for (int g{0}; g < G; ++g)
    {
      if (n == 0)
        sigma_r(g, n) = sigma_t[g] - sigma_s.Elem(g, g);
      else
        sigma_r(g, n) = sigma_t[g];
    }
  }
}

void XSRecord::self_scatter_rebalance()
{
  for (int g{0}; g < G; ++g)
  {
    if (sigma_s.Elem(g, g) == 0.0)
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
    is_upscatter[g] = false;
    for (int i{0}; i < row_size; ++i)
    {
      if (row_columns[i] > g)
      {
        is_upscatter[g] = true;
        break;
      }
    }
  }
}

void XSLibrary::read_userxs(const char* const fname)
{
  const string fname_str{fname};
  read_userxs(fname_str);
}

void XSLibrary::read_userxs(const std::string& fname)
{
  ifstream ifs{fname};
  if (!ifs) throw Invalid_file_name{fname};
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
      if (N > 0) initialize_records();
    }
    else if (card == "niso")
    {
      ifs >> N;
      if (G > 0) initialize_records();
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
    else if (card == "kappasf")
    {
      rec[mat_count].is_fiss = true;
      read_vector(ifs, rec[mat_count].kappasf);
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
      SparseMatrix tmp_sparse{G, G};
      for (int g{0}; g < G; ++g) tmp_sparse.Set(g, g, 0.0);

      double val;
      for (int i{0}; i < G; ++i)
      {
        for (int j{0}; j < G; ++j)
        {
          ifs >> val;
          // abs to potentially allow for higher-order scattering in the future
          if (std::abs(val) > 0.0) tmp_sparse.Set(i, j, val);
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

// TODO move this or add a better declaration
// This is a "rather specialized" function that removes leading and trailing
// space characters. Not valid for general whitespace or only trimming on one
// side.
string trim(const string& str)
{
  const auto first{str.find_first_not_of(' ')};
  if (first == string::npos) return "";
  const auto last{str.find_last_not_of(' ')};
  return str.substr(first, (last - first + 1));
}

void XSLibrary::read_hdf5xs(const char* const fname, const char* const lib_name)
{
  const string fname_str{fname};
  const string lib_name_str{lib_name};
  read_hdf5xs(fname_str, lib_name_str);
}

void XSLibrary::read_hdf5xs(const string& fname, const string& lib_name)
{
  try
  {
    const H5::H5File fid{fname, H5F_ACC_RDONLY};
    Hdf5_access_wrapper acc{fid};
    acc.change_group(lib_name);

    // check version and format
    acc.change_dataset("format");
    const string format{acc.read_string()};
    acc.change_dataset("version");
    const string version{acc.read_string()};

    // eventually, this will select between various hdf5xs readers
    if ((format == "ISOTXS") && (version == "0.2"))
      read_hdf5xs_isotxs(acc, lib_name);
    else
      throw Invalid_input_card{"Invalid HDF5 version and/or format"};

    for (XSRecord& record : rec)
    {
      record.calculate_removal_xs();
      record.self_scatter_rebalance();
      record.evaluate_upscatter();
    }
  }

  // catch failure caused by the H5File operations
  catch (const H5::FileIException& error)
  {
    // only available in newer versions of HDF5
    // error.printErrorStack();
    MFEM_ABORT("H5::FileIException");
    throw;
  }

  // catch failure caused by the Group operations
  catch (const H5::GroupIException& error)
  {
    // only available in newer versions of HDF5
    // error.printErrorStack();
    MFEM_ABORT("H5::GroupIException");
    throw;
  }

  // catch failure caused by the DataSet operations
  catch (const H5::DataSetIException& error)
  {
    // only available in newer versions of HDF5
    // error.printErrorStack();
    MFEM_ABORT("H5::DataSetIException");
    throw;
  }

  // catch failure caused by the DataSpace operations
  catch (const H5::DataSpaceIException& error)
  {
    // only available in newer versions of HDF5
    // error.printErrorStack();
    MFEM_ABORT("H5::DataSpaceIException");
    throw;
  }

  // catch failure caused by the DataType operations
  catch (const H5::DataTypeIException& error)
  {
    // only available in newer versions of HDF5
    // error.printErrorStack();
    MFEM_ABORT("H5::DataTypeIException");
    throw;
  }

  catch (...)
  {
    MFEM_ABORT("Unexpected H5 error.");
    throw;
  }
}

void XSLibrary::read_hdf5xs_isotxs(Hdf5_access_wrapper& acc,
                                   const string& lib_name)
{
  constexpr int ell{0}; // store only zeroth order xs (for now)

  // get niso & ngroup to setup data structure
  acc.change_dataset("niso");
  N = acc.read_int();
  acc.change_dataset("ngroup");
  G = acc.read_int();
  initialize_records();

  // get the list of isotope names
  vector<string> iso_names(N);

  { // scope of "data"
    vector<char> data(N * hollerith_length);
    acc.change_dataset("hisonm");
    acc.read(data.data());

    for (int i{0}; i < N; ++i)
    {
      const string str{data.begin() + i * hollerith_length,
                       data.begin() + (i + 1) * hollerith_length};
      // trim is necessary for subsequent group access
      iso_names[i] = trim(str);
    }
  }

  for (int i{0}; i < N; ++i)
  {
    acc.change_group(lib_name + "/" + iso_names[i]);
    rec[i].name = iso_names[i];

    vector<double> data;

    // sigma_tr
    acc.change_dataset("ltrn");
    const int ltrn{acc.read_int()};
    data.resize(ltrn * G);
    acc.change_dataset("strpl");
    acc.read(data.data(), H5::PredType::NATIVE_DOUBLE);
    for (int g{0}; g < G; ++g) rec[i].sigma_tr[g] = data[g + ell];

    // sigma_t
    acc.change_dataset("ltot");
    const int ltot{acc.read_int()};
    data.resize(ltot * G);
    acc.change_dataset("stotpl");
    acc.read(data.data(), H5::PredType::NATIVE_DOUBLE);
    for (int g{0}; g < G; ++g) rec[i].sigma_t[g] = data[g + ell];

    acc.change_dataset("ifis");
    rec[i].is_fiss = acc.read_int(); // convert int to bool

    if (rec[i].is_fiss)
    {
      // sigma_f (and initialize nusf)
      data.resize(G);
      acc.change_dataset("sfis");
      acc.read(data.data(), H5::PredType::NATIVE_DOUBLE);
      for (int g{0}; g < G; ++g) rec[i].nusf[g] = data[g];

      // nu_f (and combine with sigma_f to finally calculate nusf)
      acc.change_dataset("snutot");
      acc.read(data.data(), H5::PredType::NATIVE_DOUBLE);
      for (int g{0}; g < G; ++g) rec[i].nusf[g] *= data[g];
    }

    acc.change_dataset("ichi");
    const int ichi{acc.read_int()};
    if (ichi == 1)
    {
      acc.change_dataset("chiso");
      acc.read(data.data(), H5::PredType::NATIVE_DOUBLE);
      for (int g{0}; g < G; ++g) rec[i].chi[g] = data[g];
    }

    // get the integer total number of scattering blocks
    acc.change_dataset("nscatter");
    const int nscatter{acc.read_int()};

    // initialize the sparse matrix that will be used
    // initialize with zero-entries for self-scattering
    SparseMatrix tmp_sparse{G, G};
    for (int g{0}; g < G; ++g) tmp_sparse.Set(g, g, 0.0);

    // get ready to track if total scattering is present
    bool total_scattering_present{false};

    // read and accumulate all of the scattering blocks
    for (int j{0}; j < nscatter; ++j)
    {
      stringstream name;
      name << "scatter" << setfill('0') << setw(3) << j + 1; // one-based index
      acc.change_group(lib_name + "/" + iso_names[i] + "/" + name.str());

      // check legendre scattering order
      acc.change_dataset("legendre");
      const int legendre{acc.read_int()};
      // only read zeroth order scattering for now
      if (legendre != 0) continue;

      // check scattering type
      acc.change_dataset("type");
      const int scattering_type(acc.read_int());
      /*
       * The "type" returned is:
       * Type | Description
       * -----|------------
       * 0    | Total scattering
       * 1    | Elastic scattering
       * 2    | Inelastic scattering
       * 3    | (n,2n) scattering
       * other| Undefined
       */
      if ((total_scattering_present) && (scattering_type != 0)) continue;
      total_scattering_present =
      (total_scattering_present) && (scattering_type == 0);

      acc.change_dataset("csr_i");
      auto csr_i{make_unique<int[]>(G + 1)};
      acc.read(csr_i.get(), H5::PredType::NATIVE_INT);

      acc.change_dataset("csr_j");
      auto nnz{acc.dataset().getStorageSize() / sizeof(int)};
      auto csr_j{make_unique<int[]>(nnz)};
      acc.read(csr_j.get(), H5::PredType::NATIVE_INT);

      acc.change_dataset("csr_data");
      auto csr_data{make_unique<double[]>(nnz)};
      acc.read(csr_data.get(), H5::PredType::NATIVE_DOUBLE);

      // adjust for zero-based indexing
      for (int k{0}; k < G + 1; ++k) --csr_i[k];
      for (decltype(nnz) k{0}; k < nnz; ++k) --csr_j[k];

      // adjust if n2n scattering
      if (scattering_type == 3)
        for (decltype(nnz) k{0}; k < nnz; ++k) csr_data[k] *= 2.0;

      SparseMatrix tmp_tmp{csr_i.release(), csr_j.release(), csr_data.release(),
                           G, G};
      tmp_sparse += tmp_tmp;
    }

    // 0 - do not skip zero entries (necessary for diagonal)
    // true - fix empty rows (if a row is empty, store a zero on the diagonal)
    tmp_sparse.Finalize(0, true);
    tmp_sparse.SortColumnIndices();
    rec[i].sigma_s = tmp_sparse; // hopefully a move operation

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
    if (record.zaid > 0) os << "zaid " << record.zaid << '\n';
    if (record.nden > 0.0) os << "nden " << record.nden << '\n';
    os << "transport\n";
    write_vector(os, record.sigma_tr);
    os << "total\n";
    write_vector(os, record.sigma_t);

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
      os << "kappasf\n";
      write_vector(os, record.kappasf);
      os << "chi\n";
      write_vector(os, record.chi);
    }
  }
}

void XSLibrary::initialize_records()
{
  rec.reserve(N);
  for (int i{0}; i < N; ++i) rec.push_back(XSRecord(G));
}

void XSLibrary::calc_tcp0()
{
  for (int n{0}; n < N; ++n)
  {
    XSRecord &record{rec[n]};
    for (int g{0}; g < G; ++g)
    {
      double sigma_s_hat = record.sigma_s.Elem(g,g) + 
        (record.sigma_tr[g] - record.sigma_t[g]);
      record.sigma_s.Elem(g,g) = sigma_s_hat;
    }
    // really just need
    // sigma_t=sigma_tr
    // but store the old total xs in the transport xs array in case we want it later
    std::swap(record.sigma_t, record.sigma_tr);
  }
  return;
}
