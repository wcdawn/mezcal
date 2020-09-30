#include "material_library.hpp"

using namespace mfem;

using std::cerr;

using std::cout;
using std::endl;

using std::ifstream;
using std::string;

using std::vector;

Material_library::Material_library(const XSLibrary& xslib)
  : xs{xslib}, G{xs.ngroup()}, M{xs.nmoment()}
{
}

void Material_library::read_material(ifstream& ifs)
{
  string card;

  int mat_count{-1};

  ifs >> card;
  if (card != "[MATERIAL]")
    throw Invalid_material_card{"Expected [MATERIAL]. Got: " + card};

  while (ifs >> card)
  {

    if (card == "name")
    {
      string name;
      ifs >> name;
      mat.push_back(Material_record{});
      ++mat_count;
      mat[mat_count].name = name;
      mat[mat_count].G    = xs.ngroup();
      continue;
    }

    // determine if the card is an isotope name
    int iso_idx{0};
    for (iso_idx = 0; iso_idx < xs.niso(); ++iso_idx)
      if (card == xs.rec[iso_idx].name) break;

    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n');)
        ;
    }
    else if (iso_idx < xs.niso())
    {
      // this is an isotope
      double nden;
      ifs >> nden;
      mat[mat_count].iso.push_back(
      Material_isotope{card, nden, iso_idx, xs.rec[iso_idx]});
    }
    else if (card == "[STATE]")
    {
      // expecting to lead directly into the first [STATE] block
      // cannot lead into any other block
      reset_block(ifs, "[STATE]");
      break;
    }
    else
    {
      cerr << "INVALID CARD\n";
      cerr << card << '\n';
      throw Invalid_material_card{"Invalid card: " + card};
    }
  }

  for (size_t i{0}; i < mat.size(); ++i)
  {
    mat[i].niso = static_cast<int>(mat[i].iso.size());
    mat[i].calculate_macroscopic_xs();
  }
}

void Material_record::calculate_macroscopic_xs()
{

  rec.set_ngroup(ngroup());
  SparseMatrix scat(ngroup(), ngroup());

  for (int i{0}; i < niso; ++i)
  {
    // this is a hack because there is no multiply/scaling operator provided for
    // sparse matrices and "rec" is const-qualified
    SparseMatrix tmp{iso[i].rec.sigma_s};
    tmp *= iso[i].nden;
    scat += tmp;
    for (int g{0}; g < rec.ngroup(); ++g)
    {
      rec.sigma_tr[g] += iso[i].nden * iso[i].rec.sigma_tr[g];
      rec.sigma_t[g] += iso[i].nden * iso[i].rec.sigma_t[g];
      rec.nusf[g] += iso[i].nden * iso[i].rec.nusf[g];
      // note: fixed source is simply summed without number densities
      rec.source[g] += iso[i].rec.source[g];
    }

    for (int n{0}; n < rec.nmoment(); ++n)
      for (int g{0}; g < rec.ngroup(); ++g)
        rec.sigma_r(g,n) += iso[i].nden * iso[i].rec.sigma_r(g,n);

    rec.is_fiss = rec.is_fiss || iso[i].rec.is_fiss;
  }
  rec.sigma_s = scat;
  rec.sigma_s.Finalize();
  rec.sigma_s.SortColumnIndices();
  rec.chi = flux_free_chi(); // initialize chi with the flux_free_chi formula
}

vector<double> Material_record::flux_free_chi() const
{
  vector<double> numerator(ngroup());
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec.is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec.nusf[gprime];
    for (int g{0}; g < ngroup(); ++g) numerator[g] += iso[i].rec.chi[g] * f_i;
    denominator += f_i;
  }
  if (denominator > 0.0)
    for (int g{0}; g < ngroup(); ++g) numerator[g] /= denominator;
  return numerator;
}

double Material_record::flux_free_chi(int g) const
{
  double numerator{0.0};
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec.is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec.nusf[gprime];
    numerator += iso[i].rec.chi[g] * f_i;
    denominator += f_i;
  }
  if (denominator > 0.0) numerator /= denominator;
  return numerator;
}

vector<double> Material_record::calculate_chi(const vector<double>& phi_g) const
{
  vector<double> numerator(ngroup());
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec.is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec.nusf[gprime] * phi_g[gprime];
    for (int g{0}; g < ngroup(); ++g) numerator[g] += iso[i].rec.chi[g] * f_i;
    denominator += f_i;
  }
  if (denominator > 0.0)
    for (int g{0}; g < ngroup(); ++g) numerator[g] /= denominator;
  return numerator;
}

double Material_record::calculate_chi(const vector<double>& phi_g, int g) const
{
  double numerator{0.0};
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec.is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec.nusf[gprime] * phi_g[gprime];
    numerator += iso[i].rec.chi[g] * f_i;
    denominator += f_i;
  }
  if (denominator > 0.0) numerator /= denominator;
  return numerator;
}

void Material_library::reset_block(ifstream& ifs, const string& block_head)
{
  for (size_t i{0}; i < block_head.size(); ++i) ifs.unget();
}

void Material_library::set_nmoment(int m)
{
  M = m;

  if (M > mat[0].rec.sigma_r.Width())
  {
    for (Material_record& this_mat : mat)
    {
      XSRecord& record{this_mat.rec};
      // after SetSize, need to rebuild entire matrix because the new matrix is
      // zero-initialized
      record.sigma_r.SetSize(G,M);
      for (int n{0}; n < M; ++n)
      {
        for (int g{0}; g < G; ++g)
        {
          if (n == 0)
            record.sigma_r(g,n) = record.sigma_t[g] - record.sigma_s.Elem(g,g);
          else
            record.sigma_r(g,n) = record.sigma_t[g];
        }
      }
    }
  }
}
