#include "material_library.hpp"

#include <cmath>

using namespace mfem;

using std::cerr;

using std::cout;
using std::endl;

using std::ifstream;
using std::string;

using std::vector;

Material_library::Material_library(const std::vector<XSLibrary> &xslib)
  : xs{xslib}, G{xs[0].ngroup()}, M{xs[0].nmoment()}, NT{static_cast<int>(xs.size())}
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
      mat[mat_count].G    = G;
      mat[mat_count].NT   = NT;
      continue;
    }
    else if (card == "exact_area_volume")
    {
      ifs >> mat[mat_count].exact_area_volume;
      continue;
    }
    else if (card == "thermal_material")
    {
      std::string str;
      ifs >> str;
      mat[mat_count].k_mat = Material_record::read_thermal_material(str);
      continue;
    }

    // determine if the card is an isotope name
    int iso_idx{0};
    for (iso_idx = 0; iso_idx < xs[0].niso(); ++iso_idx)
      if (card == xs[0].rec[iso_idx].name) break;

    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n');) ;
    }
    else if (iso_idx < xs[0].niso())
    {
      // this is an isotope
      double nden;
      ifs >> nden;
      Material_isotope this_iso;
      this_iso.name = card;
      this_iso.nden = nden;
      this_iso.idx  = iso_idx;
      for (int t = 0; t < NT; ++t)
        this_iso.rec.push_back(&xs[t].rec[iso_idx]);
      mat[mat_count].iso.push_back(this_iso);
    }
    else if (card == "chi")
    {
      rx_wide_chi = true;
      chi.reserve(G);
      for (int g{0}; g < G; ++g)
      {
        double tmp;
        ifs >> tmp;
        chi.emplace_back(tmp);
      }
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

  rec.resize(NT);
  is_fiss = false;
  for (int t = 0; t < NT; ++t)
  {

    rec[t].set_ngroup(ngroup());
    SparseMatrix scat(ngroup(), ngroup());

    for (int i{0}; i < niso; ++i)
    {
      // this is a hack because there is no multiply/scaling operator provided for
      // sparse matrices and "rec" is const-qualified
      SparseMatrix tmp{iso[i].rec[t]->sigma_s};
      tmp *= iso[i].nden;
      scat += tmp;
      for (int g{0}; g < rec[t].ngroup(); ++g)
      {
        rec[t].sigma_tr[g] += iso[i].nden * iso[i].rec[t]->sigma_tr[g];
        rec[t].sigma_t[g]  += iso[i].nden * iso[i].rec[t]->sigma_t[g];
        rec[t].nusf[g]     += iso[i].nden * iso[i].rec[t]->nusf[g];
        // note: fixed source is simply summed without number densities
        rec[t].source[g] += iso[i].rec[t]->source[g];
      }

      for (int n{0}; n < rec[t].nmoment(); ++n)
        for (int g{0}; g < rec[t].ngroup(); ++g)
          rec[t].sigma_r(g,n) += iso[i].nden * iso[i].rec[t]->sigma_r(g,n);

      rec[t].is_fiss = rec[t].is_fiss || iso[i].rec[t]->is_fiss;
      for (int g{0}; g < rec[t].ngroup(); ++g)
        rec[t].is_upscatter[g] = rec[t].is_upscatter[g] || iso[i].rec[t]->is_upscatter[g];
    }
    rec[t].sigma_s = scat;
    rec[t].sigma_s.Finalize();
    rec[t].sigma_s.SortColumnIndices();
    rec[t].chi = flux_free_chi(t); // initialize chi with the flux_free_chi formula
    is_fiss = is_fiss || rec[t].is_fiss;
  }

  return;
}

std::vector<double> Material_record::flux_free_chi(const int temp_idx) const
{
  std::vector<double> numerator(ngroup());
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec[temp_idx]->is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec[temp_idx]->nusf[gprime];
    for (int g{0}; g < ngroup(); ++g)
      numerator[g] += iso[i].rec[temp_idx]->chi[g] * f_i;
    denominator += f_i;
  }
  if (denominator > 0.0)
    for (int g{0}; g < ngroup(); ++g)
      numerator[g] /= denominator;
  return numerator;
}

double Material_record::flux_free_chi(const int g, const int temp_idx) const
{
  double numerator{0.0};
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec[temp_idx]->is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec[temp_idx]->nusf[gprime];
    numerator += iso[i].rec[temp_idx]->chi[g] * f_i;
    denominator += f_i;
  }
  if (denominator > 0.0) numerator /= denominator;
  return numerator;
}

std::vector<double> Material_record::calculate_chi(const vector<double>& phi_g,
                                                   const int temp_idx) const
{
  std::vector<double> numerator(ngroup());
  double denominator{0.0};

  for (double& num : numerator)
    num = 0.0;

  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec[temp_idx]->is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec[temp_idx]->nusf[gprime] * phi_g[gprime];
    for (int g{0}; g < ngroup(); ++g)
      numerator[g] += iso[i].rec[temp_idx]->chi[g] * f_i;
    denominator += f_i;
  }

  if (denominator > 0.0)
  {
    for (double& num : numerator)
    {
      num /= denominator;
    }
  }

  return numerator;
}

double Material_record::calculate_chi(const vector<double>& phi_g,
                                      const int g,
                                      const int temp_idx) const
{
  double numerator{0.0};
  double denominator{0.0};
  for (int i{0}; i < niso; ++i)
  {
    if (!iso[i].rec[temp_idx]->is_fiss) continue;
    double f_i{0.0};
    for (int gprime{0}; gprime < ngroup(); ++gprime)
      f_i += iso[i].nden * iso[i].rec[temp_idx]->nusf[gprime] * phi_g[gprime];
    numerator += iso[i].rec[temp_idx]->chi[g] * f_i;
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

  if (M > mat[0].rec[0].sigma_r.Width())
  {
    for (Material_record& this_mat : mat)
    {
      for (int t = 0; t < NT; ++t)
      {
        XSRecord& record{this_mat.rec[t]};
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
}

// returns value in [W/cm/K]
double Material_record::thermal_conductivity(Thermal_material k_mat_in, double T)
{

  constexpr double Tmax = 1500.0;
  constexpr double Tmin = 200.0;

  if (T > Tmax)
  {
    std::cout << "TEMPERATURE_ERROR: too hot! T=" << T << std::endl;
    T = Tmax;
  }
  else if (T < Tmin)
  {
    std::cout << "TEMPERATURE_ERROR: too cold! T=" << T << std::endl;
    T = Tmin;
  }

  switch (k_mat_in)
  {
    case (SS316):
      return (-7.301e-6*std::pow(T,2) + 2.716e-2*T + 6.308)*1e-2;
    case (NaK78):
      return 2.75860242e-02 + 8.56037380e-04*T + -1.00784204e-06*std::pow(T,2) +
        3.78618012e-10*std::pow(T,3);
    case (BeO):
      return 7.82844502e+00 + -3.19009906e-02*T + 5.58861707e-05*std::pow(T,2) +
        -4.57107468e-08*std::pow(T,3) + 1.42900889e-11*std::pow(T,4);
    case (air):
      {
        const double Tr = T/132.52;
        const double Trc = std::cbrt(Tr);
        constexpr double c_1 = 33.9729025;
        constexpr double c_2 = -164.702679;
        constexpr double c_3 = 262.108546;
        constexpr double c_4 = -21.5346955;
        constexpr double c_5 = -443.455815;
        constexpr double c_6 = 607.339582;
        constexpr double c_7 = -368.790121;
        constexpr double c_8 = 111.296674;
        constexpr double c_9 = -13.4122465;
        const double lambda_or = c_1/Tr + c_2/std::pow(Trc,2) + c_3/Trc + c_4 +
          c_5*Trc + c_6*std::pow(Trc,2) + c_7*Tr + c_8*std::pow(Trc,4) +
          c_9*std::pow(Trc,5);
        return lambda_or*4.358e-3*1e-2;
      }
    case (Be):
      {
        const double Tc = T - 273.15;
        return (189.8 - 0.2964*Tc + 2.5429e-4*std::pow(Tc,2) - 1.0104e-7*std::pow(Tc,3))*1e-2;
      }
    case (B4C):
      {
        const double Tc = T - 273.15;
        return 1.0/(6.87+0.0171*Tc)*4.184;
      }
    case(UZrH):
      {
        const double Tc = T - 273.15;
        return (3.7267e-2 + 3.817e-5*Tc - 7.01e-8*std::pow(Tc,2) + 1.223e-10*std::pow(Tc,3) - 8.31e-14*std::pow(Tc,4))*4.184;
      }
  }

  return -1e-3;
}

Material_record::Thermal_material Material_record::read_thermal_material(const std::string &str)
{

  if (str == "SS316")
    return Material_record::Thermal_material::SS316;
  else if (str == "NaK78")
    return Material_record::Thermal_material::NaK78;
  else if (str == "BeO")
    return Material_record::Thermal_material::BeO;
  else if (str == "air")
    return Material_record::Thermal_material::air;
  else if (str == "Be")
    return Material_record::Thermal_material::Be;
  else if (str == "B4C")
    return Material_record::Thermal_material::B4C;
  else if (str == "UZrH")
    return Material_record::Thermal_material::UZrH;

  std::cout << "ERROR: invalid thermal material -- " << str << std::endl;
  throw;
  return Material_record::Thermal_material::SS316;

}
