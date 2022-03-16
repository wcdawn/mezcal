#include "output.hpp"
#include <cmath>

#include "saaf_void_tau_coefficient.hpp"

using namespace mfem;

constexpr int xs_temp_idx = 0;

void total_xs_plot(ParMesh *pmesh,
                   const Material_library &mat_lib,
                   const std::string out_name)
{

  H1_FECollection fec{1, pmesh->Dimension()}; // linear elements...
  ParFiniteElementSpace fespace{pmesh, &fec};

  ParaViewDataCollection output{out_name, pmesh};

  std::vector<ParGridFunction> tot_pgf;
  tot_pgf.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    tot_pgf.emplace_back(&fespace);

  std::vector<ParGridFunction> void_pgf;
  void_pgf.reserve(mat_lib.ngroup());
  for (int g{0}; g < mat_lib.ngroup(); ++g)
    void_pgf.emplace_back(&fespace);

  const size_t group_width{std::to_string(mat_lib.ngroup()-1).length()};

  for (int g{0}; g < mat_lib.ngroup(); ++g)
  {

    Vector tot_vec(pmesh->attributes.Max());
    for (int i{0}; i < pmesh->attributes.Max(); ++i)
      tot_vec(i) = mat_lib.mat[i].rec[xs_temp_idx].sigma_t[g];
    PWConstCoefficient tot_func{tot_vec};
    tot_pgf[g].ProjectCoefficient(tot_func);

    // a good value is probably closer to 1e-3
    constexpr double squigma = 4.5e-2; // default=0.5
    SAAFVoidTauCoefficient saaf_void_func{pmesh, mat_lib, g, squigma};
    //SAAFVoidTauCoefficient saaf_void_func{pmesh, mat_lib, g};
    void_pgf[g].ProjectCoefficient(saaf_void_func);

    std::ostringstream record_name;
    record_name << "total_xs_g"
                << std::setfill('0')
                << std::setw(group_width)
                << g;
    output.RegisterField(record_name.str(), &tot_pgf[g]);

    std::ostringstream saaf_record_name;
    saaf_record_name << "saaf_void_inv_tot_g"
                     << std::setfill('0')
                     << std::setw(group_width)
                     << g;
    output.RegisterField(saaf_record_name.str(), &void_pgf[g]);

    if (fespace.GetMyRank() == 0)
      std::cout << "total_xs_g=" << g
                << " ratio=" << tot_vec.Max() / tot_vec.Min()
                << " modified_ratio=" << void_pgf[g].Max() / void_pgf[g].Min()
                << std::endl;

  }

  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  return;
}

void coarse_energy_plot(ParMesh *pmesh,
                        const Result &scalar_flux,
                        const std::vector<int> collapse_vector,
                        const std::string out_name)
{

  const int ncoarse{static_cast<int>(collapse_vector.size())+1};

  std::vector<ParGridFunction> coarse_flux;
  coarse_flux.reserve(ncoarse);
  for (int i{0}; i < ncoarse; ++i)
    coarse_flux.emplace_back(scalar_flux(0));

  for (ParGridFunction &pgf : coarse_flux)
    pgf = 0.0;

  int coarse_idx{0};
  
  for (int g{0}; g < scalar_flux.ngroup(); ++g)
  {

    // see if we have passed the coarse energy boundary but also make sure we
    // don't access collapse_vector out-of-bounds
    // note: takes advantage of short-circuiting
    if ((coarse_idx < static_cast<int>(collapse_vector.size())) && 
        (g >= collapse_vector[coarse_idx]))
      ++coarse_idx;

    coarse_flux[coarse_idx] += scalar_flux(g);

  }

  const size_t group_width{std::to_string(scalar_flux.ngroup()).length()};

  ParaViewDataCollection output{out_name, pmesh};

  for (int g{0}; g < ncoarse; ++g)
  {
    std::ostringstream coarse_flux_record;
    coarse_flux_record << "coarse_flux_g"
                       << std::setfill('0')
                       << std::setw(group_width)
                       << g;
    output.RegisterField(coarse_flux_record.str(), &coarse_flux[g]);
  }

  output.SetFormat(DataCollection::PARALLEL_FORMAT);
  output.Save();

  return;
}

void mesh_stats(ParMesh *pmesh,
                std::ostream &os,
                Material_library *mat_lib)
{

  const int myid{pmesh->GetMyRank()};

  Vector area_vol(pmesh->attributes.Max());
  area_vol = 0.0;

  for (int e{0}; e < pmesh->GetNE(); ++e)
  {
    const int att{pmesh->GetAttribute(e)-1};
    const double vol{pmesh->GetElementVolume(e)};
    area_vol(att) += vol;
  }

  const auto old_precision{os.precision()};
  os << std::setprecision(20);

  if (myid == 0) os << "=== AREA/VOLUME ===" << std::endl;
  if (myid == 0) os << "mat # | vol [cm^2 or cm^3] | exact vol | rel. err. [%] | correction factor" << std::endl;
  double total_area_vol{0.0};
  for (int i{0}; i < pmesh->attributes.Max(); ++i)
  {
    double sum_area_vol{0.0};
    MPI_Allreduce(&area_vol(i), &sum_area_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    total_area_vol += sum_area_vol;
    if (myid == 0) os << i+1 << " " << sum_area_vol;

    if ((mat_lib) && (mat_lib->mat[i].exact_area_volume >= 0.0))
    {
      const double &exact_area_vol{mat_lib->mat[i].exact_area_volume};
      const double rel_error{(exact_area_vol > 0.0) ? (sum_area_vol - exact_area_vol) / exact_area_vol : 0.0};
      const double factor{(sum_area_vol > 0.0) ? exact_area_vol / sum_area_vol : 1.0};
      
      for (int t = 0; t < mat_lib->ntemp(); ++t)
      {
        XSRecord &rec{mat_lib->mat[i].rec[t]};

        for (Material_isotope &isotope : mat_lib->mat[i].iso)
          isotope.nden *= factor;

        for (int g{0}; g < mat_lib->ngroup(); ++g)
        {
          rec.sigma_t[g] *= factor;
          rec.sigma_tr[g] *= factor;
          rec.nusf[g] *= factor;
        }

        rec.sigma_r *= factor;
        rec.sigma_s *= factor;
      }

      if (myid == 0)
        std::cout << " " << exact_area_vol
                  << " " << rel_error*1e2
                  << " " << factor;

    }

    if (myid == 0) std::cout << std::endl;

  }
  if (myid == 0) os << "total " << total_area_vol << std::endl;
  if (myid == 0) os << std::endl;

  os << std::setprecision(old_precision);

  return;
}

class PinCoefficient final : public mfem::Coefficient
{
public:
  PinCoefficient(const Pin &p_) : p{p_} {}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    double x[3];
    Vector v(x, 3);
    T.Transform(ip, v);
    const double dist = std::sqrt(std::pow(p.x-v(0),2) + std::pow(p.y-v(1),2));
    return (dist <= p.radius) ? 1.0 : 0.0;
  }
private:
  const Pin &p;
};

void calc_pin_power(const QuadratureFunction &power_quad,
                    const std::vector<Pin> &pins)
{

  const QuadratureSpace * const qspace = power_quad.GetSpace();
  Mesh *mesh = qspace->GetMesh();
    
  const double * pquad = power_quad.HostRead();

  int idx = 0;

  Vector pin_power(pins.size());
  pin_power = 0.0;
  double lost_power = 0.0;

  for (int e = 0; e < qspace->GetNE(); ++e)
  {

    const IntegrationRule &ir = qspace->GetElementIntRule(e);
    ElementTransformation *T = mesh->GetElementTransformation(e);
    
    for (int i = 0; i < ir.GetNPoints(); ++i)
    {

      const IntegrationPoint &ip = ir.IntPoint(i);
      const double i_weight = ip.weight;

      T->SetIntPoint(&ip);
      const double t_weight = T->Weight();

      Vector r(3);
      T->Transform(ip, r);

      double p = pquad[idx];

      if (p > 0.0)
      {

        p *= i_weight*t_weight;

        bool found = false;
        for (int j = 0; j < pin_power.Size(); ++j)
        {

          const double dist = std::sqrt( std::pow(r(0)-pins[j].x,2) +
            std::pow(r(1)-pins[j].y,2) );

          if (dist < pins[j].radius)
          {
            pin_power(j) += p;
            found = true;
            break;
          }

        }

        if (!found)
          lost_power += p;

      }

      ++idx;

    }

  }

  double *p = pin_power.HostReadWrite();
  for (int i = 0; i < pin_power.Size(); ++i)
  {
    const double local_val = p[i];
    double val;
    MPI_Allreduce(&local_val, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    p[i] = val;
  }

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0)
  {

    // NOTE:
    // some pins may be fractional so need to do normalization in post processing
    std::cout << "pin_power" << std::endl;
    pin_power.Print(std::cout, 1);

    if (lost_power > 0.0)
      std::cout << "WARNING: lost_power=" << lost_power << std::endl;

  }

  return;
}

void edit_flux(const std::string &stub,
               const std::vector<std::unique_ptr<ParGridFunction>> &scalar_flux)
{

  ParFiniteElementSpace *parfespace = scalar_flux[0]->ParFESpace();
  ParMesh *pmesh = parfespace->GetParMesh();
  const int myid = parfespace->GetMyRank();

  const int ngroup = scalar_flux.size();

  std::vector<double> flux_tot(pmesh->attributes.Max()*ngroup);

  for (int i = 0; i < pmesh->attributes.Max(); ++i)
  {

    Vector v(pmesh->attributes.Max());
    v = 0.0;
    v(i) = 1.0;
    PWConstCoefficient coeff{v};

    ParLinearForm b{parfespace};
    b.AddDomainIntegrator(new DomainLFIntegrator{coeff});
    b.Assemble();

    for (int g = 0; g < ngroup; ++g)
    {

      const int idx = g + ngroup*i;
      flux_tot[idx] = b(*scalar_flux[g]);

    }

  }

  if (myid == 0)
  {
    std::ofstream ofs{stub + ".flux"};
    for (int i = 0; i < pmesh->attributes.Max(); ++i)
    {
      ofs << "attribute=" << i+1 << std::endl;
      const auto old_precision{ofs.precision()};
      ofs << std::setprecision(20);
      for (int g = 0; g < ngroup; ++g)
      {
        const int idx = g + ngroup*i;
        ofs << flux_tot[idx] << std::endl;
      }
      ofs << std::setprecision(old_precision);
    }
  }

  return;
}
