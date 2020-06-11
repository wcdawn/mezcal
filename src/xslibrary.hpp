#ifndef XSLIBRARY
#define XSLIBRARY

#include <string>
#include <vector>
#include <exception>

#include <iostream>

struct XSRecord
{
  XSRecord() {} 
  XSRecord(int ngroup) : G{ngroup}
  {
    sigma_tr.resize(G);
    sigma_r.resize(G);
    nusf.resize(G);
    chi.resize(G);
  }

  std::string name;
  int zaid;
  double nden;

  int G;

  std::vector<double> sigma_tr;
  std::vector<double> sigma_r;
  std::vector<double> nusf;
  std::vector<double> chi;
};

class XSLibrary
{
  public:

    int ngroup() { return G; }
    int niso() { return N; }

    void read_userxs(std::string fname);

    class Invalid_input_card : std::exception
    {
      public:
        Invalid_input_card(std::string str)
        {
          msg = "Invalid input card: " + str;
        }
        const char* what() const noexcept override
        {
          return msg.c_str();
        }
      private:
        std::string msg;
    };

    class Invalid_file_name : std::exception
    {
      public:
        Invalid_file_name(std::string str)
        {
          msg = "Invalid file name: " + str;
        }
        const char* what() const noexcept override
        {
          return msg.c_str();
        }
      private:
        std::string msg;
    };

    std::vector<XSRecord> rec;

  private:
    int G{0}; // number of energy groups
    int N{0}; // number of materials/isotopes present

    void initialize_records();

    static constexpr char comment_char{'#'};

    static void read_vector(std::ifstream&, std::vector<double>&);
};

#endif
