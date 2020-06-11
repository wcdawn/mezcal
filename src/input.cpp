#include "input.hpp"

#include <cstring>
using std::strcpy;
using std::strlen;

#include <iostream>
using std::cout;

using std::string;
using std::ifstream;
using std::ostream;
using std::vector;

Input::Input(std::string fname)
{
  open(fname);
}

void Input::open(string fname)
{
  ifs.open(fname);
  if (!ifs) throw Invalid_file_name{fname};
  stub = path_stub(fname);
}


void Input::read_core()
{
  if (read_core_done) throw Invalid_reread_input{};
  string card;

  ifs >> card;
  if (card != "[CORE]") 
    throw Invalid_input_card{"Expected [CORE]. Got: " + card};

  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n'); )
      continue;
    }
    else if (card == "[STATE]")
    {
      reset_state(ifs);
      return;
    }
    else if (card == "mesh_file")
    {
      ifs >> fname_mesh;
      fname_mesh = stub + fname_mesh;
    }
    else if (card == "boundary_conditions")
    {
      read_vector(ifs, bc);
    }
    else if (card == "refine_serial")
    {
      ifs >> rs;
    }
    else if (card == "refine_parallel")
    {
      ifs >> rp;
    }
    else if (card == "reference")
    {
      string reference_name;
      ifs >> reference_name;
      ref = Analytic::reference_str_to_enum(reference_name);
    }
    else if (card == "xs_file")
    {
      ifs >> fname_xs;
      fname_xs = stub + fname_xs;
    }
    else
      throw Invalid_input_card{card};
  }

  read_core_done = true;
}

void Input::read_state()
{
  string card;

  ifs >> card;
  if (card != "[STATE]")
    throw Invalid_input_card{"Expected [STATE]. Got: " + card};

  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n'); )
      continue;
    }
    else if (card == "[STATE]")
    {
      reset_state(ifs);
      return;
    }
    else if (card == "max_outer")
    {
      ifs >> tol.max_outer;
    }
    else if (card == "k_tol")
    {
      ifs >> tol.k;
    }
    else if (card == "phi_tol")
    {
      ifs >> tol.phi;
    }
    else if (card == "max_inner")
    {
      ifs >> tol.max_inner;
    }
    else if (card == "inner_rel_tol")
    {
      ifs >> tol.inner_rel;
    }
    else if (card == "inner_abs_tol")
    {
      ifs >> tol.inner_abs;
    }
    else if (card == "verbose")
    {
      verb = read_bool(ifs);
    }
    else
    {
      throw Invalid_input_card{card};
    }
  }
}

void Input::echo(ostream& os) const
{
  os << "[CORE]\n";

  os << "mesh_file " << fname_mesh << '\n';
  os << "boundary_conditions";
  for (int i : bc)
    os << ' ' << i;
  os << '\n';
  if (rs > 0) os << "refine_serial " << rs << '\n';
  if (rp > 0) os << "refine_parallel " << rp << '\n';

  os << "xs_file " << fname_xs << '\n';

  if (ref != Analytic::invalid) 
    os << "reference " << Analytic::reference_enum_to_str(ref) << '\n';

  os << "\n[STATE]\n";
  if (verb) os << "verbatim " << verb << '\n';
  os << "max_inner " << tol.max_inner << '\n';
  os << "inner_rel_tol " << tol.inner_rel << '\n';
  os << "inner_abs_tol " << tol.inner_abs << '\n';
  os << "max_outer " << tol.max_outer << '\n';
  os << "k_tol " << tol.k << '\n';
  os << "phi_tol " << tol.phi << '\n';

}

string Input::path_stub(string path)
{
  string stub;
  // assume absolute paths start with ~ or /
  if ((path[0] == '~') || (path[0] == '/')) return stub;
  // return up-to-and-including the final /
  const auto final_slash{path.find_last_of('/')};
  stub = path.substr(0,final_slash+1);
  return stub;
}

void Input::reset_state(ifstream& ifs)
{
  for (int i{0}; i < strlen("[STATE]"); ++i)
    ifs.unget();
}

bool Input::read_bool(ifstream& ifs)
{
  string str;
  ifs >> str;
  for (char& c : str)
    c = tolower(c);
  return ((str == "1") || (str == "t") || (str == "true"));
}

void Input::read_vector(ifstream& ifs, vector<int>& vec)
{
  for (int val; ifs >> val; )
    vec.push_back(val);
  ifs.unget();
  ifs.clear();
}

// copy input data to an arguments data structure
Arguments::Arguments(const Input& inp) : order{1}, 
  verbose{inp.verbose()}, static_cond{false}, pa{false}
{
  // The const_cast is necessary.
  // MFEM argument parsing requires a const char* and the variable is declared
  // as such.
  // strcpy (logically) requires a non-const char*.
  // Therefore, the const is discarded here and retained at MFEM parsing.
  /*
  strcpy(dynamic_cast<char*>(mesh_file), inp.mesh_file().c_str());
  strcpy(const_cast<char*>(device_config), device.c_str());
  */
}
