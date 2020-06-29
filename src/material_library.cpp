#include "material_library.hpp"

using std::cerr;

using std::regex;
using std::smatch;
using std::regex_match;

using std::ifstream;
using std::string;

const regex Material_library::block_head_regex_format{R"(\[[A-Z]+\])"};

Material_library::Material_library(const XSLibrary& xslib)
  : xs{xslib}, G{xs.ngroup()}
{
}

void Material_library::read_material(ifstream& ifs)
{
  string card;
  smatch matches; // regex matches
  
  ifs >> card;
  if (card != "[MATERIAL]")
    throw Invalid_material_card{"Expected [MATERIAL]. Got: " + card};

  while (ifs >> card)
  {
    if (card[0] == comment_char)
    {
      // eat until newline
      for (char c{0}; (ifs.get(c)) && (c != '\n'); ) continue;
    }
    else if (regex_match(card, matches, block_head_regex_format))
    {
      reset_block(ifs, matches[0]);
    }
    else
    {
      cerr << "INVALID CARD\n";
      cerr << card << '\n';
      throw Invalid_material_card{"Invalid card: " + card};
    }
  }
}

void Material_library::reset_block(ifstream& ifs, const string& block_head)
{
  for (size_t i{0}; i < block_head.size(); ++i) ifs.unget();
}
