#include "version.hpp"
#include <iostream>

void version(std::ostream& os)
{
  const std::string logo =
 "  __  __ ______ ___________          _              ;\"--,   / \\   ,--\"; \n"
 " |  \\/  |  ____|___  / ____|   /\\   | |       ,'\"--,;    ' /   \\ '    ;,--\"', \n"
 " | \\  / | |__     / / |       /  \\  | |       \",    '\"--, /     \\ ,--\"'    ,\" \n"
 " | |\\/| |  __|   / /| |      / /\\ \\ | |        '\"-,      |       |      ,-\"' \n"
 " | |  | | |____ / /_| |____ / ____ \\| |____        '\"--, \\       / ,--\"' \n"
 " |_|  |_|______/_____\\_____/_/    \\_\\______|      __..:::::::::::::::..__ \n";

  os << logo;
  os << "         Microreactor eZ CALculation" << "\n";
  os << "\n";

  os << "git version   = " << Version::git_version   << "\n";
  os << "git sha       = " << Version::git_sha       << "\n";
  os << "compiled date = " << Version::compiled_date << "\n";
  os << "compiled on   = " << Version::compiled_host << "\n";
}
