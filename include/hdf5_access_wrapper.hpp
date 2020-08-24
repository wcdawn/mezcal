#ifndef HDF5_ACCESS_WRAPPER
#define HDF5_ACCESS_WRAPPER

#include "H5Cpp.h"
#include "make_unique.hpp"

#include <memory>

class Hdf5_access_wrapper
{
public:
  Hdf5_access_wrapper(const H5::H5File& associated_file) : file{associated_file}
  {
  }

  void change_group(const std::string& new_group_name);
  void change_dataset(const std::string& new_dset_name);

  H5::Group group() const { return *grp; }
  H5::DataSet dataset() const { return *dset; }

  std::string read_string() const;
  int read_int() const;
  void read(void* data) const;
  void read(void* data, const H5::DataType& dtype) const;

private:
  const H5::H5File& file;
  std::unique_ptr<H5::Group> grp{nullptr};
  std::unique_ptr<H5::DataSet> dset{nullptr};
};

#endif
