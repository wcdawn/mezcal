#include "hdf5_access_wrapper.hpp"

void Hdf5_access_wrapper::change_group(const std::string& new_group_name)
{
  dset.reset();
  grp = std::make_unique<H5::Group>(H5::Group{file.openGroup(new_group_name)});
}

void Hdf5_access_wrapper::change_dataset(const std::string& new_dset_name)
{
  dset = std::make_unique<H5::DataSet>(grp->openDataSet(new_dset_name));
}

std::string Hdf5_access_wrapper::read_string() const
{
  std::string val;
  if (dset) dset->read(val, dset->getStrType(), dset->getSpace());
  return val;
}

int Hdf5_access_wrapper::read_int() const
{
  int val{};
  if (dset) dset->read(&val, dset->getIntType(), dset->getSpace());
  return val;
}

void Hdf5_access_wrapper::read(void* data) const
{
  dset->read(data, dset->getDataType(), dset->getSpace());
}

void Hdf5_access_wrapper::read(void* data, const H5::DataType& dtype) const
{
  dset->read(data, dtype, dset->getSpace());
}
