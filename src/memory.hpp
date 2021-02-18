#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <iostream>

#include "sys/types.h"
#include "sys/sysinfo.h" 

class Memory_stats
{

public:
  
  enum Unit_choice
  {
    gigabyte,
    megabyte,
    kilobyte,
    byte
  };

  Memory_stats() { update(); }
  Memory_stats(Unit_choice uc) : unit{uc} { update(); }

  void update()
  { 
    sysinfo(&memInfo); 

    total_physical_memory = memInfo.totalram;
    total_physical_memory *= memInfo.mem_unit;

    free_physical_memory = memInfo.freeram;
    free_physical_memory *= memInfo.mem_unit;

    used_physical_memory = total_physical_memory - free_physical_memory;

    switch (unit)
    {
      case (gigabyte):
        shift_data(30);
        break;
      case (megabyte):
        shift_data(20);
        break;
      case (kilobyte):
        shift_data(10);
        break;
      case (byte):
        break;
    }

  }

  void print(std::ostream& os=std::cout) const
  {
    os << "total=" << total_physical_memory 
       << " free=" << free_physical_memory 
       << " used=" << used_physical_memory 
       << std::endl;
  }

  void set_unit(Unit_choice new_unit)
  {
    if (new_unit != unit)
    {
      unit = new_unit;
      update();
    }
    unit = new_unit;
  }

private:

  void shift_data(int shift)
  {
    total_physical_memory = total_physical_memory >> shift;
    free_physical_memory  = free_physical_memory >> shift;
    used_physical_memory  = used_physical_memory >> shift;
  }

  struct sysinfo memInfo;

  unsigned long long total_physical_memory{0};
  unsigned long long free_physical_memory{0};
  unsigned long long used_physical_memory{0};

  Unit_choice unit{byte};

};

#endif
