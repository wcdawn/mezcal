#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <iostream>
#include <fstream>

#include <chrono>
#include <thread>

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

    switch (backend)
    {
      case (SYSINFO):
        sysinfo(&memInfo); 

        total_physical_memory = memInfo.totalram;
        total_physical_memory *= memInfo.mem_unit;

        free_physical_memory = memInfo.freeram;
        free_physical_memory *= memInfo.mem_unit;

        used_physical_memory = total_physical_memory - free_physical_memory;
        break;
      case (MEMINFO):

        unsigned long long buffers{0}, cache{0};

        while (ifs.good())
        {

          std::string card;
          ifs >> card;

          if (card == "MemTotal:")
          {
            ifs >> total_physical_memory;
            total_physical_memory = total_physical_memory << meminfo_shift;
            ifs >> card;
            if (card != "kB")
              throw "error";
          }
          else if (card == "MemFree:")
          {
            ifs >> free_physical_memory;
            free_physical_memory = free_physical_memory << meminfo_shift;
            ifs >> card;
            if (card != "kB")
              throw "error";
          }
          else if (card == "Buffers:")
          {
            ifs >> buffers;
            buffers = buffers << meminfo_shift;
            ifs >> card;
            if (card != "kB")
              throw "error";
          }
          else if ((card == "Cached:") || (card == "SReclaimable:"))
          {
            unsigned long long tmp;
            ifs >> tmp;
            tmp = tmp << meminfo_shift;
            ifs >> card;
            if (card != "kB")
              throw "error";
            cache += tmp;
          }

        }

        ifs.clear();
        ifs.seekg(0);

        used_physical_memory = total_physical_memory - free_physical_memory - buffers - cache;
        break;
    }

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
    if (new_unit != unit) { update(); }
    unit = new_unit;
  }

  void update_and_print(std::ostream& os=std::cout)
  {
    update();
    print(os);
  }

  void init_background(int milliseconds=1)
  {
    std::cout << "BACKGROUND_STARTED" << std::endl;
    t = std::thread{[this,milliseconds] { this->background(milliseconds); }};
    run_background = true;
  }

  void stop_background()
  {
    std::cout << "BACKGROUND_STOPPED" << std::endl;
    run_background = false;
    t.join();
  }

  void background(int milliseconds=1)
  {
    const std::chrono::milliseconds timespan{milliseconds};
    while (run_background)
    {
      std::this_thread::sleep_for(timespan);
      update_and_print();
    }
  }

private:

  std::thread t;
  bool run_background{true};

  void shift_data(int shift)
  {
    total_physical_memory = total_physical_memory >> shift;
    free_physical_memory  = free_physical_memory >> shift;
    used_physical_memory  = used_physical_memory >> shift;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  struct sysinfo memInfo;
#pragma GCC diagnostic pop

  std::ifstream ifs{"/proc/meminfo"};

  unsigned long long total_physical_memory{0};
  unsigned long long free_physical_memory{0};
  unsigned long long used_physical_memory{0};

  Unit_choice unit{byte};

  enum Backend_choice
  {
    SYSINFO,
    MEMINFO
  };

  static constexpr Backend_choice backend{MEMINFO};
  static constexpr int meminfo_shift{10};

};

#endif
