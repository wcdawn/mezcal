#include "timer.hpp"

#include <algorithm>
#include <iomanip>
#include <vector>

void Timer::Time_record::stop()
{
  if (running)
  {
    elapsed_time += clock::now() - start_time;
    running = false;
  }
}

void Timer::Time_record::start()
{
  running    = true;
  start_time = clock::now();
}

void Timer::start(const std::string& time_name)
{
  timers[time_name].start();
  most_recent = time_name;
}

/*
void Timer::restart(const string& time_name)
{
  auto this_time_record{find_record(time_name)};
  if (this_time_record != times.end())
  {
    // cleanup any old timers
    this_time_record->stop();
    // start the timer again
    this_time_record->start();
  }
}
*/

void Timer::stop(const std::string& time_name)
{
  timers[time_name].stop();
}

void Timer::stop()
{
  timers[most_recent].stop();
}

void Timer::Print(std::ostream& os)
{

  const auto final_clock{clock::now()};
  const auto total_elapsed{final_clock - init_clock};

  // finalize times and figure out maximum length
  std::size_t max_name_length{0};
  for (auto& tr : timers)
  {
    tr.second.stop();
    max_name_length = std::max(max_name_length, tr.first.length());
  }
  max_name_length++;

  os << "=== Timing Summary [s] ===\n";
  for (const auto& tr : timers)
  {
    const std::string& name{tr.first};
    const Time_record& time{tr.second};
    os << std::left << std::setw(max_name_length) << name << std::right << std::setw(5)
       << std::chrono::duration_cast<std::chrono::milliseconds>(time.elapsed()).count() / 1000.0 << " ("
       << std::setprecision(4)
       << static_cast<double>(time.elapsed().count()) / total_elapsed.count()
          * 100.
       << "%)\n";
  }

  os << std::left << std::setw(20) << "Total" << std::right << std::setw(5)
     << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - init_clock).count() / 1000.
     << '\n';
}

/*
std::vector<Timer::Time_record>::iterator
Timer::find_record(const std::string& time_name)
{
  return std::find_if(
  times.begin(), times.end(),
  [&time_name](const Time_record& tr) { return tr.name() == time_name; });
}
*/
