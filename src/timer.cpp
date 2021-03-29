#include "timer.hpp"

#include <algorithm>
#include <iomanip>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::milliseconds;

using std::ostream;

using std::left;
using std::right;
using std::setprecision;
using std::setw;

using std::string;
using std::vector;

using std::find_if;

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

void Timer::start(const string& time_name)
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

void Timer::stop(const string& time_name)
{
  timers[time_name].stop();
}

void Timer::stop()
{
  timers[most_recent].stop();
}

void Timer::Print(ostream& os)
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
    const string& name{tr.first};
    const Time_record& time{tr.second};
    os << left << setw(max_name_length) << name << right << setw(5)
       << duration_cast<milliseconds>(time.elapsed()).count() / 1000.0 << " ("
       << setprecision(4)
       << static_cast<double>(time.elapsed().count()) / total_elapsed.count()
          * 100.
       << "%)\n";
  }

  os << left << setw(20) << "Total" << right << setw(5)
     << duration_cast<milliseconds>(clock::now() - init_clock).count() / 1000.
     << '\n';
}

/*
std::vector<Timer::Time_record>::iterator
Timer::find_record(const string& time_name)
{
  return find_if(
  times.begin(), times.end(),
  [&time_name](const Time_record& tr) { return tr.name() == time_name; });
}
*/
