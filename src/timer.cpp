#include "timer.hpp"

#include <algorithm>
#include <iomanip>

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
  times.push_back(Time_record());
  times.back().set_name(time_name);
  times.back().start();
}

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

void Timer::stop(const string& time_name)
{
  auto this_time_record{find_record(time_name)};
  if (this_time_record != times.end()) this_time_record->stop();
}

void Timer::stop()
{
  times.back().stop();
}

void Timer::Print(ostream& os)
{

  const auto final_clock{clock::now()};
  const auto total_elapsed{final_clock - init_clock};

  // finalize times
  for (Time_record& tr : times) tr.stop();

  os << "=== Timing Summary [s] ===\n";
  for (const Time_record& time : times)
    os << left << setw(20) << time.name() << right << setw(5)
       << duration_cast<milliseconds>(time.elapsed()).count() / 1000.0 << " ("
       << setprecision(4)
       << static_cast<double>(time.elapsed().count()) / total_elapsed.count()
          * 100.
       << "%)\n";
  os << left << setw(20) << "Total" << right << setw(5)
     << duration_cast<milliseconds>(clock::now() - init_clock).count() / 1000.
     << '\n';
}

std::vector<Timer::Time_record>::iterator
Timer::find_record(const string& time_name)
{
  return find_if(
  times.begin(), times.end(),
  [&time_name](const Time_record& tr) { return tr.name() == time_name; });
}
