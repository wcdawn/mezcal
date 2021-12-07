#ifndef TIMER
#define TIMER

#include <chrono>
#include <iostream>
#include <string>
#include <map>

// TODO this currently only tracks a single thread, using wall-clock time.
// may want to accumulate accross threads.
// could also generate statistics regarding load-balancing.

class Timer
{
public:
  using clock    = std::chrono::system_clock;
  using point    = std::chrono::time_point<clock>;
  using duration = std::chrono::duration<clock::rep, clock::period>;

  Timer() = default;

  void start(const std::string& time_name);
  void stop(const std::string& time_name);
  void stop();
  double elapsed(); // return elapsed time in seconds

  void Print(std::ostream& os = std::cout);

  const std::string matrix_assembly{"Matrix assembly"};
  const std::string solver{"Solver"};

private:
  const point init_clock{clock::now()};

  std::string most_recent;

  class Time_record
  {
  public:
    Time_record() = default;
    duration elapsed() const { return elapsed_time; }
    void start();
    void stop();

  private:
    bool running{true};
    point start_time;
    duration elapsed_time;
  };

  std::map<std::string,Time_record> timers;

};

#endif
