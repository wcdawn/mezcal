#ifndef TIMER
#define TIMER

#include <chrono>
#include <iostream>

#include <string>
#include <vector>

// TODO this currently only tracks a single thread, using wall-clock time.
// may want to accumulate accross threads.
// could also generate statistics regarding load-balancing.

class Timer
{
  public:
    using clock = std::chrono::system_clock;
    using point = std::chrono::time_point<clock>;
    using duration = std::chrono::duration<clock::rep, clock::period>;

    Timer() = default;

    void start(const std::string& time_name);
    void restart(const std::string& time_name);
    void stop(const std::string& time_name);
    void stop();

    void Print(std::ostream& os);

  private:
    const point init_clock{clock::now()};

    class Time_record
    {
      public:
        Time_record() = default;
        void set_name(const std::string& str) { name_time = str; }
        std::string name() const { return name_time; }
        duration elapsed() const { return elapsed_time; }

        void start();
        void stop();

      private:
        bool running{true};
        std::string name_time;
        point start_time;
        duration elapsed_time;
    };

    std::vector<Time_record> times;

    std::vector<Time_record>::iterator find_record(const std::string& time_name);

};

#endif
