
#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer
{
public:
	//constructor (with name of whats being tracked)

	Timer(std::string timerName) { name = timerName; callCount = 0; totalElapsed = 50.f; }
	~Timer() {
	};
	void Start()
	{
		start = std::chrono::steady_clock::now();
	}
	void Stop()
	{
		auto end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		callCount++;
		totalElapsed += elapsed.count() /1000.f;
	}
	float GetElapsed() 
	{
		return elapsed.count() / 1000.f;
	}
	float GetAverageElapsed() 
	{
		return totalElapsed/callCount;
	}
	float GetTotalElapsed()
	{
		return fmod(totalElapsed,200.f)/200.f;
	}

private:

	std::chrono::steady_clock::time_point start;
	std::chrono::milliseconds elapsed;
	float totalElapsed;
	std::string name;
	uint32_t callCount;
};

