
#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer
{
public:

	Timer(std::string timerName) { name = timerName; callCount = 0; totalElapsed = 0.f; }
	~Timer() {
	};
	void Start()
	{
		start = std::chrono::steady_clock::now();
	}
	void Stop()
	{
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<float, std::milli> duration = end - start;
		elapsed = duration.count();
		callCount++;
		totalElapsed += elapsed;
	}
	float GetElapsed() 
	{
		return elapsed;
	}
	float GetAverageElapsed() 
	{
		return totalElapsed/callCount;
	}
	float GetTotalElapsed()
	{
		return totalElapsed;
	}

private:

	std::chrono::steady_clock::time_point start;
	float elapsed;
	float totalElapsed;
	std::string name;
	uint32_t callCount;
};

