#include "PID.h"
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double _Kp, double _Ki, double _Kd) {
	K.resize(3);
	K[0] = _Kp;
	K[1] = _Ki;
	K[2] = _Kd;

	sum_cte = 0.0;
	pre_cte = 0.0;

	error = 0.0;
	dp = {0.01, 0.00002, 0.2};
	n_twiddle = 0;
	best_err = 1.0e6;
	twiddle_initialized = false;
	skip_once = false;
}

void PID::Reset() {
	sum_cte = 0.0;
	pre_cte = 0.0;
	error = 0.0;
	time_steps = 1;
}

void PID::UpdateError(double cte) {
	sum_cte += cte;

	p_error = K[0] * cte;
	d_error = K[2]* (cte - pre_cte);
	i_error = K[1] * sum_cte;

	// std::cout << "p_error: " << p_error << ", d_error: " << d_error << ", i_error: " << i_error << std::endl;

	pre_cte = cte;
	time_steps++;

	// skip the first 200 steps in Twiddle error accumulation
	if(time_steps > 200) {
		error += cte*cte;
	}
}

double PID::TotalError() {
	return  -(p_error + d_error + i_error);
}

void PID::Twiddle() {
	int idx = 0;
	if(twiddle_initialized == false) {
		best_err = error;
		K[idx] += dp[idx];
		twiddle_initialized = true;
		std::cout << "0 twiddle: " << n_twiddle << ", error=" << error << ", Kp=" << K[idx] << std::endl;
	} else {
		std::cout << "1 twiddle: " << n_twiddle << ", error=" << error << ", Kp=" << K[idx] << std::endl;
		if(error < best_err && skip_once != true) {
			best_err = error;
			dp[idx] *= 1.1;
			K[idx] += dp[idx];
			std::cout << "Smaller error found: " << "Kp=" << K[idx] << ", dp=" << dp[idx] << std::endl;
		} else {
			if(skip_once == false) {
				K[idx] -= 2 * dp[idx];
				skip_once = true;
				std::cout << "Reduce kp by 2 dp: " << "Kp=" << K[idx] << ", dp=" << dp[idx] << std::endl;
			} else {
				skip_once = false;

				if(error < best_err) {
					best_err = error;
					dp[idx] *= 1.1;
					K[idx] += dp[idx];
					std::cout << "Smaller error found2: " << "Kp=" << K[idx] << ", dp=" << dp[idx] << std::endl;
				} else {
					dp[idx] *= 0.9;
					K[idx] += dp[idx];
					std::cout << "Shrink dp by 0.9 times: " << "Kp=" << K[idx] << ", dp=" << dp[idx] << std::endl;
				}
			}
		}
	}

	n_twiddle++;
}

