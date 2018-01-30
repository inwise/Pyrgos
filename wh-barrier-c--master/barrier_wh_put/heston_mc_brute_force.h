#pragma once
#define uint unsigned int
#include <cmath>
#include <random>
#include <time.h>

double payoff(double S, double K)
{
	return MAX(0, K - S);
}
double generate_heston_trajectory_return(double T, double S0, double H, double K, double r_premia,
	double V0, double kappa, double theta, double sigma, double rho, uint N)
{
	/*simulates Heston monte-carlo for Down-and-out put directly through equations*/
	double r = log(r_premia / 100.0 + 1.0);
	double dt = double(T) / double(N);
	double sqrt_dt = sqrt(dt);
	// trajectory started
	// initials
	double S_t = S0;
	double V_t = V0;
	double t = 0.0;

	double random_value_for_V;
	double random_value_for_S;
	double dZ_V, dZ_S, dV_t, dS_t;
	// Seed with a real random value, if available
	std::random_device rdev;
	int mean = 0;
	// Generate a normal distribution around that mean
	std::seed_seq seed2{ rdev(), rdev(), rdev(), rdev(), rdev(), rdev(), rdev(), rdev()};
	//mersenne twister engine
	std::mt19937 mersenne_engine(seed2);
	std::normal_distribution<> normal_dist(mean, 1);

	while(t <= T)
	{
		// Seed with a real random value, if available
		// random walk for V
		random_value_for_V = normal_dist(mersenne_engine);
		dZ_V = random_value_for_V * sqrt_dt;
		// random walk for S + correlation
		random_value_for_S = normal_dist(mersenne_engine);
		random_value_for_S = rho * random_value_for_V + sqrt(1 - pow(rho, 2.0)) * random_value_for_S;
		dZ_S = random_value_for_S * sqrt_dt;

		// equation for V
		dV_t = kappa * (theta - V_t) * dt + sigma * sqrt(V_t) * sqrt_dt * dZ_V;
		V_t += dV_t;
		// equation for S
		dS_t = S_t * r * dt + S_t * sqrt(V_t) * dZ_S;
		S_t += dS_t;
		// trajectory ended
		t += dt;
		// check barrier crossing on each step
		if (S_t <= H)
		{
			return 0.0;
		}	
	}
	return payoff(S_t, K);
}