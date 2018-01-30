#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include "pnl/pnl_fft.h"
#include "pnl/pnl_complex.h"

#define MEMORY_ALLOCATION_FAILURE 1
#define uint unsigned int
#define PI M_PI
/*Vectors and matrices*/

static double **V, **P_old, **P_new;
static double **f;
static int **f_down, **f_up;
static double **pu_f, **pd_f;
static dcomplex **F_n_plus_1; /*to store derivative price*/
static dcomplex **F_n; /*to store derivative price*/
static double *ba_log_prices; /*basic asset price line, of length M*/
static double *ba_prices; /*basic asset price line, of length M*/
static double * fftfreqs; /*fft frequencies*/
static dcomplex * f_n_plus_1_k_u, * f_n_k_u; /*to store the upper wh solution precalculations, of length M*/
static dcomplex * f_n_plus_1_k_d, * f_n_k_d; /*to store the upper wh solution precalculations, of length M*/
static dcomplex * phi_plus_array; /*to store factor-functions, of length M*/
static dcomplex * phi_minus_array; /*to store factor-functions, of length M*/
static double * f_n_plus_1_k_u_re, * f_n_plus_1_k_u_im;
static double * f_n_plus_1_k_d_re, *f_n_plus_1_k_d_im;
static dcomplex * f_n_plus_1_k_u_fft_results, * f_n_plus_1_k_d_fft_results;
static double * f_n_plus_1_k_u_fft_results_re, *f_n_plus_1_k_u_fft_results_im;
static double * f_n_plus_1_k_d_fft_results_re, *f_n_plus_1_k_d_fft_results_im;
static dcomplex * f_n_k;

/*Memory allocation*/
static int memory_allocation(uint Nt, uint N, uint M)
{
	uint i;

	/*V is the Nt+1 x Nt+1 matrice, storing volatility values*/
	V = (double**)calloc(Nt + 1, sizeof(double*));
	if (V == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		V[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (V[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	/*pu_f is the Nt+1 x Nt+1 matrice*/
	pu_f = (double**)calloc(Nt + 1, sizeof(double*));
	if (pu_f == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		pu_f[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (pu_f[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	/*pd_f is the Nt+1 x Nt+1 matrice*/
	pd_f = (double**)calloc(Nt + 1, sizeof(double*));
	if (pd_f == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		pd_f[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (pd_f[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	/*is used on the voltree procedure*/
	f = (double**)calloc(Nt + 1, sizeof(double*));
	if (f == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		f[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (f[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	/*f_down is the Nt+1 x Nt+1 matrice*/
	f_down = (int**)calloc(Nt + 1, sizeof(int*));
	if (f_down == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		f_down[i] = (int *)calloc(Nt + 1, sizeof(int));
		if (f_down[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	/*f_up is the Nt+1 x Nt+1 matrice*/
	f_up = (int**)calloc(Nt + 1, sizeof(int*));
	if (f_up == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		f_up[i] = (int *)calloc(Nt + 1, sizeof(int));
		if (f_up[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	/*P_old is the N+1 x N+1 matrice*/
	P_old = (double **)malloc((N + 1)*sizeof(double*));
	for (i = 0; i <= N; i++)
		P_old[i] = (double *)malloc((Nt + 1)*sizeof(double));
	/*P_new is the N+1 x N+1 matrice*/
	P_new = (double **)malloc((N + 1)*sizeof(double*));
	for (i = 0; i <= N; i++)
		P_new[i] = (double *)malloc((Nt + 1)*sizeof(double));

	/*----------------------here are the variables I added----------------------------*/

	F_n_plus_1 = (dcomplex**)calloc(M, sizeof(dcomplex*));
	if (F_n_plus_1 == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}

	for (uint j = 0; j < M; j++) /*for each price grid element we generate a markov chain-resided vector, size Nt+1*/
	{
		F_n_plus_1[j] = (dcomplex *)calloc(Nt + 1, sizeof(dcomplex));
		if (F_n_plus_1[j] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	F_n = (dcomplex**)calloc(M, sizeof(dcomplex*));
	if (F_n == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}


	for (uint j = 0; j < M; j++) /*for each price grid element we generate a markov chain-resided vector, size Nt+1*/
	{
		F_n[j] = (dcomplex *)calloc(Nt + 1, sizeof(dcomplex));
		if (F_n[j] == NULL)
		{
			return MEMORY_ALLOCATION_FAILURE;
		}
	}

	/*a vector of length M to store logarithms of prices divided by barrier log(S/H). Needs an additional procedure to be
	converted to original priceline. The conversion would look like H*exp(s) */
	ba_log_prices = (double *)calloc(M, sizeof(double));
	if (ba_log_prices == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector of length M to store prices*/
	ba_prices = (double *)calloc(M, sizeof(double));
	if (ba_prices == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*to contain frequences of fft of a given number of points with a given step. Used in wh-coefficients calculations*/
	fftfreqs = (double *)calloc(M, sizeof(double));
	if (fftfreqs == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}

	/*a vector to store the upper wh initial condition, of length M, used in the backward induction procedure*/
	f_n_plus_1_k_u = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_plus_1_k_u == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store the lower wh initial condition, of length M, used in the backward induction procedure*/
	f_n_plus_1_k_d = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_plus_1_k_d == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store the upper wh pre-solution, of length M, used in the backward induction procedure*/
	f_n_k_u = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_plus_1_k_u == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store the lower wh pre-solution, of length M, used in the backward induction procedure*/
	f_n_k_d = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_k_d == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store factor-function values, of length M*/
	phi_minus_array = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (phi_minus_array == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store factor-function values, of length M*/
	phi_plus_array = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (phi_plus_array == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store fft real results for upper wh answer, of length M*/
	f_n_plus_1_k_u_re = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_u_re == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store fft imaginary results for upper wh answer, of length M*/
	f_n_plus_1_k_u_im = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_u_im == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store fft real results for lower wh answer, of length M*/
	f_n_plus_1_k_d_re = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_d_re == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*a vector to store fft imaginary results for lower wh answer, of length M*/
	f_n_plus_1_k_d_im = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_d_im == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*, of length M*/
	f_n_plus_1_k_u_fft_results = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_plus_1_k_u_fft_results == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	/*, of length M*/
	f_n_plus_1_k_d_fft_results = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_plus_1_k_d_fft_results == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	f_n_plus_1_k_u_fft_results_re = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_u_fft_results_re == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	f_n_plus_1_k_u_fft_results_im = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_u_fft_results_im == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	f_n_plus_1_k_d_fft_results_re = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_d_fft_results_re == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	f_n_plus_1_k_d_fft_results_im = (double *)calloc(M, sizeof(double));
	if (f_n_plus_1_k_d_fft_results_im == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	f_n_k = (dcomplex *)calloc(M, sizeof(dcomplex));
	if (f_n_k == NULL)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	return OK;
}

static void free_memory(uint Nt, uint N, uint M)
{
	uint i;

	for (i = 0; i<Nt + 1; i++)
		free(V[i]);
	free(V);
	for (i = 0; i<Nt + 1; i++)
		free(pu_f[i]);
	free(pu_f);

	for (i = 0; i<Nt + 1; i++)
		free(pd_f[i]);
	free(pd_f);

	for (i = 0; i<Nt + 1; i++)
		free(f[i]);
	free(f);

	for (i = 0; i<Nt + 1; i++)
		free(f_down[i]);
	free(f_down);

	for (i = 0; i<Nt + 1; i++)
		free(f_up[i]);
	free(f_up);

	for (i = 0; i<N + 1; i++)
		free(P_old[i]);
	free(P_old);

	for (i = 0; i<N + 1; i++)
		free(P_new[i]);
	free(P_new);

	for (uint j = 0; j < M; j++)
	{
		free(F_n_plus_1[j]);
	}
	free(F_n_plus_1);

	for (uint j = 0; j < M; j++)
	{
		free(F_n[j]);
	}
	free(F_n);

	free(ba_log_prices);
	free(ba_prices);
	free(f_n_plus_1_k_d);
	free(f_n_plus_1_k_u);
	free(phi_minus_array);
	free(phi_plus_array);
	free(f_n_k_u);
	free(f_n_k_d);
	free(f_n_plus_1_k_u_re);
	free(f_n_plus_1_k_u_im);
	free(f_n_plus_1_k_d_re);
	free(f_n_plus_1_k_d_im);
	free(f_n_plus_1_k_u_fft_results);
	free(f_n_plus_1_k_d_fft_results);
	free(f_n_plus_1_k_u_fft_results_re);
	free(f_n_plus_1_k_u_fft_results_im);
	free(f_n_plus_1_k_d_fft_results_re);
	free(f_n_plus_1_k_d_fft_results_im);
	free(f_n_k);
	free(fftfreqs);
	return;
}

static double compute_f(double r, double omega)
{
	return 2.*sqrt(r) / omega;
}

static double compute_v(double R, double omega)
{
	double val;

	val = SQR(R)*SQR(omega) / 4.;
	if (R>0.)
		val = SQR(R)*SQR(omega) / 4.;
	else
		val = 0.0;
	return val;
}

static int tree_v(double tt, double v0, double kappa, double theta, double omega, int Nt)
{
	int i, j;
	int z; /*a variable for k_u or k_d, to add to k on n+1 step*/
	double Ru, Rd; /*stores k_u(n,k) and k_d(n,k), respectively*/
	double mu_r, v_curr;
	double dt, sqrt_dt;

	/*Fixed tree for R=f*/
	f[0][0] = compute_f(v0, omega);

	dt = tt / (double)Nt;
	sqrt_dt = sqrt(dt);

	V[0][0] = compute_v(f[0][0], omega);
	f[1][0] = f[0][0] - sqrt_dt;
	f[1][1] = f[0][0] + sqrt_dt;
	V[1][0] = compute_v(f[1][0], omega);
	V[1][1] = compute_v(f[1][1], omega);
	for (i = 1; i<Nt; i++)
		for (j = 0; j <= i; j++)
		{
			f[i + 1][j] = f[i][j] - sqrt_dt;
			f[i + 1][j + 1] = f[i][j] + sqrt_dt;
			V[i + 1][j] = compute_v(f[i + 1][j], omega);
			V[i + 1][j + 1] = compute_v(f[i + 1][j + 1], omega);
		}

	/*Evolve tree for f*/
	for (i = 0; i<Nt; i++)
	{
		printf("Making voltree. Layer: %d of %d\n", i, Nt - 1);
		for (j = 0; j <= i; j++)
		{
			/*Compute mu_f*/
			v_curr = V[i][j];

			mu_r = kappa*(theta - v_curr);

			z = 0;
			while ((V[i][j] + mu_r*dt<V[i + 1][j - z])
				&& (j - z >= 0)) {
				z = z + 1;
			}
			f_down[i][j] = -z;
			Rd = V[i + 1][j - z];

			z = 0;
			while ((V[i][j] + mu_r*dt>V[i + 1][j + z])
				&& (j + z <= i))
			{
				z = z + 1;
			}
			f_up[i][j] = z;
			Ru = V[i + 1][j + z];


			pu_f[i][j] = (V[i][j] + mu_r*dt - Rd) / (Ru - Rd);

			if ((Ru - 1.e-9>V[i + 1][i + 1]) || (j + f_up[i][j]>i + 1))
			{
				pu_f[i][j] = 1;

				f_up[i][j] = i + 1 - j;
				f_down[i][j] = i - j;
			}

			if ((Rd + 1.e-9<V[i + 1][0]) || (j + f_down[i][j]<0))
			{
				pu_f[i][j] = 0.;
				f_up[i][j] = 1 - j;
				f_down[i][j] = 0 - j;
			}
			pd_f[i][j] = 1. - pu_f[i][j];
		}
	}

	return 1;
}

static int fftfreq(uint M, double d)
{
	int n = int(M);
	double val = 1.0 / (n * d);
	int middle = ((n - 1)/2) + 1;
	int i,k;
	for (k = 0; k<middle; k++)
	{
		fftfreqs[k] = k;
	}
	double * p2 = (double *)calloc(n/2, sizeof(double));
	for (i = 0; i< n/2; i++)
	{
		p2[i] = -n/2 + i;
	}
	for (k = 0; k<n/2; k++)
	{
		fftfreqs[middle + k] = p2[k];
	}

	for (k = 0; k<n; k++)
	{
		fftfreqs[k] = val * fftfreqs[k];
	}
	free(p2);
	return 0;
}

static double G(double S, double K)
{
	return MAX(0, K - S);
}

static int compute_price(double tt, double H, double K, double r_premia, double v0, double kappa, double theta, double sigma, double rho,
	double L, int M, int Nt )
{
	/*Variables*/
	int j, n, k;
	double r; /*continuous rate*/
	double min_log_price, max_log_price;
	double ds, dt; /*price and time discretization steps*/
	double rho_hat; /*parameter after substitution*/
	double q, factor, discount_factor; /*pde parameters*/
	double treshold = 1e-9; /* when we assume probability to be zero and switch to a different equation*/

	int k_d, k_u; /*n+1 vertice numbers, depending on [n][k]*/
	double sigma_local, gamma; /*wh factors parameters*/
	double beta_minus, beta_plus; /*wh-factors coefficients*/
	double local_barrier; /*a barrier depending on [n][k], to check crossing on each step*/

	//if (2.0 * kappa * theta < pow(sigma, 2))
	//	return 1; /*Novikov condition not satisfied, probability values could be incorrect*/
	/*Body*/
	r = log(1 + r_premia / 100);

	/*building voltree*/
	tree_v(tt, v0, kappa, theta, sigma, Nt);

	/*spacial variable. Price space construction*/
	min_log_price = L*log(0.5) - (rho / sigma)* V[Nt][Nt];
	max_log_price = L*log(2);
	ds = (max_log_price - min_log_price) / double(M);

	for (j = 0; j < M; j++)
	{
		ba_log_prices[j] = min_log_price + j*ds;
		ba_prices[j] = H*exp(ba_log_prices[j] + (rho / sigma)* V[0][0]);
	}
	dt = tt / double(Nt);

	/*fft frequences we'll need in every vertice of a tree*/
	fftfreq(M, ds);
	rho_hat = sqrt(1.0 - pow(rho, 2.0));
	q = 1.0 / dt + r;
	factor = pow(q*dt, -1.0);
	//discount_factor = exp(r*dt);
	discount_factor = r - rho / sigma * kappa * theta;

	/*filling F_next matrice by initial (in time T) conditions*/
	for (j = 0; j < M; j++)
		for (k = 0; k < Nt + 1; k++)
		{
			F_n_plus_1[j][k] = Complex(G(H*exp(ba_log_prices[j] + (rho / sigma)* V[Nt][k]), K), 0);
		}

	/*here the main cycle starts - the backward induction procedure*/
	for (n = Nt - 1; n >= 0; n--)
	{
		printf("Processing: %d of %d\n", n, Nt-1);
		for (k = 0; k <= n; k++)
		{
			/*to calculate the binomial expectation we should use matrices from the tree method.
			After (n,k) vertice one could either get to (n+1,k_u) or (n+1, k_d). The numbers k_u and k_d could be
			read from f_up and f_down matrices, by the rule of addition, for example:

			f_down[i][j] = -z;
			Rd = V[i + 1][j - z]
			f_up[i][j] = z;
			Ru = V[i + 1][j + z];
			*/
			k_u = k + f_up[n][k];
			k_d = k + f_down[n][k];
			local_barrier = - (rho / sigma) * V[n][k];

			/*initial conditions of a step*/
			for (j = 0; j < M; j++)
			{
				//f_n_plus_1_k_u[j] = F[j][n+1][k_u];
				//f_n_plus_1_k_d[j] = F[j][n+1][k_d];
				f_n_plus_1_k_u[j] = F_n_plus_1[j][k_u];
				f_n_plus_1_k_d[j] = F_n_plus_1[j][k_d];
			}
			/*applying indicator function*/
			for (j = 0; j < M; j++)
			{
				if (ba_log_prices[j] < local_barrier)
				{
					f_n_plus_1_k_u[j].r = 0.0;
					f_n_plus_1_k_u[j].i = 0.0;
					f_n_plus_1_k_d[j].r = 0.0;
					f_n_plus_1_k_d[j].i = 0.0;
				}
			}
			if (V[n][k] >= treshold)
			{
				/*set up variance - dependent parameters for a given step*/
				sigma_local = rho_hat * sqrt(V[n][k]);
				gamma = r - 0.5 * V[n][k] - rho / sigma * kappa * (theta - V[n][k]);  /*also local*/
				/* beta_plus and beta_minus, for factors*/
				beta_minus = -(gamma + sqrt(pow(gamma,2) + 2 * pow(sigma_local,2) * q)) / pow(sigma_local,2);
				beta_plus = -(gamma - sqrt(pow(gamma,2) + 2 * pow(sigma_local,2) * q)) / pow(sigma_local,2);

				for (j = 0; j < M; j++)
				{
					/* factor functions
					phi_plus_array = ([beta_plus / (beta_plus - i * 2 * pi*xi) for xi in xi_space])
					phi_minus_array = ([-beta_minus / (-beta_minus + i * 2 * pi*xi) for xi in xi_space]) */
					phi_plus_array[j] = RCdiv(beta_plus, RCsub(beta_plus, RCmul((2.0 * PI * fftfreqs[j]), CI)));
					phi_minus_array[j] = RCdiv(-beta_minus, RCadd(-beta_minus, RCmul((2.0 * PI * fftfreqs[j]), CI)));
				}

				/*factorization calculation*/

				/*f_n_k_u = factor * fft.ifft(phi_minus_array *	fft.fft(
				indicator(original_prices_array, 0) * fft.ifft(phi_plus_array * fft.fft(f_n_plus_1_k_u))))*/
				for (int j = 0; j < M; j++)
				{
					f_n_plus_1_k_u_re[j] = f_n_plus_1_k_u[j].r;
					f_n_plus_1_k_u_im[j] = f_n_plus_1_k_u[j].i;

				}
				pnl_fft2(f_n_plus_1_k_u_re, f_n_plus_1_k_u_im, M);
				for (j = 0; j < M; j++) {
					/*putting complex and imaginary part together again*/
					f_n_plus_1_k_u_fft_results[j] = Complex(f_n_plus_1_k_u_re[j], f_n_plus_1_k_u_im[j]);
					/*multiplying by phi_plus*/
					f_n_plus_1_k_u_fft_results[j] = Cmul(phi_plus_array[j], f_n_plus_1_k_u_fft_results[j]);
					/*extracting imaginary and complex parts to use in further fft*/
					f_n_plus_1_k_u_fft_results_re[j] = f_n_plus_1_k_u_fft_results[j].r;
					f_n_plus_1_k_u_fft_results_im[j] = f_n_plus_1_k_u_fft_results[j].i;

				}

				pnl_ifft2(f_n_plus_1_k_u_fft_results_re, f_n_plus_1_k_u_fft_results_im, M);
				/*applying indicator function, after ifft*/
				for (j = 0; j < M; j++)
				{
					if (ba_log_prices[j] < local_barrier)
					{
						f_n_plus_1_k_u_fft_results_re[j] = 0.0;
						f_n_plus_1_k_u_fft_results_im[j] = 0.0;
					}
				}

				/*performing second fft */
				pnl_fft2(f_n_plus_1_k_u_fft_results_re, f_n_plus_1_k_u_fft_results_im, M);

				for (j = 0; j < M; j++) {
					/*putting complex and imaginary part together again*/
					f_n_plus_1_k_u_fft_results[j] = Complex(f_n_plus_1_k_u_fft_results_re[j], f_n_plus_1_k_u_fft_results_im[j]);
					/*multiplying by phi_minus*/
					f_n_plus_1_k_u_fft_results[j] = Cmul(phi_minus_array[j], f_n_plus_1_k_u_fft_results[j]);
					/*extracting imaginary and complex parts to use in further fft*/
					f_n_plus_1_k_u_fft_results_re[j] = f_n_plus_1_k_u_fft_results[j].r;
					f_n_plus_1_k_u_fft_results_im[j] = f_n_plus_1_k_u_fft_results[j].i;
				}

				/*the very last ifft*/
				pnl_ifft2(f_n_plus_1_k_u_fft_results_re, f_n_plus_1_k_u_fft_results_im, M);
				/*multiplying by factor*/
				for (j = 0; j < M; j++) {
					f_n_k_u[j].r = factor * f_n_plus_1_k_u_fft_results_re[j];
					f_n_k_u[j].i = factor * f_n_plus_1_k_u_fft_results_im[j];
				}

				/*f_n_k_d = factor * fft.ifft(phi_minus_array * fft.fft(
				indicator(original_prices_array, 0) * fft.ifft(phi_plus_array * fft.fft(f_n_plus_1_k_d))))*/
				for (int j = 0; j < M; j++)
				{
					f_n_plus_1_k_d_re[j] = f_n_plus_1_k_d[j].r;
					f_n_plus_1_k_d_im[j] = f_n_plus_1_k_d[j].i;

				}
				pnl_fft2(f_n_plus_1_k_d_re, f_n_plus_1_k_d_im, M);
				for (j = 0; j < M; j++) {
					/*putting complex and imaginary part together again*/
					f_n_plus_1_k_d_fft_results[j] = Complex(f_n_plus_1_k_d_re[j], f_n_plus_1_k_d_im[j]);
					/*multiplying by phi_plus*/
					f_n_plus_1_k_d_fft_results[j] = Cmul(phi_plus_array[j], f_n_plus_1_k_d_fft_results[j]);
					/*extracting imaginary and complex parts to use in further fft*/
					f_n_plus_1_k_d_fft_results_re[j] = f_n_plus_1_k_d_fft_results[j].r;
					f_n_plus_1_k_d_fft_results_im[j] = f_n_plus_1_k_d_fft_results[j].i;
				}
				pnl_ifft2(f_n_plus_1_k_d_fft_results_re, f_n_plus_1_k_d_fft_results_im, M);
				/*applying indicator function, after ifft*/
				for (j = 0; j < M; j++)
				{
					if (ba_log_prices[j] < local_barrier)
					{
						f_n_plus_1_k_d_fft_results_re[j] = 0.0;
						f_n_plus_1_k_d_fft_results_im[j] = 0.0;
					}
				}
				/*performing second fft */
				pnl_fft2(f_n_plus_1_k_d_fft_results_re, f_n_plus_1_k_d_fft_results_im, M);

				for (j = 0; j < M; j++) {
					/*putting complex and imaginary part together again*/
					f_n_plus_1_k_d_fft_results[j] = Complex(f_n_plus_1_k_d_fft_results_re[j], f_n_plus_1_k_d_fft_results_im[j]);
					/*multiplying by phi_minus*/
					f_n_plus_1_k_d_fft_results[j] = Cmul(phi_minus_array[j], f_n_plus_1_k_d_fft_results[j]);
					/*extracting imaginary and complex parts to use in further fft*/
					f_n_plus_1_k_d_fft_results_re[j] = f_n_plus_1_k_d_fft_results[j].r;
					f_n_plus_1_k_d_fft_results_im[j] = f_n_plus_1_k_d_fft_results[j].i;
				}
				/*the very last ifft*/
				pnl_ifft2(f_n_plus_1_k_d_fft_results_re, f_n_plus_1_k_d_fft_results_im, M);
				/*multiplying by factor*/
				for (j = 0; j < M; j++) {
					f_n_k_d[j].r = factor * f_n_plus_1_k_d_fft_results_re[j];
					f_n_k_d[j].i = factor * f_n_plus_1_k_d_fft_results_im[j];
				}
			}
			else if (V[n][k] < treshold)
			{
				/*applying indicator function*/
				for (j = 0; j < M; j++)
				{
					if (ba_log_prices[j] < local_barrier)
					{
						f_n_plus_1_k_u[j].r = 0.0;
						f_n_plus_1_k_u[j].i = 0.0;
						f_n_plus_1_k_d[j].r = 0.0;
						f_n_plus_1_k_d[j].i = 0.0;
					}
				}
				for (j = 0; j < M; j++)
				{
					//f_n_plus_1_k_u[j] = F[j][n + 1][k_u];
					f_n_plus_1_k_u[j] = F_n_plus_1[j][k_u];
					f_n_k_u[j] = CRsub(f_n_plus_1_k_u[j], discount_factor * dt);
					f_n_k_d[j] = f_n_k_u[j];

				}
			}
			/*
            f_n_k = pd_f[n, k] * f_n_k_d + pu_f[n, k] * f_n_k_u
			*/
			for (j = 0; j < M; j++)
			{
				f_n_k[j] = Cadd(RCmul(pd_f[n][k], f_n_k_d[j]), RCmul(pu_f[n][k], f_n_k_u[j]));
				F_n[j][k] = f_n_k[j];
			}
		}
		for (j = 0; j < M; j++)
		{
			for (int state = 0; state < Nt; state++)
			{
				F_n_plus_1[j][state] = F_n[j][state];
				F_n[j][state] = Complex(0,0);
			}
		}
	}
	/*Preprocessing F before showing*/
	for (j = 0; j < M; j++)
	{
		if (ba_prices[j] <= H)
		{
			F_n_plus_1[j][0].r = 0;
		}
		if (F_n_plus_1[j][0].r < 0.)
		{
			F_n_plus_1[j][0].r = 0;
		}
	}
	return OK;
}

static int find_nearest_left_price_position(double price, uint M)
{
	double left_price = -DBL_MAX;
	uint i = 0;

	while ((ba_prices[i] <= price) && (i < M))
	{
		left_price = ba_prices[i];
		i++;
	}
	return i-1;
}
static int find_nearest_right_price_position(double price, uint M)
{
	double right_price = DBL_MAX;
	uint i = M - 1;
	if(price <= 0)
	{
		return -1;
	}
	else
	{
		while ((ba_prices[i]) >= price && (i >= 0))
		{
			right_price = ba_prices[i];
			i--;
		}
		return i+1;
	}

}
static double quadratic_interpolation(double spot_price, uint M)
{

	//Price, quadratic interpolation
	int pos = find_nearest_left_price_position(spot_price, M);
	int i = pos;
	double Sl = ba_prices[i-1];
	double Sm = ba_prices[i];
	double Sr = ba_prices[i+1];

	// S0 is between Sm and Sr
	double pricel = F_n_plus_1[i - 1][0].r;
	double pricem = F_n_plus_1[i][0].r;
	double pricer = F_n_plus_1[i + 1][0].r;

	//quadratic interpolation
	double A = pricel;
	double B = (pricem - pricel) / (Sm - Sl);
	double C = (pricer - A - B*(Sr - Sl)) / (Sr - Sl) / (Sr - Sm);

	//Price
	return A + B*(spot_price - Sl) + C*(spot_price - Sl)*(spot_price - Sm);
}

int main()
{
	/*Option parameters*/
	double tt = 1;
	double H = 90.0;
	double K = 100.0;
	double r_premia = 10;
	double spot = 95.0;
	double spot_step = 5;
	uint spot_iterations = 21;

	/*Heston model parameters*/
	double v0 = 0.1; /* initial volatility */
	double kappa = 2.0; /*Heston parameter, mean reversion*/
	double theta = 0.2; /*Heston parameter, long-run variance*/
	double sigma = 0.2; /*Heston parameter, volatility of variance*/
	double omega = sigma; /*sigma is used everywhere, omega - in the variance tree*/
	double rho = 0.5; /*Heston parameter, correlation*/

	/*method parameters*/
	uint Nt = 100; /*number of time steps*/
	uint M = pow(2,12);/*space grid. should be a power of 2*/
	double L = 3; /*scaling coefficient*/

	int allocation = memory_allocation(Nt, M, M);
	if (allocation == MEMORY_ALLOCATION_FAILURE)
	{
		return MEMORY_ALLOCATION_FAILURE;
	}
	else
	{
		double start_time = clock() / double(CLOCKS_PER_SEC);
		compute_price(tt, H, K, r_premia, v0, kappa, theta, sigma, rho, L, M, Nt);
		double end_time = clock() / double(CLOCKS_PER_SEC);
		for (int j = find_nearest_right_price_position(1.5*K,M); j >= find_nearest_left_price_position(H, M); j--)
		{
			printf("ba_price %f Price %f + %f i\n", ba_prices[j], F_n_plus_1[j][0].r, F_n_plus_1[j][0].i);
		}
		for (uint i = 0; i < spot_iterations; i++)
		{
			printf("interpolated ba_price %f Price %f\n", spot + i*spot_step, quadratic_interpolation(spot + i*spot_step, M));
		}
		printf("Time elapsed (in seconds): %f\n", end_time - start_time);

		free_memory(Nt, M, M);
		getchar();
		return OK;
	}
}
