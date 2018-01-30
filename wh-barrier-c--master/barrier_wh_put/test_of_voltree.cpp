#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define SQR(X) (X*X)
#define MAX(A,B) ( (A) > (B) ? (A):(B))
#define MIN(A,B) ( (A) < (B) ? (A):(B))
#define MEMORY_ALLOCATION_FAILURE 0
#define OK 1

#ifdef VOLATREE_TEST
/*Vettori*/
static double **V, **P_old, **P_new;
static double **y, **f;
static int **f_down, **f_up;
static int **y_down, **y_up;
static double **pu_y, **pd_y;
static double **pu_f, **pd_f;

/*Memory Allocation*/
static int memory_allocation(int Nt, int N)
{
	int i;

	V = (double**)calloc(Nt + 1, sizeof(double*));
	if (V == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		V[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (V[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	pu_y = (double**)calloc(Nt + 1, sizeof(double*));
	if (pu_y == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		pu_y[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (pu_y[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	pd_y = (double**)calloc(Nt + 1, sizeof(double*));
	if (pd_y == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		pd_y[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (pd_y[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	pu_f = (double**)calloc(Nt + 1, sizeof(double*));
	if (pu_f == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		pu_f[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (pu_f[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	pd_f = (double**)calloc(Nt + 1, sizeof(double*));
	if (pd_f == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		pd_f[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (pd_f[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	y = (double**)calloc(Nt + 1, sizeof(double*));
	if (y == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		y[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (y[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	f = (double**)calloc(Nt + 1, sizeof(double*));
	if (f == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		f[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (f[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	f_down = (int**)calloc(Nt + 1, sizeof(int*));
	if (f_down == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		f_down[i] = (int *)calloc(Nt + 1, sizeof(int));
		if (f_down[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	f_up = (int**)calloc(Nt + 1, sizeof(int*));
	if (f_up == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		f_up[i] = (int *)calloc(Nt + 1, sizeof(int));
		if (f_up[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	y_down = (int**)calloc(Nt + 1, sizeof(int*));
	if (y_down == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		y_down[i] = (int *)calloc(Nt + 1, sizeof(int));
		if (y_down[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	y_up = (int**)calloc(Nt + 1, sizeof(int*));
	if (y_up == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (i = 0; i<Nt + 1; i++)
	{
		y_up[i] = (int *)calloc(Nt + 1, sizeof(int));
		if (y_up[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}

	P_old = (double **)malloc((N + 1)*sizeof(double*));
	for (i = 0; i <= N; i++)
		P_old[i] = (double *)malloc((Nt + 1)*sizeof(double));

	P_new = (double **)malloc((N + 1)*sizeof(double*));
	for (i = 0; i <= N; i++)
		P_new[i] = (double *)malloc((Nt + 1)*sizeof(double));

	return OK;
}


static void free_memory(int Nt, int N)
{
	int i;

	for (i = 0; i<Nt + 1; i++)
		free(V[i]);
	free(V);

	for (i = 0; i<Nt + 1; i++)
		free(pu_y[i]);
	free(pu_y);

	for (i = 0; i<Nt + 1; i++)
		free(pd_y[i]);
	free(pd_y);

	for (i = 0; i<Nt + 1; i++)
		free(y[i]);
	free(y);

	for (i = 0; i<Nt + 1; i++)
		free(y_up[i]);
	free(y_up);

	for (i = 0; i<Nt + 1; i++)
		free(y_down[i]);
	free(y_down);

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
		free(f_up[i]);
	free(f_up);

	for (i = 0; i<Nt + 1; i++)
		free(f_down[i]);
	free(f_down);

	for (i = 0; i<N + 1; i++)
		free(P_old[i]);
	free(P_old);

	for (i = 0; i<N + 1; i++)
		free(P_new[i]);
	free(P_new);

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

static double compute_S(double Y, double rv, double omega, double rho)
{
	double val;

	val = exp(Y)*exp(rho*rv / omega);

	return val;
}

/*Calibration of the tree  the stochastic volatilty v*/
static int tree_v(double tt //время до экспирации
	, double v0, double kappa, double theta, double omega, int Nt)
{
	int i, j;
	int z;
	double Ru, Rd;
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

			Ru = V[i + 1][j + z];

			f_up[i][j] = z;
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

/*Compute Price Option*/
static int compute_price(int am, double tt, double K, double s0, double r_fisso, double divid, double v0, double kappa, double theta, double omega, double rho, int Nt, int N, double *price, double *delta)
{
	int dummy;
	int i, j, k;
	double puu, pud, pdu, pdd, stock;
	int fv_up, fv_down;
	double l;
	double alpha, beta, gamma, alpha1, beta1, gamma1;
	double dx;
	double log_s0;
	double discount;
	double bound1, bound2;
	double z, vv;
	double soglia;
	double sigma;
	double dt, sqrt_dt;
	double  *A, *B, *C, *A1, *B1, *C1, *Price, *S, *vect_y;
	double precision_fd;
	int Index, PriceIndex;
	double a, b, c, a1, b1, c1;

	A = (double *)malloc((N + 1)*sizeof(double));
	B = (double *)malloc((N + 1)*sizeof(double));
	C = (double *)malloc((N + 1)*sizeof(double));
	A1 = (double *)malloc((N + 1)*sizeof(double));
	B1 = (double *)malloc((N + 1)*sizeof(double));
	C1 = (double *)malloc((N + 1)*sizeof(double));

	vect_y = (double *)malloc((N + 1)*sizeof(double));
	Price = (double *)malloc((N + 1)*sizeof(double));
	S = (double *)malloc((N + 1)*sizeof(double));

	dt = tt / (double)Nt;
	sqrt_dt = sqrt(dt);

	sigma = 2.*sqrt(MAX(v0, theta));
	precision_fd = 1.0e-7; /*Precision for the localization of FD methods*/
	if (tt>20.) //getting rid of too big periods T
		tt = 20;
	l = sigma*sqrt(tt)*sqrt(log(1.0 / precision_fd)) + fabs((r_fisso - divid - 0.5*sigma)*tt);

	dx = 2.0*l / (double)N;
	log_s0 = (log(s0) - rho / omega*V[0][0]);

	for (j = 0; j <= N; j++)
		vect_y[j] = log_s0 - l + (double)j*dx; //у dynamics description in terms of S0 and l

											   /*Maturity conditions*/
	for (k = 0; k <= Nt; k++) //volatility indices
	{
		for (j = 0; j <= N; j++) //price indices
		{
			stock = compute_S(vect_y[j], V[Nt][k], omega, rho);
			P_old[j][k] = MAX(0., K - stock);
		}
	}
	k,
		/*Rhs Factors*/
		alpha1 = 0.;
	beta1 = 1.;
	gamma1 = 0.;

	for (PriceIndex = 1; PriceIndex <= N - 1; PriceIndex++) //strange vectors filling
	{
		A1[PriceIndex] = alpha1;
		B1[PriceIndex] = beta1;
		C1[PriceIndex] = gamma1;
	}

	discount = exp(-r_fisso*dt);

	/*Dynamic Programming*/
	for (i = Nt - 1; i >= 0; i--)
	{
		for (k = 0; k <= i; k++)
		{
			z = (r_fisso - divid - 0.5*V[i][k] - rho*kappa*(theta - V[i][k]) / omega);
			vv = 0.5*V[i][k] * (1. - SQR(rho));

			fv_up = f_up[i][k];
			fv_down = f_down[i][k];

			bound1 = 0.;
			bound2 = MAX(compute_S(vect_y[N], V[i][k], omega, rho)*exp(-divid*i*dt) - K*exp(-r_fisso*i*dt), 0);
			bound2 = 0;

			//bound1=K*exp(-r_fisso*i*dt)-compute_S(vect_y[N],V[i][k],omega,rho)*exp(-(divid)*i*dt);
			//bound2=0;

			soglia = 0.00001;

			if (V[i][k]<soglia)//Explicit Scheme
			{

				//Up_wind Scheme Coefficients
				a = 0.;
				b = 1 - z*dt / dx;
				c = z*dt / dx;

				a1 = -z*dt / dx;
				b1 = 1 + z*dt / dx;
				c1 = 0.;

				//F_U
				if (z >= 0)//Up_wind Scheme
				{
					P_old[N][k + fv_up] = bound2;
					P_old[0][k + fv_up] = bound1;
					for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
						Price[PriceIndex] = a*P_old[PriceIndex - 1][k + fv_up] + b*P_old[PriceIndex][k + fv_up] + c*P_old[PriceIndex + 1][k + fv_up];
				}
				else
				{
					P_old[0][k + fv_up] = bound1;
					P_old[N][k + fv_up] = bound2;

					for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
						Price[PriceIndex] = a1*P_old[PriceIndex - 1][k + fv_up] + b1*P_old[PriceIndex][k + fv_up] + c1*P_old[PriceIndex + 1][k + fv_up];
				}

				for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
					P_new[PriceIndex][k] = discount*pu_f[i][k] * Price[PriceIndex];


				//F_D
				if (z >= 0)//Up_wind Scheme
				{

					P_old[N][k + fv_down] = bound2;
					P_old[0][k + fv_down] = bound1;
					for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
						Price[PriceIndex] = a*P_old[PriceIndex - 1][k + fv_down] + b*P_old[PriceIndex][k + fv_down] + c*P_old[PriceIndex + 1][k + fv_down];
				}
				else
				{
					P_old[0][k + fv_down] = bound1;
					P_old[N][k + fv_down] = bound2;

					for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
						Price[PriceIndex] = a1*P_old[PriceIndex - 1][k + fv_down] + b1*P_old[PriceIndex][k + fv_down] + c1*P_old[PriceIndex + 1][k + fv_down];
				}


				for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
					P_new[PriceIndex][k] += discount*(1 - pu_f[i][k])*Price[PriceIndex];

			}
			//Fully Implicit
			else {

				/*Lhs Factor of the fully implicit scheme*/
				alpha = -vv*dt / SQR(dx) + z*dt / (2.*dx);
				beta = 1 + vv * 2 * dt / SQR(dx);
				gamma = -vv*dt / SQR(dx) - z*dt / (2 * dx);

				for (PriceIndex = 1; PriceIndex <= N - 1; PriceIndex++)
				{
					A[PriceIndex] = alpha;
					B[PriceIndex] = beta;
					C[PriceIndex] = gamma;
				}

				B[1] = beta + alpha;
				B[N - 1] = beta + gamma;

				B1[1] = beta1 + alpha1;
				B1[N - 1] = beta1 + gamma1;

				/*Set Gauss*/
				for (PriceIndex = N - 2; PriceIndex >= 1; PriceIndex--)
					B[PriceIndex] = B[PriceIndex] - C[PriceIndex] * A[PriceIndex + 1] / B[PriceIndex + 1];
				for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
					A[PriceIndex] = A[PriceIndex] / B[PriceIndex];
				for (PriceIndex = 1; PriceIndex<N - 1; PriceIndex++)
					C[PriceIndex] = C[PriceIndex] / B[PriceIndex + 1];


				//F_U

				//Initialise
				for (PriceIndex = 1; PriceIndex<N; PriceIndex++) {
					Price[PriceIndex] = P_old[PriceIndex][k + fv_up];
				}

				/*Set Rhs*/
				S[1] = B1[1] * Price[1] + C1[1] * Price[2] + A1[1] * bound1 - alpha*bound1;
				for (PriceIndex = 2; PriceIndex<N - 1; PriceIndex++)
					S[PriceIndex] = A1[PriceIndex] * Price[PriceIndex - 1] +
					B1[PriceIndex] * Price[PriceIndex] +
					C1[PriceIndex] * Price[PriceIndex + 1];
				S[N - 1] = A1[N - 1] * Price[N - 2] + B1[N - 1] * Price[N - 1] + C1[N - 1] * bound2 - gamma*bound2;

				/*Solve the system*/
				for (PriceIndex = N - 2; PriceIndex >= 1; PriceIndex--)
					S[PriceIndex] = S[PriceIndex] - C[PriceIndex] * S[PriceIndex + 1];

				Price[1] = S[1] / B[1];

				for (PriceIndex = 2; PriceIndex<N; PriceIndex++)
					Price[PriceIndex] = S[PriceIndex] / B[PriceIndex] - A[PriceIndex] * Price[PriceIndex - 1];

				for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
				{
					P_new[PriceIndex][k] = discount*pu_f[i][k] * Price[PriceIndex];
				}

				//F_D

				//Initialise
				for (PriceIndex = 1; PriceIndex<N; PriceIndex++) {
					Price[PriceIndex] = P_old[PriceIndex][k + fv_down];
				}

				/*Set Rhs*/
				S[1] = B1[1] * Price[1] + C1[1] * Price[2] + A1[1] * bound1 - alpha*bound1;
				for (PriceIndex = 2; PriceIndex<N - 1; PriceIndex++)
					S[PriceIndex] = A1[PriceIndex] * Price[PriceIndex - 1] +
					B1[PriceIndex] * Price[PriceIndex] +
					C1[PriceIndex] * Price[PriceIndex + 1];
				S[N - 1] = A1[N - 1] * Price[N - 2] + B1[N - 1] * Price[N - 1] + C1[N - 1] * bound2 - gamma*bound2;


				/*Solve the system*/
				for (PriceIndex = N - 2; PriceIndex >= 1; PriceIndex--)
					S[PriceIndex] = S[PriceIndex] - C[PriceIndex] * S[PriceIndex + 1];

				Price[1] = S[1] / B[1];
				for (PriceIndex = 2; PriceIndex<N; PriceIndex++)
					Price[PriceIndex] = S[PriceIndex] / B[PriceIndex] - A[PriceIndex] * Price[PriceIndex - 1];

				for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
				{
					P_new[PriceIndex][k] += discount*pd_f[i][k] * Price[PriceIndex];
				}
			}//end fully-implicit else

			if (am)
				for (PriceIndex = 1; PriceIndex<N; PriceIndex++)
					P_new[PriceIndex][k] = MAX(P_new[PriceIndex][k], K - compute_S(vect_y[PriceIndex], V[i][k], omega, rho));
		}//end k

		 //Copy
		for (j = 0; j <= N; j++)
			for (k = 0; k <= i; k++)
				P_old[j][k] = P_new[j][k];
	}//end i

	Index = (int)floor((double)N / 2.0);

	/*Price*/
	*price = P_new[Index][0];

	/*Delta*/
	*delta = (P_new[Index + 1][0] - P_new[Index - 1][0]) / (2.0*s0*dx);

	printf("Price %.6f  %.6f\n", *price, *delta);

	/*Memory Disallocation*/
	free_memory(Nt, N);

	free(A);
	free(B);
	free(C);
	free(A1);
	free(B1);
	free(C1);
	free(vect_y);
	free(S);
	free(Price);

	return 1;
}

static double compute_bond_price(int Nt, double tt, double r_fisso)
{
	int k_u;
	int k_d;
	double ** bond_price;
	bond_price = (double**)calloc(Nt + 1, sizeof(double*));
	if (bond_price == NULL)
		return MEMORY_ALLOCATION_FAILURE;
	for (int i = 0; i<Nt + 1; i++)
	{
		bond_price[i] = (double *)calloc(Nt + 1, sizeof(double));
		if (bond_price[i] == NULL)
			return MEMORY_ALLOCATION_FAILURE;
	}
	double dt = Nt / tt;
	double discount_factor = exp(-dt * r_fisso);
	for (int i = 0; i < Nt + 1; i++)
		for (int j = 0; j < Nt + 1; j++)
		{
			if (i < Nt)
			{
				bond_price[i][j] = 0;
			}
			else if (i = Nt)
			{
				bond_price[i][j] = exp(V[i][j]);
			}
		}
	for (int i = Nt - 1; i >= 0; i--)
	{
		for (int k = 0; k <= i; k++)
		{
			k_u = k + f_up[i][k];
			k_d = k + f_down[i][k];
			bond_price[i][k] = discount_factor * (pu_f[i][k] * bond_price[i + 1][k_u] + pd_f[i][k] * bond_price[i + 1][k_d]);
		}
	}
	return bond_price[0][0];
}
/*Compute Price of Bond*/

int main()
{
	int dummy;
	double r_fisso, v0, kappa, omega, theta, rho, tt, s0, K, divid;
	int am, N, Nt;
	double price, delta, r_bond, price_bond;

	//Parameters Heston
	r_fisso = log(1.1);
	v0 = 1;
	kappa = 2;
	theta = 0.2;
	rho = 0.5;
	tt = 1.;
	s0 = 100;
	K = 100.;
	divid = 0.;
	omega = 0.2;//vol of vol

	am = 1;

	Nt = 500;
	N = 200;

	/*Memory Allocation*/
	dummy = memory_allocation(Nt, N);

	//Tree construction for v
	dummy = tree_v(tt, v0, kappa, theta, omega, Nt);

	//Finite Difference Tree Algorithm
	//dummy=compute_price(am,tt,K,s0,r_fisso,divid,v0,kappa,theta,omega,rho,Nt,N,&price,&delta);
	r_bond = compute_bond_price(Nt, tt, v0);
	price_bond = exp((v0 - r_bond)*tt);
	printf("rate of CIR bond is %6f, price is %6f", r_bond, price_bond);
	getchar();
	return OK;
}
#endif