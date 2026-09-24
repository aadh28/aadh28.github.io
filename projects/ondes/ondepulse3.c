#include <stdlib.h>
#include <stdio.h>
#include <math.h>

///////////////////////////////////////////////////////////////////////

double const T0 = 30.0;
double const u0 = 0.1;
double const L = 19;
double const temps = 1;
int const h = 2;
double const pi = 3.14159265358979323846;

#define nb 100
#define dx (L/nb)
#define dt (temps/nb)
#define v sqrt(T0/u0)
#define f (v/2*L)

typedef double vect [nb];

///////////////////////////////////////////////////////////////////////

void ecriture(double *X,double *T);
void ecriture2(vect *Y);
void echantillonage(double *x,double *t);
void initialisation(vect *y,double *x,double *t);
void pulse(vect *y,double cfl);

///////////////////////////////////////////////////////////////////////

int main(void)
{
	int i,n;
	double cfl = (v*dt/dx);
	double *x,*t;
	vect y[nb];
	
	x=(double*)malloc(nb*sizeof(double));
	t=(double*)malloc(nb*sizeof(double));
	
	echantillonage(x,t);
	initialisation(y,x,t);
	ecriture(x,t);
	
	/*printf("\n\n\n");
	for(n=0;n<nb;n++)
	{
		for(i=0;i<nb;i++)
		{
			printf("%.3lf\t",y[n][i]);
		}
		printf("\n");
	}*/
	
	printf("\n\n");
	
	pulse(y,cfl);
	ecriture2(y);
	
	for(i=0;i<nb;i++)
	{
		printf("%lf\t%lf\n",x[i],t[i]);
	}
	
	printf("\n\n v = %lf\n dt = %lf\n dx = %lf\n CFL : vdt/dx = %lf\n\n\n",v,dt,dx,v*dt/dx);
	
	free(x);
	free(t);

	return 0;
}

void echantillonage(double *x,double *t)
{
	int i;
	
	x[0] = 0;
	t[0] = 0;
	
	for (i=1;i<nb;i++)
	{
		x[i] = x[i-1] + dx;
	}
	
	for (i=1;i<nb;i++)
	{
		t[i] = t[i-1] + dt;
	}
}

void initialisation(vect *y,double *x,double *t)
{
	int i,n;
	
	for(n=0;n<nb;n++)
	{
		for (i=0;i<nb;i++)
		{	if (t[n] < temps/4 && x[i] > L*3/4)
			{
				y[0][i] = sin(h*pi*x[i]/(L/4));
				y[1][i] = sin(h*pi*x[i]/(L/4));
			}
			else
			{
				y[n][0] = 0;
				y[n][i] = 0;
			}
		}
		y[n][nb-1] = 0;
	}              
}

void pulse(vect *y,double cfl)
{
	int i,n;
	
	for(n=1;n<nb;n++)
	{
		for(i=1;i<nb-1;i++)
		{
			y[n+1][i] = pow(cfl,2)*(y[n][i+1] + y[n][i-1]) + 2*y[n][i]*(1 - pow(cfl,2)) - y[n-1][i];
			//y[n+1][i] = 2*y[n][i] - y[n-1][i] + pow(cfl,2)*(y[n][i+1] - 2*y[n][i] + y[n][i-1]);
		}
	}
	
	/*for(n=0;n<nb;n++)
	{
		for(i=0;i<nb;i++)
		{
			printf("%.3lf\t",y[n][i]);
		}
		printf("\n");
	}
	
	printf("\n\n");*/
}

void ecriture(double *X,double *T)
{
	FILE *onde;
	int i,n;
	
	onde = fopen("onde.txt","a");
	
	if (onde == NULL)
	{
		printf("\nErreur souscis fichier\n\n");
		exit(1);
	}
	
	for(i=0;i<nb;i++)
	{
		fprintf(onde,"%lf\t%lf\n",X[i],T[i]);
	}


	fclose(onde);
}

void ecriture2(vect *Y)
{
	FILE *onde2;
	int i,n;
	
	onde2 = fopen("onde2.txt","a");
	
	if (onde2 == NULL)
	{
		printf("\nErreur souscis fichier\n\n");
		exit(1);
	}
	
	for(n=0;n<nb;n++)
	{
		for(i=0;i<nb;i++)
		{
			fprintf(onde2,"%lf\t",Y[n][i]);
		}
		fprintf(onde2,"\n");
	}

	fclose(onde2);
}