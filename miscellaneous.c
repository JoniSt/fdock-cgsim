/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * miscellaneous.c
 *
 *  Created on: 2008.09.29.
 *      Author: pechan.imre
 */


#include "miscellaneous.h"

int float2fracint(double toconv, int frac)
//The function converts a float value to a fixed pont fractional number in (32-frac).frac format,
//and returns it as an integer.
{
	if (toconv >= 0)
		return (int) floor(toconv*pow(2, frac));
	else
		return (int) ceil(toconv*pow(2, frac));
}

long long float2fraclint(double toconv, int frac)
//The function converts a float value to a fixed pont fractional number in (32-frac).frac format,
//and returns it as a long integer.
{
	if (toconv >= 0)
		return (long long) floor(toconv*pow(2, frac));
	else
		return (long long) ceil(toconv*pow(2, frac));
}

double timer_gets(void)
//The function returns the current time in seconds.
{
  struct timeval ts;
  double timesec;

  gettimeofday(&ts, (struct timezone*)0);
  timesec = ((double) ts.tv_sec*1000000000.0 + (double) ts.tv_usec*1000.0)/1000000000.0;
  return timesec;
}

double myrand(void)
//The functon returns a random double number between 0 and 1
{
	static int first_call = 0;
	double temprand;

	if (first_call == 0)
	{
		srand((unsigned int) time(NULL));
		first_call++;
	}

	do
#if defined (REPRO)
		temprand = ((double) 1)/((double) RAND_MAX);
#else
		temprand = ((double) rand())/((double) RAND_MAX);
#endif
	while ((temprand == 0.0) || (temprand == 1.0));

	return temprand;
}

unsigned int myrand_int(unsigned int limit)
//The function returns a random integer which is lower than the given limit.
{
	return (unsigned int) floor(limit*myrand());
}

void vec_point2line(const double point [], const double line_pointA [], const double line_pointB [], double vec [])
//The function calculates the vector which moves a point given by the first parameter to its perpendicular projection
//on a line given by to of its points (line_pointA and line_pointB parameters). The result vector is the vec parameter.
{
	double posvec_of_line [3];
	double proj_of_point [3];
	double posvec_of_line_length2, temp;
	int i;

	for (i=0; i<3; i++)
		posvec_of_line [i] = line_pointB [i] - line_pointA [i];		//vector parallel to line

	posvec_of_line_length2 = pow(posvec_of_line [0], 2) + pow(posvec_of_line [1], 2) + pow(posvec_of_line [2], 2);	//length^2 of posvec_of_line

	temp = 0;
	for (i=0; i<3; i++)
		temp += posvec_of_line [i] * (point [i] - line_pointA [i]);
	temp = temp/posvec_of_line_length2;

	for (i=0; i<3; i++)
		proj_of_point [i] = temp * posvec_of_line [i] + line_pointA [i];	//perpendicular projection of point to the line

	for (i=0; i<3; i++)
		vec [i] = proj_of_point [i] - point [i];
}

double angle_of_vectors(const double vector1 [], const double vector2 [])
//The function's inputs are two position vectors (whose starting point is the origo).
//The function returns the angle between them.
{
	int i;
	double len_vec1, len_vec2, scalmul;
	double zerovec [3] = {0, 0, 0};
	double temp;

	scalmul = 0;

	len_vec1 = distance(vector1, zerovec);
	len_vec2 = distance(vector2, zerovec);

	for (i=0; i<3; i++)
		scalmul += vector1 [i]*vector2 [i];

	temp = scalmul/(len_vec1*len_vec2);

	if (temp > 1)
		temp = 1;
	if (temp < -1)
		temp = -1;

	return (acos(temp)*180/M_PI);

}

void vec_crossprod(const double vector1 [], const double vector2 [], double crossprodvec [])
//The function calculates the cross product of position vectors vector1 and vector2, and returns
//it in the third parameter.
{
	crossprodvec [0] = vector1 [1]*vector2 [2] - vector1 [2]*vector2 [1];
	crossprodvec [1] = vector1 [2]*vector2 [0] - vector1 [0]*vector2 [2];
	crossprodvec [2] = vector1 [0]*vector2 [1] - vector1 [1]*vector2 [0];
}

void get_trilininterpol_weights(double weights [][2][2], const double dx, const double dy, const double dz)
//The function calculates the weights for trilinear interpolation based on the location of the point inside
//the cube which is given by the second, third and fourth parameters.
{

	weights [0][0][0] = (1-dx) * (1-dy) * (1-dz);
	weights [1][0][0] = dx     * (1-dy) * (1-dz);
	weights [0][1][0] = (1-dx) * dy     * (1-dz);
	weights [1][1][0] = dx     * dy     * (1-dz);
	weights [0][0][1] = (1-dx) * (1-dy) * dz;
	weights [1][0][1] = dx     * (1-dy) * dz;
	weights [0][1][1] = (1-dx) * dy     * dz;
	weights [1][1][1] = dx     * dy     * dz;
}

int stricmp(const char* str1, const char* str2)
//The function compares the two input strings and returns 0 if they are identical (case-UNsensitive)
//and 1 if not.
{
	const char* c1_poi;
	const char* c2_poi;
	char c1;
	char c2;
	char isdifferent = 0;

	c1_poi = str1;
	c2_poi = str2;

	c1 = *c1_poi;
	c2 = *c2_poi;

	while ((c1 != '\0') && (c2 != '\0'))
	{
		if (toupper(c1) != toupper(c2))
			isdifferent = 1;

		c1_poi++;
		c2_poi++;

		c1 = *c1_poi;
		c2 = *c2_poi;
	}

	if (toupper(c1) != toupper(c2))
		isdifferent = 1;

	return isdifferent;
}

