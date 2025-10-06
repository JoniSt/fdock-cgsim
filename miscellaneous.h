/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * miscallenous.h
 *
 *  Created on: 2008.09.29.
 *      Author: pechan.imre
 */

#ifndef MISCELLANEOUS_H_
#define MISCELLANEOUS_H_

#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

#include "miscellaneous_inline.h"

#ifdef __cplusplus
extern "C" {
#endif


int float2fracint(double, int);

long long float2fraclint(double, int);

double timer_gets(void);

double myrand(void);

unsigned int myrand_int(unsigned int);

void vec_point2line(const double [], const double [], const double [], double []);

double angle_of_vectors(const double [], const double []);

void vec_crossprod(const double [], const double [], double []);

void get_trilininterpol_weights(double [][2][2], const double, const double, const double);

int stricmp(const char*, const char*);

#ifdef __cplusplus
}
#endif

#endif /* MISCELLANEOUS_H_ */
