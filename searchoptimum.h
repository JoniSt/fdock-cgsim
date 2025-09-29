/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * searchoptimum.h
 *
 *  Created on: 2008.11.18.
 *      Author: pechan.imre
 */

#ifndef SEARCHOPTIMUM_H_
#define SEARCHOPTIMUM_H_

extern long first_better, first_worser, second_better, second_worser;

#include <stdio.h>

#include "defines.h"
#include "processligand.h"
#include "processgrid.h"
#include "getparameters.h"
#include "miscellaneous.h"

void map_angle(double*, double);

void find_best(double [][40], const int, int*);

void gen_new_genotype(double [], double [], const double, const double, const double, const double, const int, double [], double [], int);

void perform_LS(double [], const Liganddata*, const double, const double, const double, const int, const int, const int, int*,
				const Gridinfo*, const double*, int, const double, const double, const double, int);

void binary_tournament_selection(double [][40], const int, int*, int*, double, int);

//void genetic_steady_state(double [][40], const Liganddata*, const Gridinfo*, const double*, const Dockpars*, int, int);

void genetic_generational(double [][40], const Liganddata*, const Gridinfo*, const double*, Dockpars*, int, int);

#endif /* SEARCHOPTIMUM_H_ */
