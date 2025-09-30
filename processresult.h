/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * processresult.h
 *
 *  Created on: 2008.09.22.
 *      Author: pechan.imre
 */

#ifndef PROCESSRESULT_H_
#define PROCESSRESULT_H_

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "defines.h"
#include "processligand.h"
#include "getparameters.h"

#define PRINT1000(file, x) fprintf(file,  ((fabs((x)) >= 0.0) && ((fabs(x)) <= 1000.)) ? "%+7.2f" : "%+11.2e" , (x));

typedef struct
{
	Liganddata reslig_realcoord;
	double interE;
	double interE_elec;
	double intraE;
	double peratom_vdw [256];
	double peratom_elec [256];
	double rmsd_from_ref;
	double rmsd_from_cluscent;
	int clus_id;
	int clus_subrank;
	int run_number;
	double runtime;

} Ligandresult;

void arrange_result(double [][40], const int);

void write_basic_info(FILE*, const Liganddata*, const Dockpars*, const Gridinfo*, const int*, char**);

void write_basic_info_dlg(FILE*, const Liganddata*, const Dockpars*, const Gridinfo*, const int*, char**);

void make_resfiles(double [][40], const Liganddata*, const Liganddata*,
				   const Dockpars*, const Gridinfo*, const double*, const int*, char**, int, int, Ligandresult*);

//void cluster_analysis(Ligandresult [], int, char*, const Liganddata*, const Dockpars*, const Gridinfo*, const int*,
//					  char**, const double, const double);

void clusanal_gendlg(Ligandresult [], int, const Liganddata*, const Dockpars*, const Gridinfo*, const int*, char**);

#ifdef __cplusplus
}
#endif

#endif /* PROCESSRESULT_H_ */
