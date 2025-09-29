/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * processgrid.h
 *
 *  Created on: 2008.09.30.
 *      Author: pechan.imre
 */

#ifndef PROCESSGRID_H_
#define PROCESSGRID_H_


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

//Added by L30nardoSV (04/2023)
//Getting full path of the grid file
#include <libgen.h>

#include "miscellaneous.h"

#define getvalue_4Darr(mempoi, grinf, t, z, y, x) *(mempoi+(grinf).size_xyz [0]*(y+(grinf).size_xyz [1]*(z+(grinf).size_xyz [2]*t))+x)
//The macro helps to access the grid point values which were read from external files with get_gridvalues function.
//The first parameter is a pointer which points to the memory area storing the data. The second one is the corresponding
//grid info (parameter of get_gridinfo function). The other parameters are the type index, z, y and x coordinates of the grid
//point.

typedef struct
//Struct which can contain all the important informations which derives from .gpf and .xyz files.
{
	//Added by L30nardoSV (04/2023)
	//Getting full path of the grid file
	char* grid_file_path;

	char receptor_name [64];
	int size_xyz [3];
	//int truncated_size_xyz [3];
	double spacing;
	double size_xyz_angstr [3];
	char grid_types [16][3];
	int num_of_atypes;
	double origo_real_xyz [3];
} Gridinfo;

int eldes2fracint(double , double);

int get_gridinfo(const char*, Gridinfo*);

int get_gridvalues(const Gridinfo*, double**);

#endif /* PROCESSGRID_H_ */
