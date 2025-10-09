#pragma once

#include <limits.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
//Struct which describes a quaternion.
{
	double q;
	double x;
	double y;
	double z;
} Quaternion;

//macro that calculates the trilinear interpolation, the first parameter is a 2*2*2 array of the values of the function
//in the vertices of the cube, and the second one is a 2*2*2 array of the interpolation weights
#define trilin_interpol(cube, weights) (cube[0][0][0]*weights[0][0][0] + cube[1][0][0]*weights[1][0][0] +	\
									    cube[0][1][0]*weights[0][1][0] + cube[1][1][0]*weights[1][1][0] +	\
									    cube[0][0][1]*weights[0][0][1] + cube[1][0][1]*weights[1][0][1] +	\
									    cube[0][1][1]*weights[0][1][1] + cube[1][1][1]*weights[1][1][1])

static inline double trilin_interpol_inline(const double cube [2][2][2], const double weights [2][2][2]) {
    return (cube[0][0][0]*weights[0][0][0] + cube[1][0][0]*weights[1][0][0] +
            cube[0][1][0]*weights[0][1][0] + cube[1][1][0]*weights[1][1][0] +
            cube[0][0][1]*weights[0][0][1] + cube[1][0][1]*weights[1][0][1] +
            cube[0][1][1]*weights[0][1][1] + cube[1][1][1]*weights[1][1][1]);
}

static inline double distance(const double point1 [], const double point2 [])
//Returns the distance between point1 and point2 (the arrays have to store the x, y and z coordinates of the
//point, respectively.
{
	double sub1, sub2, sub3;

	sub1 = point1 [0] - point2 [0];
	sub2 = point1 [1] - point2 [1];
	sub3 = point1 [2] - point2 [2];

	return sqrt(sub1*sub1 + sub2*sub2 + sub3*sub3);
}


static inline void rotate(double point [], const double movvec [], const double normvec [], const double* angle)
//The function rotates the point given by the first parameter around an axis which is parallel to vector normvec and which
//can be moved to the origo with vector movvec. The direction of rotation with angle is considered relative to normvec
//according to right hand rule. If debug is 1, debug messages will be printed to the screen.
{

	Quaternion quatrot_left, quatrot_right, quatrot_temp;
	double anglediv2, cos_anglediv2, sin_anglediv2;


	//the point must be moved according to moving vector
	point [0] = point [0] - movvec [0];
	point [1] = point [1] - movvec [1];
	point [2] = point [2] - movvec [2];

	//Related equations:
	//q = quater_w+i*quater_x+j*quater_y+k*quater_z
	//v = i*point_x+j*point_y+k*point_z
	//The coordinates of the rotated point can be calculated as:
	//q*v*(q^-1), where
	//q^-1 = quater_w-i*quater_x-j*quater_y-k*quater_z
	//and * is the quaternion multiplication defined as follows:
	//(a1+i*b1+j*c1+k*d1)*(a2+i*b2+j*c2+k*d2) = (a1a2-b1b2-c1c2-d1d2)+
	//										  	i*(a1b2+a2b1+c1d2-c2d1)+
	//											j*(a1c2+a2c1+b2d1-b1d2)+
	//											k*(a1d2+a2d1+b1c2-b2c1)
	//

	anglediv2 = (*angle)/2/180*M_PI;
	cos_anglediv2 = cos(anglediv2);
	sin_anglediv2 = sin(anglediv2);

	quatrot_left.q = cos_anglediv2;				//rotation quaternion
	quatrot_left.x = sin_anglediv2*normvec [0];
	quatrot_left.y = sin_anglediv2*normvec [1];
	quatrot_left.z = sin_anglediv2*normvec [2];


	quatrot_right.q = quatrot_left.q;					//inverse of rotation quaternion
	quatrot_right.x = -1*quatrot_left.x;
	quatrot_right.y = -1*quatrot_left.y;
	quatrot_right.z = -1*quatrot_left.z;

	//Quaternion multiplications
	//Since the q field of v is 0 as well as the result's q element, simplifications can be made...

	quatrot_temp.q = 0 - quatrot_left.x*point [0] - quatrot_left.y*point [1] - quatrot_left.z*point [2];
	quatrot_temp.x = quatrot_left.q*point [0] + 0 + quatrot_left.y*point [2] - quatrot_left.z*point [1];
	quatrot_temp.y = quatrot_left.q*point [1] - quatrot_left.x*point [2] + 0 + quatrot_left.z*point [0];
	quatrot_temp.z = quatrot_left.q*point [2] + quatrot_left.x*point [1] - quatrot_left.y*point [0] + 0;

	point [0] = quatrot_temp.q*quatrot_right.x + quatrot_temp.x*quatrot_right.q + quatrot_temp.y*quatrot_right.z - quatrot_temp.z*quatrot_right.y;
	point [1] = quatrot_temp.q*quatrot_right.y - quatrot_temp.x*quatrot_right.z + quatrot_temp.y*quatrot_right.q + quatrot_temp.z*quatrot_right.x;
	point [2] = quatrot_temp.q*quatrot_right.z + quatrot_temp.x*quatrot_right.y - quatrot_temp.y*quatrot_right.x + quatrot_temp.z*quatrot_right.q;

	//Moving the point back

	point [0] = point [0] + movvec [0];
	point [1] = point [1] + movvec [1];
	point [2] = point [2] + movvec [2];
}

static inline void rotate_precomputed_sincos(double point [], const double movvec [], const double normvec [], const double* sincos)
//The function rotates the point given by the first parameter around an axis which is parallel to vector normvec and which
//can be moved to the origo with vector movvec. The direction of rotation with angle is considered relative to normvec
//according to right hand rule. If debug is 1, debug messages will be printed to the screen.
{

	Quaternion quatrot_left, quatrot_right, quatrot_temp;
	double cos_anglediv2, sin_anglediv2;


	//the point must be moved according to moving vector
	point [0] = point [0] - movvec [0];
	point [1] = point [1] - movvec [1];
	point [2] = point [2] - movvec [2];

	//Related equations:
	//q = quater_w+i*quater_x+j*quater_y+k*quater_z
	//v = i*point_x+j*point_y+k*point_z
	//The coordinates of the rotated point can be calculated as:
	//q*v*(q^-1), where
	//q^-1 = quater_w-i*quater_x-j*quater_y-k*quater_z
	//and * is the quaternion multiplication defined as follows:
	//(a1+i*b1+j*c1+k*d1)*(a2+i*b2+j*c2+k*d2) = (a1a2-b1b2-c1c2-d1d2)+
	//										  	i*(a1b2+a2b1+c1d2-c2d1)+
	//											j*(a1c2+a2c1+b2d1-b1d2)+
	//											k*(a1d2+a2d1+b1c2-b2c1)
	//

	sin_anglediv2 = sincos[0];
	cos_anglediv2 = sincos[1];

	quatrot_left.q = cos_anglediv2;				//rotation quaternion
	quatrot_left.x = sin_anglediv2*normvec [0];
	quatrot_left.y = sin_anglediv2*normvec [1];
	quatrot_left.z = sin_anglediv2*normvec [2];


	quatrot_right.q = quatrot_left.q;					//inverse of rotation quaternion
	quatrot_right.x = -1*quatrot_left.x;
	quatrot_right.y = -1*quatrot_left.y;
	quatrot_right.z = -1*quatrot_left.z;

	//Quaternion multiplications
	//Since the q field of v is 0 as well as the result's q element, simplifications can be made...

	quatrot_temp.q = 0 - quatrot_left.x*point [0] - quatrot_left.y*point [1] - quatrot_left.z*point [2];
	quatrot_temp.x = quatrot_left.q*point [0] + 0 + quatrot_left.y*point [2] - quatrot_left.z*point [1];
	quatrot_temp.y = quatrot_left.q*point [1] - quatrot_left.x*point [2] + 0 + quatrot_left.z*point [0];
	quatrot_temp.z = quatrot_left.q*point [2] + quatrot_left.x*point [1] - quatrot_left.y*point [0] + 0;

	point [0] = quatrot_temp.q*quatrot_right.x + quatrot_temp.x*quatrot_right.q + quatrot_temp.y*quatrot_right.z - quatrot_temp.z*quatrot_right.y;
	point [1] = quatrot_temp.q*quatrot_right.y - quatrot_temp.x*quatrot_right.z + quatrot_temp.y*quatrot_right.q + quatrot_temp.z*quatrot_right.x;
	point [2] = quatrot_temp.q*quatrot_right.z + quatrot_temp.x*quatrot_right.y - quatrot_temp.y*quatrot_right.x + quatrot_temp.z*quatrot_right.q;

	//Moving the point back

	point [0] = point [0] + movvec [0];
	point [1] = point [1] + movvec [1];
	point [2] = point [2] + movvec [2];
}


static inline void get_trilininterpol_weights(double weights [][2][2], const double dx, const double dy, const double dz)
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

#ifdef __cplusplus
}
#endif
