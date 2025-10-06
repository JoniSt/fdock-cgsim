#pragma once

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline double calc_ddd_Mehler_Solmajer(double distance)
 //The function returns the value of the distance-dependend dielectric function.
//(Whole function copied from AutoDock...)
{

    double epsilon = 1.0L;
    double lambda = 0.003627L;
    double epsilon0 = 78.4L;
    double A = -8.5525L;
    double B;
    B = epsilon0 - A;
    double rk= 7.7839L;
    double lambda_B;
    lambda_B = -lambda * B;

    epsilon = A + B / (1.0L + rk*exp(lambda_B * distance));

    return epsilon;
}

#ifdef __cplusplus
}
#endif
