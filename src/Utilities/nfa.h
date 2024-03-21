/*
 * Copyright (c) 2017, Thibaud Ehret <ehret.thibaud@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NFA_H_INCLUDED
#define NFA_H_INCLUDED

#include <vector>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include "LibImages.h"

#define SQRT2 1.41421356237

void coloredNoiseStatistic(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE);

void coloredNoiseStatistic_fast(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE);

void coloredNoiseStatistic_pocketfft(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE);

void coloredNoiseStatistic_cufft(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE);


#endif // NFA_H_INCLUDED


#pragma once


// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
