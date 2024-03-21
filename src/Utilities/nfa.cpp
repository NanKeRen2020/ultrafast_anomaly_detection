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

/**
 * @file nfa.cpp
 * @brief NFA functions.
 *
 * @author Thibaud Ehret <ehret.thibaud@gmail.com>
 **/

#include "nfa.h"
#include <omp.h>
#include "pocketfft_hdronly.h"
#include <fstream>

using namespace std;
#include <iostream>

#include <cuda_runtime.h>
#include <cufftXt.h>


void coloredNoiseStatistic(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE)
{
	int n0 = imSize.width;
	int n1 = imSize.height;

	float n02 = n0*n0;
	float n12 = n1*n1;
    float logc = std::log(imSize.nChannels);
	float tolog10 = log(10);

    //#pragma omp parallel for
	for(int c = 0; c < imSize.nChannels; ++c)
	{
		std::vector<float> noise_gs(n0*n1);
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
				noise_gs[y + x*n1] = residual[x*imSize.nChannels + y*imSize.width*imSize.nChannels + c];

		int n1_dft = n1/2+1;

		std::vector<complex<float> > dft_rec_noise(n0*n1_dft);
		fftwf_plan plan = fftwf_plan_dft_r2c_2d(n0, n1,
				noise_gs.data(),
				reinterpret_cast<fftwf_complex*>(dft_rec_noise.data()),
				FFTW_ESTIMATE);
		fftwf_execute(plan);
		fftwf_destroy_plan(plan);

		/// Compute the measure kernel $s$ for each radius $r$
		std::vector<std::vector<float> > s(R, std::vector<float>(n0*n1));
		std::vector<std::vector<complex<float> > > ffts(R, std::vector<complex<float> >(n0*n1_dft));
		for(int r = 0; r < R; ++r)
		{
			float norms = 0;
			for(int i = 0, x = -n0/2+1; i < n0; ++i, ++x)
				for(int j = 0, y = -n1/2+1; j < n1; ++j, ++y)
				{
					if(x*x+y*y <= (r+1)*(r+1)*l*l)
					{
						s[r][j + i * n1] = 1;
						norms++;
					}
					else
						s[r][j + i * n1] = 0;
				}
			// normalize s in l_1
			for(int x = 0; x < n0; ++x)
				for(int y = 0; y < n1; ++y)
				{
					s[r][y + x*n1] = s[r][y + x*n1] / norms;
				}

			plan = fftwf_plan_dft_r2c_2d(n0, n1,
					s[r].data(),
					reinterpret_cast<fftwf_complex*>(ffts[r].data()),
					FFTW_ESTIMATE);
			fftwf_execute(plan);
			fftwf_destroy_plan(plan);
		}

		std::vector<std::vector<complex<float> > > dft_conv_noise(R, std::vector<complex<float> >(n0*n1_dft));
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1_dft; ++y)
			{
				/// Compute m(.,.)
				for(int r = 0; r < R; ++r)
					dft_conv_noise[r][y + x * n1_dft] = ffts[r][y + x * n1_dft] * dft_rec_noise[y + x * n1_dft];
			}

		std::vector<std::vector<float> > filtered_by_dft(R, std::vector<float>(n0*n1));
		/// Inverse back the dft
		for(int r = 0; r < R; ++r)
		{	
			fftwf_plan plan = fftwf_plan_dft_c2r_2d(n0, n1,
					reinterpret_cast<fftwf_complex*>(dft_conv_noise[r].data()),
					filtered_by_dft[r].data(),
					FFTW_ESTIMATE);
			fftwf_execute(plan);
			fftwf_destroy_plan(plan);
		}

		for(int r = 0; r < R; ++r)
		{
			std::vector<float> backup(n0*n1);
			for(int x = 0; x < n0; ++x) 
				for(int y = 0; y < n1; ++y) 
					backup[y + x*n1] = filtered_by_dft[r][y + x * n1];

			for(int x = 0; x < n0; ++x) 
			{
				int xs = (x + (n0)/2) % n0;
				for(int y = 0; y < n1; ++y) 
				{
					int ys = (y + (n1)/2) % n1;

					filtered_by_dft[r][ys + xs*n1] = backup[y + x * n1];
				}
			}
		}

		std::vector<double> sigmaphi(R);
		for(int r = 0; r < R; ++r)
		{
			// the residual is supposed to be centered. This doesn't change after the application of a Gaussian 
			sigmaphi[r] = 0.;
			for(int x = 0, ii = 0; x < n0; ++x)
				for(int y = 0; y < n1; ++y, ++ii)
					sigmaphi[r] += (filtered_by_dft[r][y + n1*x] * filtered_by_dft[r][y + n1*x] - sigmaphi[r]) / (float)(ii + 1);
			sigmaphi[r] /= (n02*n12);
			sigmaphi[r] = sqrt(std::max(sigmaphi[r], 0.));
		}

		for(int r = 0; r < R; ++r)
		for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
		for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
		{
			float temp;
			temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dft[r][y + n1*x] / (SQRT2*sigmaphi[r]*n0*n1);
			//temp = (abs(temp) > 26) ? -100000000 : std::log(imSize.nChannels * 4./3.*R*M*N*std::erfc(std::abs(temp)))/tolog10;
			temp = (abs(temp) > 26) ? -100000000 : std::log(M*N*std::erfc(std::abs(temp)))/tolog10;
            if(temp < pixelNFA[x + y*imSize.width])
            {
                pixelNFA[x + y*imSize.width] = temp;
                radiusNFA[x + y*imSize.width] = (r+1)*l;
            }
        }
	}

	// Add the complement coefficient to take into account the multichannel affect
	for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
	for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
	{
		pixelNFA[x + y*imSize.width] += logc;
	}
}


void coloredNoiseStatistic_fast(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE)
{
	int n0 = imSize.width;
	int n1 = imSize.height;

	float n02 = n0*n0;
	float n12 = n1*n1;

    int n1_dft = n1/2+1;
	float tolog10 = log(10);

	int n01 = n0*n1;
	int n01d = n0*n1_dft;
    int rn01 = R*n01;
    float n02_12 = n02*n12;

    float logc = std::log(imSize.nChannels);


    std::vector<std::vector<float> > s(R, std::vector<float>(n01));
	std::vector<std::vector<complex<float> > > ffts(R, std::vector<complex<float> >(n01d));


	for(int r = 0; r < R; ++r)
	{
		float norms = 0;
		for(int i = 0, x = -n0/2+1; i < n0; ++i, ++x)
			for(int j = 0, y = -n1/2+1; j < n1; ++j, ++y)
			{
				if(x*x+y*y <= (r+1)*(r+1)*l*l)
				{
					s[r][j + i * n1] = 1;
					norms++;
				}
				else
					s[r][j + i * n1] = 0;
			}
		// normalize s in l_1
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
			{
				s[r][y + x*n1] = s[r][y + x*n1] / norms;
			}

	}


    auto start = std::chrono::high_resolution_clock::now();
	fftw_init_threads();
    std::vector<fftwf_plan> plans;
	fftw_plan_with_nthreads(imSize.nChannels);
	for(int r = 0; r < R; ++r)
	{
		plans.emplace_back(fftwf_plan_dft_r2c_2d(n0, n1,
					s[r].data(),
					reinterpret_cast<fftwf_complex*>(ffts[r].data()),
					FFTW_ESTIMATE));
	}
    // #pragma omp parallel for
    for(int r = 0; r < R; ++r)
	{
        fftwf_execute(plans[r]);
	}
	for(int r = 0; r < R; ++r)
	{
       fftwf_destroy_plan(plans[r]);
	}
    plans.clear();
	

    std::vector<std::vector<float>> noise_gs(imSize.nChannels, std::vector<float>(n01));
    std::vector<std::vector<complex<float>>> dft_rec_noise(imSize.nChannels, 
	                                         std::vector<complex<float>>(n01d));
	
	std::vector<std::vector<std::vector<complex<float>>>> dft_conv_noise(imSize.nChannels, 
	    std::vector<std::vector<complex<float>>>(R, std::vector<complex<float>>(n01d)));
	std::vector<std::vector<std::vector<float>>> filtered_by_dfts(imSize.nChannels, 
	    std::vector<std::vector<float>>(R, std::vector<float>(n01)));	
	std::vector<std::vector<fftwf_plan>> plans1(imSize.nChannels, std::vector<fftwf_plan>());

    start = std::chrono::high_resolution_clock::now();
    fftw_plan_with_nthreads(imSize.nChannels);
	for(int c = 0; c < imSize.nChannels; ++c)
	{
        plans.emplace_back(fftwf_plan_dft_r2c_2d(n0, n1,
				noise_gs[c].data(),
				reinterpret_cast<fftwf_complex*>(dft_rec_noise[c].data()),
				FFTW_ESTIMATE));
		for(int r = 0; r < R; ++r)
		{
			plans1[c].emplace_back(fftwf_plan_dft_c2r_2d(n0, n1,
					reinterpret_cast<fftwf_complex*>(dft_conv_noise[c][r].data()),
					filtered_by_dfts[c][r].data(),
					FFTW_ESTIMATE));
		}

	}							

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "===>fftw time0: " << diff.count() << std::endl;

	std::vector<std::vector<float>> pixelNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));
	std::vector<std::vector<float>> radiusNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));
    
	// if (imSize.nChannels > 1)
	// #pragma omp parallel for
	#pragma omp parallel for
	for(int c = 0; c < imSize.nChannels; ++c)
	{
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
				noise_gs[c][y + x*n1] = residual[x*imSize.nChannels + y*imSize.width*imSize.nChannels + c];

        fftwf_execute(plans[c]);

        std::vector<double> sigmaphi(R);
		for(int r = 0; r < R; ++r)
		{	

		    for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1_dft; ++y)
			{
				dft_conv_noise[c][r][y + x * n1_dft] = ffts[r][y + x * n1_dft] * dft_rec_noise[c][y + x * n1_dft];
			}

            fftwf_execute(plans1[c][r]);

			std::vector<float> backup(n01);
			for(int x = 0; x < n0; ++x) 
				for(int y = 0; y < n1; ++y) 
					backup[y + x*n1] = filtered_by_dfts[c][r][y + x * n1];

			for(int x = 0; x < n0; ++x) 
			{
				int xs = (x + (n0)/2) % n0;
				for(int y = 0; y < n1; ++y) 
				{
					int ys = (y + (n1)/2) % n1;

					filtered_by_dfts[c][r][ys + xs*n1] = backup[y + x * n1];
				}
			}

			// the residual is supposed to be centered. This doesn't change after the application of a Gaussian 
			sigmaphi[r] = 0.;
			for(int x = 0, ii = 0; x < n0; ++x)
				for(int y = 0; y < n1; ++y, ++ii)
					sigmaphi[r] += (filtered_by_dfts[c][r][y + n1*x] * filtered_by_dfts[c][r][y + n1*x] - sigmaphi[r]) / (float)(ii + 1);
			sigmaphi[r] /= (n02_12);
			sigmaphi[r] = sqrt(std::max(sigmaphi[r], 0.));

			for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
			for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
			{
				float temp;
				temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts[c][r][y + n1*x] / (SQRT2*sigmaphi[r]*float(n01));
				//temp = (abs(temp) > 26) ? -100000000 : std::log(imSize.nChannels * 4./3.*R*M*N*std::erfc(std::abs(temp)))/tolog10;
				temp = (abs(temp) > 26) ? -100000000 : std::log(M*N*std::erfc(std::abs(temp)))/tolog10;
				if(temp < pixelNFA[x + y*imSize.width])
				{
					pixelNFAs[c][x + y*imSize.width] = temp;
					radiusNFAs[c][x + y*imSize.width] = (r+1)*l;
				}
			}

		}

	}

	for(int c = 0; c < imSize.nChannels; ++c)
	{
		fftwf_destroy_plan(plans[c]);
		for(int r = 0; r < R; ++r)
		{
		    fftwf_destroy_plan(plans1[c][r]);
		}	
 
		
		// Add the complement coefficient to take into account the multichannel affect
		for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
		for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
		{
			if (pixelNFAs[c][x + y*imSize.width] < pixelNFA[x + y*imSize.width])
			{
				pixelNFA[x + y*imSize.width] = pixelNFAs[c][x + y*imSize.width];
				radiusNFA[x + y*imSize.width] = radiusNFAs[c][x + y*imSize.width];
			}
			pixelNFA[x + y*imSize.width] += logc;
		}
	}
    fftw_cleanup_threads();
	
	end = std::chrono::high_resolution_clock::now();
	diff = end - start;
	std::cout << "===>fftw time1: " << diff.count() << std::endl;

	

}


void coloredNoiseStatistic_pocketfft(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE)
{
	unsigned  n0 = imSize.width;
	unsigned n1 = imSize.height;

	float n02 = n0*n0;
	float n12 = n1*n1;

    float n1_dft = n1/2+1;
	float tolog10 = log(10);

	float n01 = n0*n1;
	float n01d = n0*n1_dft;
    float rn01 = R*n01;
    float n02_12 = n02*n12;
	float mn = M*N;

    float logc = std::log(imSize.nChannels);

    pocketfft::shape_t shape{n0, n1};
    pocketfft::stride_t stride0(shape.size());
	size_t fsize=sizeof(float);
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        stride0[i]=fsize;
        fsize*=shape[i];
    }

	pocketfft::stride_t stride1(shape.size());
    size_t cfsize=sizeof(std::complex<float>);
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        stride1[i]=cfsize;
        cfsize*=n1_dft;
    }

    pocketfft::shape_t axes;
    for (size_t i=0; i<shape.size(); ++i)
       axes.push_back(i);
    // std::cout << axes[0] << ", " << axes[1] << std::endl;

    std::vector<std::vector<float> > s(R, std::vector<float>(n01));
	std::vector<std::vector<complex<float> > > ffts(R, std::vector<complex<float> >(n01d));
	
	#pragma omp parallel for
	for(int r = 0; r < R; ++r)
	{
		float norms = 0;
		for(int i = 0, x = -n0/2+1; i < n0; ++i, ++x)
			for(int j = 0, y = -n1/2+1; j < n1; ++j, ++y)
			{
				if(x*x+y*y <= (r+1)*(r+1)*l*l)
				{
					s[r][j + i * n1] = 1;
					norms++;
				}
				else
					s[r][j + i * n1] = 0;
			}
		// normalize s in l_1
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
			{
				s[r][y + x*n1] = s[r][y + x*n1] / norms;
			}

	    // in & out stride
		// auto start = std::chrono::high_resolution_clock::now();
		pocketfft::r2c(shape, stride0, stride1, axes, pocketfft::detail::FORWARD,
					s[r].data(), ffts[r].data(), 1.f, 1);
		// auto end = std::chrono::high_resolution_clock::now();
		// std::chrono::duration<double> diff = end - start;
        // std::cout << "s r2c time: " << diff.count() << std::endl;

	}

    // pocketfft::shape_t axes1;
    // for (size_t i=shape.size() - 1; i>=0; --i)
    //    axes1.push_back(i);

    pocketfft::shape_t shape1{n0, n1_dft};
    std::vector<std::vector<float>> noise_gs(imSize.nChannels, std::vector<float>(n01));
    std::vector<std::vector<complex<float>>> dft_rec_noise(imSize.nChannels, 
	                                         std::vector<complex<float>>(n01d));
	
	std::vector<std::vector<std::vector<complex<float>>>> dft_conv_noise(imSize.nChannels, 
	    std::vector<std::vector<complex<float>>>(R, std::vector<complex<float>>(n01d)));
	std::vector<std::vector<std::vector<float>>> filtered_by_dfts(imSize.nChannels, 
	    std::vector<std::vector<float>>(R, std::vector<float>(n01)));							

	std::vector<std::vector<float>> pixelNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));
	std::vector<std::vector<float>> radiusNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));
    
	//#pragma omp parallel for
	for(int c = 0; c < imSize.nChannels; ++c)
	{
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
				noise_gs[c][y + x*n1] = residual[x*imSize.nChannels + y*imSize.width*imSize.nChannels + c];

		// auto start = std::chrono::high_resolution_clock::now();
		pocketfft::r2c(shape, stride0, stride1, axes, pocketfft::detail::FORWARD,
					   noise_gs[c].data(), dft_rec_noise[c].data(), 1.f, 2);
		// auto end = std::chrono::high_resolution_clock::now();
		// std::chrono::duration<double> diff = end - start;
        // std::cout << "noise_gs r2c time: " << diff.count() << std::endl;

        std::vector<double> sigmaphi(R);
		#pragma omp parallel for
		for(int r = 0; r < R; ++r)
		{	

		    for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1_dft; ++y)
			{
				dft_conv_noise[c][r][y + x * n1_dft] = ffts[r][y + x * n1_dft] * dft_rec_noise[c][y + x * n1_dft];
			}

			// start = std::chrono::high_resolution_clock::now();
			pocketfft::c2r(shape, stride1, stride0, axes, pocketfft::detail::BACKWARD,
						   dft_conv_noise[c][r].data(), filtered_by_dfts[c][r].data(), 1.f, 1);
			// end = std::chrono::high_resolution_clock::now();
			// diff = end - start;
			// std::cout << "dft_conv_noise c2r time: " << diff.count() << std::endl;

			std::vector<float> backup(n01);
			for(int x = 0; x < n0; ++x) 
				for(int y = 0; y < n1; ++y) 
					backup[y + x*n1] = filtered_by_dfts[c][r][y + x * n1];

			for(int x = 0; x < n0; ++x) 
			{
				int xs = (x + (n0)/2) % n0;
				for(int y = 0; y < n1; ++y) 
				{
					int ys = (y + (n1)/2) % n1;

					filtered_by_dfts[c][r][ys + xs*n1] = backup[y + x * n1];
				}
			}

			// the residual is supposed to be centered. This doesn't change after the application of a Gaussian 
			sigmaphi[r] = 0.;
			for(int x = 0, ii = 0; x < n0; ++x)
				for(int y = 0; y < n1; ++y, ++ii)
					sigmaphi[r] += (filtered_by_dfts[c][r][y + n1*x] * filtered_by_dfts[c][r][y + n1*x] - sigmaphi[r]) / (float)(ii + 1);
			sigmaphi[r] /= n02_12;
			sigmaphi[r] = sqrt(std::max(sigmaphi[r], 0.));

			for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
			for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
			{
				float temp;
				temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts[c][r][y + n1*x] / (SQRT2*sigmaphi[r]*n01);
				temp = (abs(temp) > 26) ? -100000000 : std::log(M*N*std::erfc(std::abs(temp)))/tolog10;
				if(temp < pixelNFA[x + y*imSize.width])
				{
					pixelNFAs[c][x + y*imSize.width] = temp;
					radiusNFAs[c][x + y*imSize.width] = (r+1)*l;
				}

				if (pixelNFAs[c][x + y*imSize.width] < pixelNFA[x + y*imSize.width])
				{
					pixelNFA[x + y*imSize.width] = pixelNFAs[c][x + y*imSize.width];
					radiusNFA[x + y*imSize.width] = radiusNFAs[c][x + y*imSize.width];
				}
				pixelNFA[x + y*imSize.width] += logc;

			}

		}

	}

	
	// for(int c = 0; c < imSize.nChannels; ++c)
	// {
	// 	// Add the complement coefficient to take into account the multichannel affect
	// 	for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
	// 	for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
	// 	{
	// 		if (pixelNFAs[c][x + y*imSize.width] < pixelNFA[x + y*imSize.width])
	// 		{
	// 			pixelNFA[x + y*imSize.width] = pixelNFAs[c][x + y*imSize.width];
	// 			radiusNFA[x + y*imSize.width] = radiusNFAs[c][x + y*imSize.width];
	// 		}
	// 		pixelNFA[x + y*imSize.width] += logc;
	// 	}
	// }

}


void coloredNoiseStatistic_cufft_bk(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE)
{
	int n0 = imSize.width;
	int n1 = imSize.height;

	float n02 = n0*n0;
	float n12 = n1*n1;

    int n1_dft = n1/2+1;
	float tolog10 = log(10);
    float logc = std::log(imSize.nChannels);

	int n01 = n0*n1;
	int n01d = n0*n1_dft;


    std::vector<std::vector<float> > s(R, std::vector<float>(n01));
	std::vector<std::vector<std::complex<float> > > ffts(R, std::vector<std::complex<float> >(n01d));
	    
	std::ofstream outFile("/home/seeking/test_projects/anomaly_detection/results/nfa_fast_cufft.txt");

	//#pragma omp parallel for
	for(int r = 0; r < R; ++r)
	{
		std::stringstream ss;
		ss << r << " ==> ";

		float norms = 0;
		for(int i = 0, x = -n0/2+1; i < n0; ++i, ++x)
			for(int j = 0, y = -n1/2+1; j < n1; ++j, ++y)
			{
				if(x*x+y*y <= (r+1)*(r+1)*l*l)
				{
					s[r][j + i * n1] = 1;
					norms++;
					ss << "(" << i << ", " << j << ", " << norms << ")"; 

				}
				else
					s[r][j + i * n1] = 0;
			}
		// normalize s in l_1
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
			{
				s[r][y + x*n1] = s[r][y + x*n1] / norms;
			}

			outFile << ss.str() << std::endl;

	}

	std::vector<float> S;
	std::vector<std::complex<float>> FFTS;
	for (int i = 0; i < R; ++i)
	{
		S.insert(S.end(), s[i].begin(), s[i].end());
		FFTS.insert(FFTS.end(), ffts[i].begin(), ffts[i].end());
	}
    
    // apply cuda fft on s
	std::array<int, 2> fft = {n0, n1}; 
    cufftHandle plan;
    cudaStream_t stream = NULL;
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlanMany(&plan, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_R2C, R));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    float *d_input = nullptr;	
	cufftComplex *d_output = nullptr;

    auto start1 = std::chrono::high_resolution_clock::now();
    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(float) * R*n01));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(std::complex<float>) * R*n01d));

    CUDA_RT_CALL(cudaMemcpyAsync(d_input, S.data(), sizeof(float) * R*n01,
                                 cudaMemcpyHostToDevice, stream));

   
    CUFFT_CALL(cufftExecR2C(plan, d_input, d_output));

    CUDA_RT_CALL(cudaMemcpyAsync(FFTS.data(), d_output, sizeof(std::complex<float>) * R*n01d,
                                 cudaMemcpyDeviceToHost, stream));
    
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = end1 - start1;
    std::cout << "===> cufft time0: " << diff1.count() << std::endl;

	for ( int i = 0; i < ffts.size(); ++i)
	{
		std::stringstream ss;
		ss << i << " ==> ";
		for (int j = 0; j < 8; ++j)
		    ss << ffts[i][j] << ",  ";
		outFile << ss.str() << std::endl;
		std::stringstream ss1;
		for (int j = 0; j < 8; ++j)
		    ss1 << ffts[i][ffts[i].size() - j] << "  ";
		outFile << ss1.str() << std::endl;
	}

	for ( int i = 0; i < ffts.size(); ++i)
	{
		std::stringstream ss;
		ss << i << " ==> ";
		for (int j = 0; j < 8; ++j)
		    ss << FFTS[j + ffts[i].size()*i] << ",  ";
		outFile << ss.str() << std::endl;
		std::stringstream ss1;
		for (int j = 0; j < 8; ++j)
		    ss1 << FFTS[ffts[i].size() - j + ffts[i].size()*i] << "  ";
		outFile << ss1.str() << std::endl;
	}

    for (int i = 0; i < ffts.size(); ++i)
	{
		std::cout << i << ", ";
		for (int j = 0; j < 8; ++j)
		std::cout << FFTS[ffts[i].size() - j + ffts[i].size()*i] << "   ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(plan));
     //CUDA_RT_CALL(cudaStreamDestroy(stream));
    for (int i = 0; i < ffts.size(); ++i)
	{
		std::cout << i << ", ";
		for (int j = 0; j < 8; ++j)
		std::cout << ffts[i][ffts[i].size() - j] << "   ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
    

    //CUDA_RT_CALL(cudaDeviceReset());
    auto end2 = std::chrono::high_resolution_clock::now();
    diff1 = end2 - end1;
    std::cout << "===> cufft time00: " << diff1.count() << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> noise_gs(imSize.nChannels, std::vector<float>(n01));
    std::vector<std::vector<complex<float>>> dft_rec_noise(imSize.nChannels, 
	                                         std::vector<complex<float>>(n01d));
	
	for(int c = 0; c < imSize.nChannels; ++c)
	{
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
				noise_gs[c][y + x*n1] = residual[x*imSize.nChannels + y*imSize.width*imSize.nChannels + c];
	}	
    for (int i = 0; i < dft_rec_noise.size(); ++i)
	{
		std::cout << dft_rec_noise[i][dft_rec_noise[i].size() - 1] << ", ";
	}
	std::cout << std::endl;

	std::vector<float> NOISE_GS;
	std::vector<std::complex<float>> DFT_REC_NOISE;
	for (int i = 0; i < imSize.nChannels; ++i)
	{
		NOISE_GS.insert(NOISE_GS.end(), noise_gs[i].begin(), noise_gs[i].end());
		DFT_REC_NOISE.insert(DFT_REC_NOISE.end(), dft_rec_noise[i].begin(), dft_rec_noise[i].end());
	}

    cufftHandle plan0, plan1;
    cudaStream_t stream0 = NULL;
    
    CUFFT_CALL(cufftCreate(&plan0));
    CUFFT_CALL(cufftPlanMany(&plan0, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_R2C, imSize.nChannels));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan0, stream0));
    
    float *d_input0 = nullptr;	
	cufftComplex *d_output0 = nullptr;


    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input0), sizeof(float) * imSize.nChannels*n01));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output0), sizeof(std::complex<float>) * imSize.nChannels*n01d));

    CUDA_RT_CALL(cudaMemcpyAsync(d_input0, noise_gs[0].data(), sizeof(float) * imSize.nChannels*n01,
                                 cudaMemcpyHostToDevice, stream0));

    CUFFT_CALL(cufftExecR2C(plan0, d_input0, d_output0));

    CUDA_RT_CALL(cudaMemcpyAsync(dft_rec_noise[0].data(), d_output0, sizeof(std::complex<float>) * imSize.nChannels*n01d,
                                 cudaMemcpyDeviceToHost, stream0));

    CUDA_RT_CALL(cudaStreamSynchronize(stream0));

    for (int i = 0; i < dft_rec_noise.size(); ++i)
	{
		std::cout << dft_rec_noise[i][dft_rec_noise[i].size() - 1] << ", ";
	}
	std::cout << std::endl;
    

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input0));
    CUDA_RT_CALL(cudaFree(d_output0));

    CUFFT_CALL(cufftDestroy(plan0));

    //CUDA_RT_CALL(cudaStreamDestroy(stream0));

    //CUDA_RT_CALL(cudaDeviceReset());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "===> cufft time1: " << diff.count() << std::endl;


	std::vector<std::vector<std::vector<complex<float>>>> dft_conv_noise(imSize.nChannels, 
	    std::vector<std::vector<complex<float>>>(R, std::vector<complex<float>>(n01d)));
	std::vector<std::vector<std::vector<float>>> filtered_by_dfts(imSize.nChannels, 
	    std::vector<std::vector<float>>(R, std::vector<float>(n01)));	

    for(int c = 0; c < imSize.nChannels; ++c)
	{
		for(int r = 0; r < R; ++r)
		{	
			for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1_dft; ++y)
			{
				dft_conv_noise[c][r][y + x * n1_dft] = ffts[r][y + x * n1_dft] * dft_rec_noise[c][y + x * n1_dft];
			}

		}
	}

	cufftComplex *d_input1 = nullptr;
	float *d_output1 = nullptr;	
	cudaStream_t stream1 = NULL;
    fft = std::array<int, 2>{n0, n1_dft}; 

    CUFFT_CALL(cufftCreate(&plan1));
    CUFFT_CALL(cufftPlanMany(&plan1, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_C2R, R*imSize.nChannels));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan1, stream1));

    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input1), sizeof(std::complex<float>) * R*imSize.nChannels*n01d));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output1), sizeof(float) * R*imSize.nChannels*n01));

    CUDA_RT_CALL(cudaMemcpyAsync(d_input1, dft_conv_noise[0][0].data(), sizeof(std::complex<float>) * R*imSize.nChannels*n01d,
                                 cudaMemcpyHostToDevice, stream1));

    CUFFT_CALL(cufftExecC2R(plan1, d_input1, d_output1));

    CUDA_RT_CALL(cudaMemcpyAsync(filtered_by_dfts[0][0].data(), d_output1, sizeof(float) * R*imSize.nChannels*n01,
                                 cudaMemcpyDeviceToHost, stream1));

    CUDA_RT_CALL(cudaStreamSynchronize(stream1));

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input1));
    CUDA_RT_CALL(cudaFree(d_output1));

    CUFFT_CALL(cufftDestroy(plan1));

    CUDA_RT_CALL(cudaStreamDestroy(stream1));

    //CUDA_RT_CALL(cudaDeviceReset());


	std::vector<std::vector<float>> pixelNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));
	std::vector<std::vector<float>> radiusNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));

	for(int c = 0; c < imSize.nChannels; ++c)
	{
        std::vector<double> sigmaphi(R);
		for(int r = 0; r < R; ++r)
		{	

			std::vector<float> backup(n0*n1);
			for(int x = 0; x < n0; ++x) 
				for(int y = 0; y < n1; ++y) 
					backup[y + x*n1] = filtered_by_dfts[c][r][y + x * n1];

			for(int x = 0; x < n0; ++x) 
			{
				int xs = (x + (n0)/2) % n0;
				for(int y = 0; y < n1; ++y) 
				{
					int ys = (y + (n1)/2) % n1;

					filtered_by_dfts[c][r][ys + xs*n1] = backup[y + x * n1];
				}
			}

			// the residual is supposed to be centered. This doesn't change after the application of a Gaussian 
			sigmaphi[r] = 0.;
			for(int x = 0, ii = 0; x < n0; ++x)
				for(int y = 0; y < n1; ++y, ++ii)
					sigmaphi[r] += (filtered_by_dfts[c][r][y + n1*x] * filtered_by_dfts[c][r][y + n1*x] - sigmaphi[r]) / (float)(ii + 1);
			sigmaphi[r] /= (n02*n12);
			sigmaphi[r] = sqrt(std::max(sigmaphi[r], 0.));

			for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
			for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
			{
				float temp;
				temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts[c][r][y + n1*x] / (SQRT2*sigmaphi[r]*n0*n1);
				//temp = (abs(temp) > 26) ? -100000000 : std::log(imSize.nChannels * 4./3.*R*M*N*std::erfc(std::abs(temp)))/tolog10;
				temp = (abs(temp) > 26) ? -100000000 : std::log(M*N*std::erfc(std::abs(temp)))/tolog10;
				if(temp < pixelNFA[x + y*imSize.width])
				{
					pixelNFAs[c][x + y*imSize.width] = temp;
					radiusNFAs[c][x + y*imSize.width] = (r+1)*l;
				}
			}

		}

	}


	for(int c = 0; c < imSize.nChannels; ++c)
	{

		// Add the complement coefficient to take into account the multichannel affect
		for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
		for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
		{
			if (pixelNFAs[c][x + y*imSize.width] < pixelNFA[x + y*imSize.width])
			{
				pixelNFA[x + y*imSize.width] = pixelNFAs[c][x + y*imSize.width];
				radiusNFA[x + y*imSize.width] = radiusNFAs[c][x + y*imSize.width];
			}
			pixelNFA[x + y*imSize.width] += logc;
		}
	}

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "===> cufft time: " << diff.count() << std::endl;



}


void coloredNoiseStatistic_cufft(std::vector<float>& residual, ImageSize& imSize, int R, float l, std::vector<float>& pixelNFA, std::vector<float>& radiusNFA, int M, int N, int HALFPATCHSIZE)
{
	int n0 = imSize.width;
	int n1 = imSize.height;

	float n02 = n0*n0;
	float n12 = n1*n1;

    int n1_dft = n1/2+1;
	float tolog10 = log(10);
    float logc = std::log(imSize.nChannels);

	int n01 = n0*n1;
	int n01d = n0*n1_dft;
    int rn01 = R*n01;
	int n02_12 = n02*n12;

    std::vector<float> s(R*n01);
	std::vector<std::complex<float>> ffts(R*n01d);
	    
	std::ofstream outFile("/home/seeking/test_projects/anomaly_detection/results/nfa_fast_cufft.txt");

	//#pragma omp parallel for
	for(int r = 0; r < R; ++r)
	{
		std::stringstream ss;
		ss << r << " ==> ";

		float norms = 0;
		for(int i = 0, x = -n0/2+1; i < n0; ++i, ++x)
			for(int j = 0, y = -n1/2+1; j < n1; ++j, ++y)
			{
				if(x*x+y*y <= (r+1)*(r+1)*l*l)
				{
					s[r*n01 + j + i * n1] = 1;
					norms++;
					ss << "(" << i << ", " << j << ", " << norms << ")"; 

				}
				else
					s[r*n01 + j + i * n1] = 0;
			}
		// normalize s in l_1
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
			{
				s[r*n01 + y + x * n1] /= norms;
			}

			outFile << ss.str() << std::endl;

	}

    
    // apply cuda fft on s
	std::array<int, 2> fft = {n0, n1}; 
    cufftHandle plan;
    cudaStream_t stream = NULL;
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlanMany(&plan, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_R2C, R));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    float *d_input = nullptr;	
	cufftComplex *d_output = nullptr;

    auto start1 = std::chrono::high_resolution_clock::now();
    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(float) * R*n01));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(std::complex<float>) * R*n01d));

    CUDA_RT_CALL(cudaMemcpyAsync(d_input, s.data(), sizeof(float) * R*n01,
                                 cudaMemcpyHostToDevice, stream));

   
    CUFFT_CALL(cufftExecR2C(plan, d_input, d_output));

    CUDA_RT_CALL(cudaMemcpyAsync(ffts.data(), d_output, sizeof(std::complex<float>) * R*n01d,
                                 cudaMemcpyDeviceToHost, stream));
    
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = end1 - start1;
    std::cout << "===> cufft time0: " << diff1.count() << std::endl;


	for ( int i = 0; i < R; ++i)
	{
		std::stringstream ss;
		ss << i << " ==> ";
		for (int j = 0; j < 8; ++j)
		    ss << ffts[j + n01d*i] << ",  ";
		outFile << ss.str() << std::endl;
		std::stringstream ss1;
		for (int j = 0; j < 8; ++j)
		    ss1 << ffts[n01d - j + n01d*i] << "  ";
		outFile << ss1.str() << std::endl;
	}
	std::cout << std::endl;

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(plan));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    

    //CUDA_RT_CALL(cudaDeviceReset());
    auto end2 = std::chrono::high_resolution_clock::now();
    diff1 = end2 - end1;
    std::cout << "===> cufft time00: " << diff1.count() << std::endl;


	std::vector<float> noise_gs(imSize.nChannels*n01);
    std::vector<complex<float>> dft_rec_noise(imSize.nChannels*n01d);

    cufftHandle plan0, plan1;
    cudaStream_t stream0 = NULL;
    
    CUFFT_CALL(cufftCreate(&plan0));
    CUFFT_CALL(cufftPlanMany(&plan0, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_R2C, imSize.nChannels));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan0, stream0));
    
    float *d_input0 = nullptr;	
	cufftComplex *d_output0 = nullptr;
    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input0), sizeof(float) * imSize.nChannels*n01));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output0), sizeof(std::complex<float>) * imSize.nChannels*n01d));


    auto start = std::chrono::high_resolution_clock::now();

	for(int c = 0; c < imSize.nChannels; ++c)
	{
		for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1; ++y)
	            noise_gs[c*n01 + y + x*n1] = residual[x*imSize.nChannels + y*imSize.width*imSize.nChannels + c];
	}	

    CUDA_RT_CALL(cudaMemcpyAsync(d_input0, noise_gs.data(), sizeof(float) * imSize.nChannels*n01,
                                 cudaMemcpyHostToDevice, stream0));

    CUFFT_CALL(cufftExecR2C(plan0, d_input0, d_output0));

    CUDA_RT_CALL(cudaMemcpyAsync(dft_rec_noise.data(), d_output0, sizeof(std::complex<float>) * imSize.nChannels*n01d,
                                 cudaMemcpyDeviceToHost, stream0));

    CUDA_RT_CALL(cudaStreamSynchronize(stream0));
    

    /* free resources */
    // CUDA_RT_CALL(cudaFree(d_input0));
    // CUDA_RT_CALL(cudaFree(d_output0));

    // CUFFT_CALL(cufftDestroy(plan0));

    // CUDA_RT_CALL(cudaStreamDestroy(stream0));

    //CUDA_RT_CALL(cudaDeviceReset());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "===> cufft plan0 time: " << diff.count() << std::endl;

	cufftComplex *d_input1 = nullptr;
	float *d_output1 = nullptr;	
	cudaStream_t stream1 = NULL;
    //fft = std::array<int, 2>{n0, n1_dft}; 

    CUFFT_CALL(cufftCreate(&plan1));
    CUFFT_CALL(cufftPlanMany(&plan1, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_C2R, R*imSize.nChannels));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan1, stream1));
    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input1), sizeof(std::complex<float>) * R*imSize.nChannels*n01d));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output1), sizeof(float) * R*imSize.nChannels*n01));


    start1 = std::chrono::high_resolution_clock::now();
    std::vector<complex<float>> dft_conv_noise(imSize.nChannels*R*n01d);
    std::vector<float> filtered_by_dfts(imSize.nChannels*R*n01);
    int rn01d = R*n01d;
    for(int c = 0; c < imSize.nChannels; ++c)
	{
		for(int r = 0; r < R; ++r)
		{	
			for(int x = 0; x < n0; ++x)
			for(int y = 0; y < n1_dft; ++y)
			{
				dft_conv_noise[c*rn01d + r*n01d + y + x * n1_dft] = ffts[r*n01d + y + x * n1_dft] * dft_rec_noise[c*n01d + y + x * n1_dft];
			}

		}
	}
    end = std::chrono::high_resolution_clock::now();
    diff = end - start1;
    std::cout << "===> cufft plan1 data time: " << diff.count() << std::endl;

    CUDA_RT_CALL(cudaMemcpyAsync(d_input1, dft_conv_noise.data(), sizeof(std::complex<float>) * R*imSize.nChannels*n01d,
                                 cudaMemcpyHostToDevice, stream1));

    CUFFT_CALL(cufftExecC2R(plan1, d_input1, d_output1));

    CUDA_RT_CALL(cudaMemcpyAsync(filtered_by_dfts.data(), d_output1, sizeof(float) * R*imSize.nChannels*n01,
                                 cudaMemcpyDeviceToHost, stream1));

    CUDA_RT_CALL(cudaStreamSynchronize(stream1));

    /* free resources */
    // CUDA_RT_CALL(cudaFree(d_input1));
    // CUDA_RT_CALL(cudaFree(d_output1));

    // CUFFT_CALL(cufftDestroy(plan1));

    // CUDA_RT_CALL(cudaStreamDestroy(stream1));

    //CUDA_RT_CALL(cudaDeviceReset());

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "===> cufft plan1 time: " << diff.count() << std::endl;


	std::vector<std::vector<float>> pixelNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));
	std::vector<std::vector<float>> radiusNFAs(imSize.nChannels, std::vector<float>(pixelNFA.size()));


	//#pragma omp parallel for num_threads(8) schedule(dynamic)
    #pragma omp parallel for
	for(int c = 0; c < imSize.nChannels; ++c)
	{
        std::vector<double> sigmaphi(R);
		for(int r = 0; r < R; ++r)
		{	

			std::vector<float> backup(n01);
			for(int x = 0; x < n0; ++x) 
				for(int y = 0; y < n1; ++y) 
					backup[y + x*n1] = filtered_by_dfts[c*rn01 + r*n01 + y + x * n1];

			for(int x = 0; x < n0; ++x) 
			{
				int xs = (x + (n0)/2) % n0;
				for(int y = 0; y < n1; ++y) 
				{
					int ys = (y + (n1)/2) % n1;

					filtered_by_dfts[c*rn01 + r*n01 + ys + xs*n1] = backup[y + x * n1];
				}
			}

			// the residual is supposed to be centered. This doesn't change after the application of a Gaussian 
			sigmaphi[r] = 0.;
			for(int x = 0, ii = 0; x < n0; ++x)
				for(int y = 0; y < n1; ++y, ++ii)
					sigmaphi[r] += (filtered_by_dfts[c*rn01 + r*n01 + y + n1*x] * filtered_by_dfts[c*rn01 + r*n01 + y + n1*x] - sigmaphi[r]) / (float)(ii + 1);
			sigmaphi[r] /= n02_12;
			sigmaphi[r] = sqrt(std::max(sigmaphi[r], 0.));

			for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
			for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
			{
				float temp;
				temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts[c*rn01 + r*n01 + y + n1*x] / (SQRT2*sigmaphi[r]*n01);
				//temp = (abs(temp) > 26) ? -100000000 : std::log(imSize.nChannels * 4./3.*R*M*N*std::erfc(std::abs(temp)))/tolog10;
				temp = (abs(temp) > 26) ? -100000000 : std::log(M*N*std::erfc(std::abs(temp)))/tolog10;
				if(temp < pixelNFA[x + y*imSize.width])
				{
					pixelNFAs[c][x + y*imSize.width] = temp;
					radiusNFAs[c][x + y*imSize.width] = (r+1)*l;
				}
			}

		}

	}


	for(int c = 0; c < imSize.nChannels; ++c)
	{

		// Add the complement coefficient to take into account the multichannel affect
		for(int x = HALFPATCHSIZE; x < (imSize.width-HALFPATCHSIZE); ++x)
		for(int y = HALFPATCHSIZE; y < (imSize.height-HALFPATCHSIZE); ++y)
		{
			if (pixelNFAs[c][x + y*imSize.width] < pixelNFA[x + y*imSize.width])
			{
				pixelNFA[x + y*imSize.width] = pixelNFAs[c][x + y*imSize.width];
				radiusNFA[x + y*imSize.width] = radiusNFAs[c][x + y*imSize.width];
			}
			pixelNFA[x + y*imSize.width] += logc;
		}
	}

    end1 = std::chrono::high_resolution_clock::now();
    diff = end1 - end;
    std::cout << "===> cufft result time: " << diff.count() << std::endl;



}



