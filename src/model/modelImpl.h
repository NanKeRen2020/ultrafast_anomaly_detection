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

#ifndef MODELIMPL_H_INCLUDED
#define MODELIMPL_H_INCLUDED

#include <cstdlib>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <opencv2/opencv.hpp>

#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include "../Utilities/LibImages.h"
#include "../Utilities/PatchManager/patchManager.h"
#include "../Utilities/PartitionTree/prmTree.h"
#include "../Utilities/PartitionTree/forestManager.h"
#include "../Utilities/PartitionTree/vptree.h"
#include "../Utilities/PatchManager/databasePatchManager.h"
#include "../Utilities/PatchManager/imagePatchManager.h"
#include "../Utilities/PatchSearch/localRefinement.h"
#include "../Utilities/pocketfft_hdronly.h"

#include "params.h"
#include <cuda_runtime.h>
#include <cufftXt.h>


/**
 * @brief Compute the final weighted aggregation.
 *
 * result: contain the aggregation, will contain the result at the end.
 * weight: contains the weights used during the aggregation;
 *
 * @return none.
 **/

inline void computeWeightedAggregation(
	std::vector<float>& result, const ImageSize imSize,
	std::vector<float> const& weight);


cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &data);

void detectNFA(const std::vector<float>& pixelNFA, const std::vector<float>& radiusNFA,
               ImageSize nfaSize, std::vector<std::tuple<float, int, int, int>>& detections,
                int pow_scale, int pow_layer, int nfa_thresh, cv::Size oriSize);


class NNForward
{

public:
   NNForward(const cv::Size& size, int channels, const std::string& proto_path, const std::string& model_path);

   cv::Mat forward(const cv::Mat& image, int layer_num);

private:
   cv::dnn::Net net;

};


cv::Mat get_nn_output(const cv::Mat& image, NNForward& nnForward, int layer_num, 
                      const std::vector<int>& scale_indexes);


std::vector<float> get_candidate(cv::Mat& image, int layer_num, int patch_size,
                                 const std::vector<int>& indexes, ImageSize& image_size,
                                 const std::string& proto_path = std::string(), 
                                 const std::string& model_path = std::string());

float get_distribution_score(float alpha, cv::Mat& residual, bool laplace, 
                             cv::Mat& data_mat, cv::Mat& cdf_data, float& scale);

void convert_distribution(std::vector<float>& residual_vec, ImageSize imSize);





class NFADetectionImpl
{

public:

    NFADetectionImpl(int R, float l, int md = 0);

	NFADetectionImpl(const ImageSize& imSize, int R, float l, int M, int N, int md = 0);

	// NFADetection(const NFADetection& nfa_detectipn);
    // NFADetection& operator=(const NFADetection& nfa_detectipn);

	void applyDetection(const std::vector<float>& residual, int HALFPATCHSIZE, 
	                    std::vector<float>& pixelNFA, std::vector<float>& radiusNFA);

	~NFADetectionImpl();

private:

    ImageSize imageSize;
    int method;

	int n0;
	int n1;

	float n02;
	float n12;

    float n1_dft;
	float tolog10;

	float n01;
	float n01d;
    float rn01;
    float n02_12;
	float mn;

    float logc;

	int rn01d;
    int R;
    float l;

	std::vector<std::vector<std::complex<float>>> ffts;

    std::vector<std::vector<float>> noise_gs;
    std::vector<std::vector<std::complex<float>>> dft_rec_noise;
	std::vector<std::vector<std::vector<std::complex<float>>>> dft_conv_noise;
	std::vector<std::vector<std::vector<float>>> filtered_by_dfts;	

	std::vector<fftwf_plan> plans0_wf;
	std::vector<std::vector<fftwf_plan>> plans1_wf;

	pocketfft::shape_t shape;
	pocketfft::stride_t stride0;
	pocketfft::stride_t stride1;
	pocketfft::shape_t axes;

    std::array<int, 2> fft_size;
	cufftHandle plan0;
	cufftHandle plan1;
	cudaStream_t stream0 = NULL;
	cudaStream_t stream1 = NULL;
    float *fdata0 = nullptr;	
	cufftComplex *ccdata0 = nullptr;
    float *fdata1 = nullptr;	
	cufftComplex *ccdata1 = nullptr;

	std::vector<float> noise_gs_cu;
	std::vector<std::complex<float>> ffts_cu;
    std::vector<std::complex<float>> dft_rec_noise_cu;
    std::vector<std::complex<float>> dft_conv_noise_cu;
    std::vector<float> filtered_by_dfts_cu;

};


#endif // MODELIMPL_H_INCLUDED
