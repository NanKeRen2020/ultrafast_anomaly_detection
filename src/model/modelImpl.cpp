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
 * @file 
 * @brief 
 *
 * @author Thibaud Ehret <ehret.thibaud@gmail.com>
 **/


#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <sys/time.h>
#include <iostream>

#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <opencv2/opencv.hpp>
#include <complex>
#include <fftw3.h>
#include <boost/math/distributions/laplace.hpp>
#include <boost/math/distributions/normal.hpp>
#include "../Utilities/comparators.h"
#include "../Utilities/nfa.h"

#include "modelImpl.h"


/****************** vector转Mat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(std::vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
	return dest;
}


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Compute the final weighted aggregation.
 *
 * result: contain the aggregation, will contain the result at the end.
 * weight: contains the weights used during the aggregation;
 *
 * @return : none.
 **/

inline void computeWeightedAggregation(
	std::vector<float>& result,
	const ImageSize imSize,
	std::vector<float> const& weight
){
	for (unsigned y = 0; y < imSize.height; ++y)
	for (unsigned x = 0; x < imSize.width; ++x)
	for (unsigned c = 0; c < imSize.nChannels; ++c)
		result[c + x*imSize.nChannels + y*imSize.nChannels*imSize.width] /= weight[x + y*imSize.width];
}
 
cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &data)
{
    cv::Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        cv::Mat image_row = data[i].clone().reshape(1,1);
        cv::Mat row_i = dst.row(i);
        image_row.convertTo(row_i, CV_32F);
    }
    return dst;
}

float get_distribution_score(float alpha, cv::Mat& residual, bool laplace, cv::Mat& data_mat, cv::Mat& cdf_data, float& scale)
{
        std::vector<float> data_vec(residual.total());

        auto start = std::chrono::high_resolution_clock::now();
        residual.forEach<float>([&](float &val, const int *position)
        { 
            data_vec[position[1] + position[0]*residual.cols] = 
            ((val < 0) ? -1*std::pow(std::abs(val), alpha) : std::pow(std::abs(val), alpha));
        });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "===>get_score_test time0: " << diff.count() << std::endl;
        data_mat = convertVector2Mat(data_vec, residual.channels(), residual.rows);

    	float sum = std::accumulate(data_vec.begin(), data_vec.end(), 0.0f);

	    float mean =  sum / data_vec.size(); //均值
	    float accum  = 0.0f;
	    std::for_each(data_vec.begin(), data_vec.end(), [&](const float d) {
		            accum  += (d-mean)*(d-mean); });
        
        if (laplace)  scale = std::sqrt(accum/(data_vec.size())/2); //方差
        else   scale = std::sqrt(accum/(data_vec.size())); //方差5
        if (scale == 0 ) scale = 1;
        auto end1 = std::chrono::high_resolution_clock::now();
        diff = end1 - end;
        std::cout << "===>get_score_test time1: " << diff.count() << std::endl;        

        boost::math::laplace_distribution<float> lap_dis(0, scale);
        boost::math::normal_distribution<float> norm_dis(0, scale);

        cdf_data = data_mat.clone();
        cdf_data.forEach<float>([&](float &val, const int *position)
        {
           val = (laplace ? boost::math::cdf(lap_dis, val) : boost::math::cdf(norm_dis, val));
        });
    
        auto end2 = std::chrono::high_resolution_clock::now();
        diff = end2 - end1;
        std::cout << "===>get_score_test time2: " << diff.count() << std::endl;        


        
        cv::Mat hist, cumsum;
        float range[] = { 0.0, 1.0 };
        const float* histRange[] = { range };
        const int channels[] = { 0 };
        const int histSize[] = { 20 };
        cv::calcHist(&cdf_data, 1, channels, cv::Mat(), hist, 1, histSize, histRange, true, false);
        hist = hist / float(cv::sum(hist)[0]);
        cv::Mat culsum_hist(hist.size(), CV_32FC1);
        culsum_hist.at<float>(0) = hist.at<float>(0);
        for(size_t k = 1; k < hist.rows; ++k)
        {
           culsum_hist.at<float>(k) = hist.at<float>(k) + culsum_hist.at<float>(k-1);
        }   
        
        
        cv::Mat one_hist = cv::Mat::ones(hist.size(), CV_32FC1)*0.05f;
        for(size_t k = 1; k < one_hist.rows; ++k)
        {
           one_hist.at<float>(k) = one_hist.at<float>(k) + one_hist.at<float>(k-1);
        }        
        cv::subtract(culsum_hist, one_hist, culsum_hist);

        // culsum_hist.forEach<float>(
        //     [](float &val, const int *position){
        //         val = std::pow(val, 2.0f);
        // });
        auto end3 = std::chrono::high_resolution_clock::now();
        diff = end3 - end2;
        std::cout << "===>get_score_test time3: " << diff.count() << std::endl; 

        return cv::sum(culsum_hist)[0];
}



void convert_distribution(std::vector<float>& residual_vec, ImageSize imSize)
{
    cv::Mat residual_mat = convertVector2Mat(residual_vec, imSize.nChannels, imSize.height);

    std::vector<float> alphas{0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4};
    float best_score = std::numeric_limits<float>::max();
    boost::math::normal_distribution<float> norm_dis;
    cv::Mat data_mat, cdf_data, best_data, residual, residual_gaussian;
    float best_alpha, scale, score;
    std::string best_model;
    std::vector<cv::Mat> residuals;
    cv::split(residual_mat, residuals);
    int res_size = residuals.size();

    //#pragma omp parallel for
    for (int i = 0; i < residual_mat.channels(); ++i)
    {
        float alpha = 0.6;
        //for (auto alpha: alphas)
        {

            auto start = std::chrono::high_resolution_clock::now();
            score = get_distribution_score(alpha, residuals[i], 0, data_mat, cdf_data, scale);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "===>get_score time: " << diff.count() << ", " << score<< std::endl;
            
            if (score < best_score)
            {
                best_score = score;
                best_alpha = alpha;

                best_data = data_mat/scale;
    
                best_model = "gaussian";
            }


            //score = get_score_test(alpha, residual, 1, data_mat, cdf_data, scale);
            
            // if (score < best_score)
            // {

            //     // norm.isf() norm.ppf() in  boost::math::
            //     cv::Mat one_data = cv::Mat::ones(data_mat.size(), residual_mat.type());
            //     cv::Mat sf_data = cv::Mat::zeros(data_mat.size(), residual_mat.type());;
            //     //cv::subtract(one_data, cdf_data, sf_data);
            //     for (auto it = data_mat.begin<float>(); it != data_mat.end<float>(); ++it)
            //     {
                    
            //         //*(it) = boost::math::quantile(norm_dis_1, 0.5);
            //         if (*it > 0) *( sf_data.begin<float>() + (it - data_mat.begin<float>()) )
            //         = 0.5*exp(-1*(*it)/scale);
            //         else *( sf_data.begin<float>() + (it - data_mat.begin<float>()) )
            //         = 1 - 0.5*exp((*it)/scale);   


            //         *( sf_data.begin<float>() + (it - data_mat.begin<float>()) ) 
            //         = boost::math::quantile( boost::math::complement(norm_dis, 
            //         *( sf_data.begin<float>() + (it - data_mat.begin<float>())) ) );

                    
            //     }

            //     cv::Mat mask_sf = cv::Mat::zeros(data_mat.size(), data_mat.type() );
            //     for (auto it = sf_data.begin<float>(); it != sf_data.end<float>(); ++it)
            //     {
            //         if (*it > 0)
            //         *(mask_sf.begin<float>() + (it - sf_data.begin<float>())) = 1;                
            //     }
            //     best_data = sf_data.mul(mask_sf);      


            //     cv::Mat ppf_data = cv::Mat::zeros(data_mat.size(), data_mat.type() ); 

            //     ppf_data = ppf_data.mul(~mask_sf);
            //     cv::add(best_data, ppf_data, best_data);
    
            //     best_alpha = alpha;
            //     best_score = score;
                
            //     best_model = "laplace";
            // }

            
        }

        residuals[i] = best_data;
        
    }

    cv::merge(residuals, residual_gaussian);

    cv::Mat flat = residual_gaussian.reshape(1, residual_gaussian.total()*residual_gaussian.channels());
    residual_vec = residual_gaussian.isContinuous() ? flat : flat.clone();

}


cv::Mat get_nn_output(const cv::Mat& image, NNForward& nnForward, int layer_num, 
                      const std::vector<int>& scale_indexes)
{
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = nnForward.forward(image, layer_num);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "===>nn forward time: " << diff.count() << std::endl;
    // cv::Mat result1 = nnForward.forward(image, layer_num);
    // auto end1 = std::chrono::high_resolution_clock::now();
    // diff = end1 - end;
    // std::cout << "===>nn forward time1: " << diff.count() << std::endl;


    std::vector<cv::Mat> images;
    cv::dnn::imagesFromBlob(result, images);
    //int row = images[0].rows;
    std::vector<cv::Mat> channels;
    cv::split(images[0], channels);
    images.clear();

    // 选择特定的特征通道图
    cv::Mat data; 
    if (!scale_indexes.empty())
    {
        for (auto i : scale_indexes)
        {
            if (i < channels.size())
           images.emplace_back(channels[i]);
        }
    }
    else
    {
        // get pca channel
        cv::Mat pca_component = formatImagesForPCA(channels);
        int component = std::min(5, pca_component.size[0]/2); 
        cv::PCA pca(pca_component, cv::Mat(), cv::PCA::DATA_AS_COL, component);
        for (int i = 0; i < pca.eigenvectors.rows; ++i)
        {
            pca.eigenvectors.rowRange(i, i+1) = pca.eigenvectors.rowRange(i, i+1)/std::sqrt(*( pca.eigenvalues.begin<float>() + i));
            if (i == 1 || i == 4)
            pca.eigenvectors.rowRange(i, i+1) = -1*pca.eigenvectors.rowRange(i, i+1);       
        }
        data = pca.project(pca_component); 
    
        for (int j = 0; j < component; ++j)
        {
            images.emplace_back(data.rowRange(j, j+1));
        }
        
    }

    // 合并通道图像
    cv::merge(images, data);

    return data;
}


void detectNFA(const std::vector<float>& pixelNFA, const std::vector<float>& radiusNFA,
               ImageSize nfaSize, std::vector<std::tuple<float, int, int, int>>& detections,
                int pow_scale, int pow_layer, int nfa_thresh, cv::Size oriSize)
{
    int paddingx = std::round(std::max(0, (oriSize.height - pow_layer*pow_scale*oriSize.height)/2));
    int paddingy = std::round(std::max(0, (oriSize.width - pow_layer*pow_scale*oriSize.width)/2));

    int m, n;
    for (int i = 0; i < pixelNFA.size(); ++i)
    {
        if (pixelNFA[i] < nfa_thresh)
        {
            n = i / nfaSize.width;
            m = i % nfaSize.width;
            detections.emplace_back( std::make_tuple(pixelNFA[i], 
                        int(paddingx + pow_layer*pow_scale*(m+1.5)),
                        int(paddingy + pow_layer*pow_scale*(n+1.5)), 
                        int(pow_layer*pow_scale*radiusNFA[i] + 1) ) );
        }
    }

}



std::vector<float> get_candidate(cv::Mat& image, int layer_num, int patch_size,
                                const std::vector<int>& indexes, ImageSize& image_size,
                                const std::string& proto_path, const std::string& model_path)
{
    assert(image.cols >= 32 && image.rows >= 32);
    
    if ( layer_num >= 0 )
    {

        // get nnnetwork output
        auto start = std::chrono::high_resolution_clock::now();
        static NNForward nnForward(image.size(), image.channels(), proto_path, model_path);
        image = get_nn_output(image, nnForward, layer_num, indexes);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "===>get_nn_output time: " << diff.count() << std::endl;

    }
    else if (layer_num < 0)
    {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        if (indexes.empty())
        image = channels[0];  
        else if (indexes.size() == 1 && indexes[0] < 3)
        image = channels[indexes[0]];      
        else
        {
            /* code */
        }
        if (image.channels() == 1)
        image.convertTo(image, CV_32FC1);
        else
        {
            image.convertTo(image, CV_32FC3);
        }           
        
    }
    else
    {   }

    
    int qw = image.cols - image.cols%patch_size;
    int qh = image.rows - image.rows%patch_size;
    image = image(cv::Rect(0, 0, qw, qh)).clone();

    

    image_size.width = image.cols;
    image_size.height = image.rows;
    image_size.nChannels = image.channels();
    image_size.wh = image_size.width*image_size.height;
    image_size.whc = image_size.width*image_size.height*image_size.nChannels;
    image_size.wc = image_size.width*image_size.nChannels;

    cv::Mat flat = image.reshape(1, image.total()*image.channels());
    std::vector<float> candidate = image.isContinuous() ? flat : flat.clone();

    return candidate;

}

//------------------------------------------------- NNForward --------------------------------

NNForward::NNForward(const cv::Size& size, int channels, 
                     const std::string& proto_path, const std::string& model_path)
{
    // 依据输入图像修改模型文件
    std::ifstream proto_file(proto_path);
    std::string str;
    std::vector<std::string> strs;
    while (std::getline(proto_file, str))
    {
        strs.push_back(str);
    }
    strs[3] = "input_dim: " + std::to_string(channels);
    strs[4] = "input_dim: " + std::to_string(size.width);
    strs[5] = "input_dim: " + std::to_string(size.height);
    std::ofstream out_proto_file(proto_path);
    for (auto str: strs)
    {
        out_proto_file << str << std::endl;
    }
    out_proto_file.close();	 

    net = cv::dnn::readNetFromCaffe(proto_path, model_path);  
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);
}

cv::Mat NNForward::forward(const cv::Mat& image, int layer_num)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 255.0, cv::Size(image.cols, image.rows), 
                           cv::Scalar(0.40760392, 0.45795686, 0.48501961), false, false);
    net.setInput(blob);
    // std::vector<cv::String> layer_names = net.getLayerNames();  
    std::vector<cv::String> layer_names{"conv1_1", "conv1_2", "pool1","conv2_1", "conv2_2", "pool2",
                                          "conv3_1", "conv3_2", "conv3_3", "conv3_4", "pool3",
                                          "conv4_1", "conv4_2", "conv4_3", "conv4_4"};
    return net.forward(layer_names[layer_num]);
}



//------------------------------------------------- NFADetectionImpl --------------------------------
//

NFADetectionImpl::NFADetectionImpl(int iR, float il, int md):
	    tolog10(log(10)), method(md), R(iR), l(il) { }

NFADetectionImpl::NFADetectionImpl(const ImageSize& imSize, int iR, float il, int M, int N, int md):
	    imageSize(imSize), n0(imSize.width), n1(imSize.height), n02(n0*n0), n12(n1*n1), 
		n1_dft(n1/2+1), tolog10(log(10)), n01(n0*n1), n01d(n0*n1_dft), rn01(R*n01), 
		n02_12(n02*n12), mn(M*N), logc(std::log(imSize.nChannels)), method(md), 
        rn01d(R*n01d), R(iR), l(il), 

        ffts(std::vector<std::vector<std::complex<float>>>(R, std::vector<std::complex<float>>(n01d))),
        noise_gs(std::vector<std::vector<float>>(imSize.nChannels, std::vector<float>(n01))),
        dft_rec_noise(std::vector<std::vector<std::complex<float>>>(imSize.nChannels, 
	                                         std::vector<std::complex<float>>(n01d))),
	    dft_conv_noise(std::vector<std::vector<std::vector<std::complex<float>>>>(imSize.nChannels, 
	        std::vector<std::vector<std::complex<float>>>(R, std::vector<std::complex<float>>(n01d)))),
	    filtered_by_dfts(std::vector<std::vector<std::vector<float>>>(imSize.nChannels, 
	        std::vector<std::vector<float>>(R, std::vector<float>(n01)))),
        
	    plans1_wf(std::vector<std::vector<fftwf_plan>>(imSize.nChannels, std::vector<fftwf_plan>())),

        shape(pocketfft::shape_t{n0, n1}), stride0(pocketfft::stride_t(shape.size())),
        stride1(pocketfft::stride_t(shape.size())), fft_size(std::array<int, 2>{n0, n1}),

        ffts_cu(std::vector<std::complex<float>>(R*n01d)),
        noise_gs_cu(std::vector<float>(imSize.nChannels*n01)),
        dft_rec_noise_cu(std::vector<std::complex<float>>(imSize.nChannels*n01d)),

        dft_conv_noise_cu(std::vector<std::complex<float>>(imSize.nChannels*R*n01d)),
        filtered_by_dfts_cu(std::vector<float>(imSize.nChannels*R*n01))

{

    if (method == 0)
    {
        std::vector<float> s(R*n01);	    
        for(int r = 0; r < R; ++r)
        {

            float norms = 0;
            for(int i = 0, x = -n0/2+1; i < n0; ++i, ++x)
                for(int j = 0, y = -n1/2+1; j < n1; ++j, ++y)
                {
                    if(x*x+y*y <= (r+1)*(r+1)*l*l)
                    {
                        s[r*n01 + j + i * n1] = 1;
                        norms++;

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
        }

        cufftHandle plan;
        cudaStream_t stream = NULL;
        float *fdata = nullptr;	
        cufftComplex *ccdata = nullptr;
        CUFFT_CALL(cufftCreate(&plan));
        CUFFT_CALL(cufftPlanMany(&plan, fft_size.size(), fft_size.data(), nullptr, 1,
                                0, nullptr, 1, 0, CUFFT_R2C, R));              
        CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUFFT_CALL(cufftSetStream(plan, stream));
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&fdata), sizeof(float) * R*n01));
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&ccdata), sizeof(std::complex<float>) * R*n01d));
        CUDA_RT_CALL(cudaMemcpyAsync(fdata, s.data(), sizeof(float) * R*n01,
                                     cudaMemcpyHostToDevice, stream));
        CUFFT_CALL(cufftExecR2C(plan, fdata, ccdata));
        CUDA_RT_CALL(cudaMemcpyAsync(ffts_cu.data(), ccdata, sizeof(std::complex<float>) * R*n01d,
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        CUDA_RT_CALL(cudaFree(fdata));
        CUDA_RT_CALL(cudaFree(ccdata));
        CUFFT_CALL(cufftDestroy(plan));
        CUDA_RT_CALL(cudaStreamDestroy(stream));

        //
        CUFFT_CALL(cufftCreate(&plan0));
        CUFFT_CALL(cufftPlanMany(&plan0, fft_size.size(), fft_size.data(), nullptr, 1,
                                0, nullptr, 1, 0, CUFFT_R2C, imSize.nChannels));              
        CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
        CUFFT_CALL(cufftSetStream(plan0, stream0));
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&fdata0), sizeof(float) * imSize.nChannels*n01));
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&ccdata0), sizeof(std::complex<float>) * imSize.nChannels*n01d));    
                
        //
        CUFFT_CALL(cufftCreate(&plan1));
        CUFFT_CALL(cufftPlanMany(&plan1, fft_size.size(), fft_size.data(), nullptr, 1,
                                0, nullptr, 1, 0, CUFFT_C2R, R*imSize.nChannels));           
        CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
        CUFFT_CALL(cufftSetStream(plan1, stream1));
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&ccdata1), sizeof(std::complex<float>) * R*imSize.nChannels*n01d));
        CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&fdata1), sizeof(float) * R*imSize.nChannels*n01));
            

    }
    else if (method == 1)
    {
        std::vector<std::vector<float> > s(R, std::vector<float>(n01));
        std::vector<fftwf_plan> plans;
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

        fftw_init_threads();
        fftw_plan_with_nthreads(R);
        for(int r = 0; r < R; ++r)
        {
            plans.emplace_back(fftwf_plan_dft_r2c_2d(n0, n1,
                        s[r].data(),
                        reinterpret_cast<fftwf_complex*>(ffts[r].data()),
                        FFTW_ESTIMATE));
        }
        #pragma omp parallel for
        for(int r = 0; r < R; ++r)
        {
            fftwf_execute(plans[r]);
        }
        for(int r = 0; r < R; ++r)
        {
            fftwf_destroy_plan(plans[r]);
        }
        plans.clear();


        fftw_plan_with_nthreads(imageSize.nChannels);
        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            plans0_wf.emplace_back(fftwf_plan_dft_r2c_2d(n0, n1,
                    noise_gs[c].data(),
                    reinterpret_cast<fftwf_complex*>(dft_rec_noise[c].data()),
                    FFTW_ESTIMATE));
            for(int r = 0; r < R; ++r)
            {
                plans1_wf[c].emplace_back(fftwf_plan_dft_c2r_2d(n0, n1,
                        reinterpret_cast<fftwf_complex*>(dft_conv_noise[c][r].data()),
                        filtered_by_dfts[c][r].data(),
                        FFTW_ESTIMATE));
            }

        }		

    }
    else if (method == 2)
    {

        size_t fsize=sizeof(float);
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            stride0[i]=fsize;
            fsize*=shape[i];
        }
        size_t cfsize=sizeof(std::complex<float>);
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            stride1[i]=cfsize;
            cfsize*=n1_dft;
        }
        for (size_t i=0; i<shape.size(); ++i)
            axes.push_back(i);

        std::vector<std::vector<float> > s(R, std::vector<float>(n01)); 
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

            pocketfft::r2c(shape, stride0, stride1, axes, pocketfft::detail::FORWARD,
                        s[r].data(), ffts[r].data(), 1.f, 1);

        }



    }
    else
    {
        /* code */
    }
    

}

NFADetectionImpl::~NFADetectionImpl()
{

    if (method == 0)
    {
        CUDA_RT_CALL(cudaFree(fdata0));
        CUDA_RT_CALL(cudaFree(ccdata0));

        CUDA_RT_CALL(cudaFree(fdata1));
        CUDA_RT_CALL(cudaFree(ccdata1));

        // ？？？
        CUFFT_CALL(cufftDestroy(plan0));
        CUDA_RT_CALL(cudaStreamDestroy(stream0));  

        CUFFT_CALL(cufftDestroy(plan1));
        CUDA_RT_CALL(cudaStreamDestroy(stream1));  

        CUDA_RT_CALL(cudaDeviceReset());     
    }   
    else if (method == 1)
    {
        
        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            fftwf_destroy_plan(plans0_wf[c]);
            for(int r = 0; r < R; ++r)
            {
                fftwf_destroy_plan(plans1_wf[c][r]);
            }	
        }
        fftw_cleanup_threads();

    }
    else if (method == 2)
    {

    }
    else
    {
        /* code */
    }
    
}

void NFADetectionImpl::applyDetection(const std::vector<float>& residual, int HALFPATCHSIZE,  
                                  std::vector<float>& pixelNFA, std::vector<float>& radiusNFA)
{

    if (method == 0)
    {

        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            for(int x = 0; x < n0; ++x)
                for(int y = 0; y < n1; ++y)
                    noise_gs_cu[c*n01 + y + x*n1] = residual[x*imageSize.nChannels + y*imageSize.wc + c];
        }	
        CUDA_RT_CALL(cudaMemcpyAsync(fdata0, noise_gs_cu.data(), sizeof(float) * imageSize.nChannels*n01,
                                     cudaMemcpyHostToDevice, stream0));
        CUFFT_CALL(cufftExecR2C(plan0, fdata0, ccdata0));
        CUDA_RT_CALL(cudaMemcpyAsync(dft_rec_noise_cu.data(), ccdata0, sizeof(std::complex<float>) * imageSize.nChannels*n01d,
                                     cudaMemcpyDeviceToHost, stream0));
        CUDA_RT_CALL(cudaStreamSynchronize(stream0));   


        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            //#pragma omp parallel for
            for(int r = 0; r < R; ++r)
            {	
                for(int x = 0; x < n0; ++x)
                for(int y = 0; y < n1_dft; ++y)
                {
                    dft_conv_noise_cu[c*rn01d + r*n01d + y + x * n1_dft] = ffts_cu[r*n01d + y + x * n1_dft] * dft_rec_noise_cu[c*n01d + y + x * n1_dft];
                }

            }
        }
        CUDA_RT_CALL(cudaMemcpyAsync(ccdata1, dft_conv_noise_cu.data(), sizeof(std::complex<float>) * R*imageSize.nChannels*n01d,
                                     cudaMemcpyHostToDevice, stream1));
        CUFFT_CALL(cufftExecC2R(plan1, ccdata1, fdata1));
        CUDA_RT_CALL(cudaMemcpyAsync(filtered_by_dfts_cu.data(), fdata1, sizeof(float) * R*imageSize.nChannels*n01,
                                     cudaMemcpyDeviceToHost, stream1));
        CUDA_RT_CALL(cudaStreamSynchronize(stream1));

        std::vector<std::vector<float>> pixelNFAs(imageSize.nChannels, std::vector<float>(pixelNFA.size()));
        std::vector<std::vector<float>> radiusNFAs(imageSize.nChannels, std::vector<float>(pixelNFA.size()));

        //#pragma omp parallel for
        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            std::vector<double> sigmaphi(R);
            #pragma omp parallel for
            for(int r = 0; r < R; ++r)
            {	
                std::vector<float> backup(n01);
                for(int x = 0; x < n0; ++x) 
                    for(int y = 0; y < n1; ++y) 
                        backup[y + x*n1] = filtered_by_dfts_cu[c*rn01 + r*n01 + y + x * n1];

                for(int x = 0; x < n0; ++x) 
                {
                    int xs = (x + (n0)/2) % n0;
                    for(int y = 0; y < n1; ++y) 
                    {
                        int ys = (y + (n1)/2) % n1;

                        filtered_by_dfts_cu[c*rn01 + r*n01 + ys + xs*n1] = backup[y + x * n1];
                    }
                }

                // the residual is supposed to be centered. This doesn't change after the application of a Gaussian 
                sigmaphi[r] = 0.;
                for(int x = 0, ii = 0; x < n0; ++x)
                    for(int y = 0; y < n1; ++y, ++ii)
                        sigmaphi[r] += (filtered_by_dfts_cu[c*rn01 + r*n01 + y + n1*x] * filtered_by_dfts_cu[c*rn01 + r*n01 + y + n1*x] - sigmaphi[r]) / (float)(ii + 1);
                sigmaphi[r] /= n02_12;
                sigmaphi[r] = std::sqrt(std::max(sigmaphi[r], 0.));

                for(int x = HALFPATCHSIZE; x < (imageSize.width-HALFPATCHSIZE); ++x)
                for(int y = HALFPATCHSIZE; y < (imageSize.height-HALFPATCHSIZE); ++y)
                {
                    float temp;
                    temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts_cu[c*rn01 + r*n01 + y + n1*x] / (SQRT2*sigmaphi[r]*n01);
                    temp = (std::abs(temp) > 26) ? -100000000 : std::log(mn*std::erfc(std::abs(temp)))/tolog10;
                    if(temp < pixelNFA[x + y*imageSize.width])
                    {
                        pixelNFAs[c][x + y*imageSize.width] = temp;
                        radiusNFAs[c][x + y*imageSize.width] = (r+1)*l;
                    }
                    if (pixelNFAs[c][x + y*imageSize.width] < pixelNFA[x + y*imageSize.width])
                    {
                        pixelNFA[x + y*imageSize.width] = pixelNFAs[c][x + y*imageSize.width];
                        radiusNFA[x + y*imageSize.width] = radiusNFAs[c][x + y*imageSize.width];
                    }
                    pixelNFA[x + y*imageSize.width] += logc;

                }

            }

        }


    }
    else if (method == 1)
    {

        std::vector<std::vector<float>> pixelNFAs(imageSize.nChannels, std::vector<float>(pixelNFA.size()));
        std::vector<std::vector<float>> radiusNFAs(imageSize.nChannels, std::vector<float>(pixelNFA.size()));
        

        //#pragma omp parallel for
        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            for(int x = 0; x < n0; ++x)
                for(int y = 0; y < n1; ++y)
                    noise_gs[c][y + x*n1] = residual[x*imageSize.nChannels + y*imageSize.wc + c];

            fftwf_execute(plans0_wf[c]);

            std::vector<double> sigmaphi(R);
            #pragma omp parallel for
            for(int r = 0; r < R; ++r)
            {	

                for(int x = 0; x < n0; ++x)
                for(int y = 0; y < n1_dft; ++y)
                {
                    dft_conv_noise[c][r][y + x * n1_dft] = ffts[r][y + x * n1_dft] * dft_rec_noise[c][y + x * n1_dft];
                }

                fftwf_execute(plans1_wf[c][r]);

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
                sigmaphi[r] = std::sqrt(std::max(sigmaphi[r], 0.));

                for(int x = HALFPATCHSIZE; x < (imageSize.width-HALFPATCHSIZE); ++x)
                for(int y = HALFPATCHSIZE; y < (imageSize.height-HALFPATCHSIZE); ++y)
                {
                    float temp;
                    temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts[c][r][y + n1*x] / (SQRT2*sigmaphi[r]*float(n01));
                    temp = (std::abs(temp) > 26) ? -100000000 : std::log(mn*std::erfc(std::abs(temp)))/tolog10;
                    if(temp < pixelNFA[x + y*imageSize.width])
                    {
                        pixelNFAs[c][x + y*imageSize.width] = temp;
                        radiusNFAs[c][x + y*imageSize.width] = (r+1)*l;
                    }
                    if (pixelNFAs[c][x + y*imageSize.width] < pixelNFA[x + y*imageSize.width])
                    {
                        pixelNFA[x + y*imageSize.width] = pixelNFAs[c][x + y*imageSize.width];
                        radiusNFA[x + y*imageSize.width] = radiusNFAs[c][x + y*imageSize.width];
                    }
                    pixelNFA[x + y*imageSize.width] += logc;
                }

            }

        }


    }
    else if (method == 2)
    {
        std::vector<std::vector<float>> pixelNFAs(imageSize.nChannels, std::vector<float>(pixelNFA.size()));
        std::vector<std::vector<float>> radiusNFAs(imageSize.nChannels, std::vector<float>(pixelNFA.size()));

        //#pragma omp parallel for
        for(int c = 0; c < imageSize.nChannels; ++c)
        {
            for(int x = 0; x < n0; ++x)
                for(int y = 0; y < n1; ++y)
                    noise_gs[c][y + x*n1] = residual[x*imageSize.nChannels + y*imageSize.wc + c];

            // auto start = std::chrono::high_resolution_clock::now();
            pocketfft::r2c(shape, stride0, stride1, axes, pocketfft::detail::FORWARD,
                           noise_gs[c].data(), dft_rec_noise[c].data(), 1.f, 1);
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
                sigmaphi[r] = std::sqrt(std::max(sigmaphi[r], 0.));

                for(int x = HALFPATCHSIZE; x < (imageSize.width-HALFPATCHSIZE); ++x)
                for(int y = HALFPATCHSIZE; y < (imageSize.height-HALFPATCHSIZE); ++y)
                {
                    float temp;
                    temp = (sigmaphi[r] < 1e-8f) ? 1. : filtered_by_dfts[c][r][y + n1*x] / (SQRT2*sigmaphi[r]*n01);
                    temp = (std::abs(temp) > 26) ? -100000000 : std::log(mn*std::erfc(std::abs(temp)))/tolog10;
                    if(temp < pixelNFA[x + y*imageSize.width])
                    {
                        pixelNFAs[c][x + y*imageSize.width] = temp;
                        radiusNFAs[c][x + y*imageSize.width] = (r+1)*l;
                    }

                    if (pixelNFAs[c][x + y*imageSize.width] < pixelNFA[x + y*imageSize.width])
                    {
                        pixelNFA[x + y*imageSize.width] = pixelNFAs[c][x + y*imageSize.width];
                        radiusNFA[x + y*imageSize.width] = radiusNFAs[c][x + y*imageSize.width];
                    }
                    pixelNFA[x + y*imageSize.width] += logc;

                }

            }

        }

    }
    else
    {
        /* code */
    }
    
}


