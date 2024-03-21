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

#include "modelImpl.h"
#include "../Utilities/comparators.h"
#include "../Utilities/nfa.h"


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

static cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &data)
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



cv::Mat get_nn_output(const cv::Mat& image, const std::string& proto_path, 
                      const std::string& model_path, int layer_num, 
                      const std::vector<int>& scale_indexes)
{
 
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(proto_path, model_path);
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 255.0, cv::Size(image.cols, image.rows), cv::Scalar(0.40760392, 0.45795686, 0.48501961), false, false);
    net.setInput(blob);
    std::vector<cv::String> layer_names = net.getLayerNames();

    // std::cout << layer_names << std::endl;
    // layer_names = std::vector<cv::String>{"conv1_1", "conv1_2", "pool1","conv2_1", "conv2_2", "pool2",
    //                                       "conv3_1", "conv3_2", "conv3_3", "conv3_4", "pool3",
    //                                       "conv4_1", "conv4_2", "conv4_3", "conv4_4"};
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = net.forward(layer_names[layer_num]);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "===>nn forward time: " << diff.count() << std::endl;
    // cv::Mat result1 = net.forward(layer_names[0]);
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
                      const std::string& proto_path = std::string(), 
                      const std::string& model_path = std::string())
{
    assert(image.cols >= 32 && image.rows >= 32);
    
    if ( layer_num >= 0 )
    {

        // get nnnetwork output
        auto start = std::chrono::high_resolution_clock::now();
        //static UpdateProtoFile updateProtoFile(image.size(), image.channels(), proto_path);
        static NNForward nnForward(image.size(), image.channels(), proto_path, model_path);
        //image = get_nn_output(image, proto_path, model_path, layer_num, indexes);
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



//------------------------------------------------- AnomalyDetection --------------------------------
//
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
    cv::dnn::blobFromImage(image, blob, 255.0, cv::Size(image.cols, image.rows), cv::Scalar(0.40760392, 0.45795686, 0.48501961), false, false);
    net.setInput(blob);
    std::vector<cv::String> layer_names = net.getLayerNames();  
    return net.forward(layer_names[layer_num]);
}

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


void AnomalyDetectionImpl::updateModel(const std::vector<faiss::idx_t>& patch_nns,
    std::vector<float>& patch_dis, const std::vector<int>& excludes, 
    int similar_num, int patch_dim, int h, int n, int patch_size, 
    cv::Size query_size, int query_channel, int nn, 
    const std::vector<std::tuple<int, int, int>>& queryPatchIndex,
    std::vector<float>& model)
{
    float norm_factor = 0.f;
    for (int m = 0; m < similar_num; ++m)
    {
        if (std::find(excludes.begin(), excludes.end(), m) == excludes.end())
        {
            patch_dis[m + nn * similar_num] = 
                exp(-1.0*patch_dis[m + nn * similar_num]/(float(h*patch_dim)));
            norm_factor += patch_dis[m + nn * similar_num];
        }

    }

    if (norm_factor > 0)
        norm_factor = 1.f / norm_factor;
    else
    {
        for (int m = 0; m < similar_num; m++)
        {
            patch_dis[m + nn * similar_num] = 1.0;
        }
        norm_factor = 1.f / (similar_num);
    } 

    for (int m = 0; m < similar_num; ++m)
    {
        if (std::find(excludes.begin(), excludes.end(), m) == excludes.end())
            patch_dis[m + nn * similar_num] *= norm_factor;
    }

    int offset = 0;
    for (unsigned y = 0; y < patch_size; y++)
    for (unsigned x = 0; x < patch_size; x++)
    {
        for (unsigned c = 0; c < query_channel; c++)
        {
            offset = y*query_size.width*query_channel + x*query_channel + c;
            // if (excludes.size() == similar_num)
            // {
            //     // model[std::get<2>(queryPatchIndex[n]) + offset] = 
            //     //     ref_datas[m_patchesToPixel[patch_nns[exclude[k - 2] + n * k]] + offset];   
            //     // //   continue;              
            // }
            {
                for (unsigned m = 0; m < similar_num; ++m)
                {
                    if (std::find(excludes.begin(), excludes.end(), m) == excludes.end())
                    {
                        model[std::get<2>(queryPatchIndex[n]) + offset] += patch_dis[m + nn * similar_num]*
                            m_reference[m_patchesToPixel[patch_nns[m + nn * similar_num]] + offset];

                    }

                }

            }
        }

    }

}

AnomalyDetectionImpl::~AnomalyDetectionImpl()
{
	// ! Delete trees
	for(auto& tree_ptr: m_trees)
    {
        delete tree_ptr; 
        tree_ptr = nullptr;
    }

}

bool AnomalyDetectionImpl::buildTreeModel(
        const cv::Mat& ref_image, const TreeParams& treeParams, int layer_num, 
        const std::vector<std::vector<int>> indexes)
{
    assert(!ref_image.empty());
    
    m_indexes = indexes;
    m_layer_num = layer_num;

    cv::Mat refImage = ref_image.clone();
    std::vector<int> scale_indexes;
    if (!m_indexes.empty()) scale_indexes = m_indexes[0];
    // convert image to float vector
    ImageSize refSize;    
    m_reference = get_candidate(refImage, m_layer_num, m_modelParams.sizePatch, scale_indexes, 
                                refSize, m_modelParams.protoPath, m_modelParams.modelPath);
    m_refSizes.emplace_back(refSize);


	const unsigned patch_num = m_modelParams.nSimilarPatches;
    std::vector<std::tuple<float, unsigned, float> > index(patch_num);
    
	const unsigned sPx = m_modelParams.sizePatch;
    const int stride = m_modelParams.patchStride;

	//! Create the VPtree forest
    m_pm.reset(new ImagePatchManager(m_reference, refSize, sPx, stride));


#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
	shared(m_trees) 
#endif
	for(int i = 0; i < m_modelParams.nbTrees; ++i)
		m_trees.push_back(new VPTree(*m_pm, treeParams));

    // vptree manager
    m_fm.reset(new ForestManager(*m_pm, m_trees, m_modelParams.nSimilarPatches, m_modelParams));

    // image patch manager with vptree
    m_dbpm.reset( new DatabasePatchManager(m_reference, refSize, sPx, stride));

    return true;

}

bool AnomalyDetectionImpl::addIndexModel(const cv::Mat& refImage, int layer_num,
                                         const std::vector<std::vector<int>> indexes)
{
    assert(!refImage.empty());

    cv::Mat ref_image = refImage.clone();
    m_indexes = indexes;
    std::vector<int> scale_indexes;
    if (!m_indexes.empty()) scale_indexes = m_indexes[0];
    m_layer_num = layer_num;
    // convert image to float vector
    ImageSize imSize;
    std::vector<float> ref = get_candidate(ref_image, m_layer_num, m_indexParams.patchSize, scale_indexes, 
                                           imSize, m_indexParams.protoPath, m_indexParams.modelPath);

    m_reference.insert(m_reference.end(), ref.begin(), ref.end());
    m_refSizes.emplace_back(imSize);

    int patch_number = ((ref_image.cols - m_indexParams.patchSize)/m_indexParams.patchStride + 1) * 
        ((ref_image.rows - m_indexParams.patchSize)/m_indexParams.patchStride + 1);
    m_indexParams.patch_dim = m_indexParams.patchSize*m_indexParams.patchSize*ref_image.channels();
    m_indexParams.continue_length = m_indexParams.patchSize*ref_image.channels();
        
    std::vector<float> patchesDatabase;
    int pp, n;
    pp = n = 0;
    for (int j = 0; j < ref_image.rows - m_indexParams.patchSize + 1; j += m_indexParams.patchStride)
    {
        for (int i = 0; i < ref_image.cols - m_indexParams.patchSize + 1; i += m_indexParams.patchStride)
        {

            int ij3 = (i + j*ref_image.cols)*ref_image.channels();
            for (unsigned hy = 0; hy < m_indexParams.patchSize; hy++)
            {
                pp = ij3 + hy*ref_image.channels()*ref_image.cols;
                patchesDatabase.insert(patchesDatabase.end(), ref.begin() + pp, 
                                ref.begin() + pp + m_indexParams.continue_length);
            }


            m_patchesToPixel.emplace_back(ref_image.channels()*(i + j*ref_image.cols));
            m_patchesWithIndex.emplace_back(std::tuple<int, int, int>(n, i, j));

            // 默认第一张图可能为相同图
            if (m_refSizes.size() == 1) m_patchesPixelIndex[m_patchesToPixel[n]] = n;
           
            ++n;

        }
    }


    // faiss search with gpu ivf_flat
    int nbits_subq = int(std::log2(patch_number + 1) / 2);     // good choice in general
    int ncentroids = 1 << (m_indexParams.nhsah * nbits_subq); // total # of centroids
    ncentroids = int(4 * std::sqrt(patch_number));
    ncentroids = 1024;

    if (m_indexParams.indexType == 0)
    {
        faiss::gpu::StandardGpuResources resources;
        resources.noTempMemory();
        faiss::gpu::GpuIndexIVFFlatConfig config_ivf_flat;
        config_ivf_flat.device = 0;
        m_ivfflat_gpu_index.reset( new faiss::gpu::GpuIndexIVFFlat(
                &resources, m_indexParams.patch_dim, ncentroids, faiss::METRIC_L2, config_ivf_flat));
        m_ivfflat_gpu_index->nprobe = m_indexParams.nprobe;

        auto s = std::chrono::high_resolution_clock::now();
        // 已添加item说明已训练完成
        if (!m_ivfflat_gpu_index->ntotal)
        m_ivfflat_gpu_index->train(patch_number, patchesDatabase.data());
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "gpu ivfflat train time: " << diff.count() 
                  << " index ntotal: " << m_ivfflat_gpu_index->ntotal << "\n";  

        s = std::chrono::high_resolution_clock::now();
        m_ivfflat_gpu_index->add(patch_number, patchesDatabase.data());
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        std::cout << "gpu ivfflat add time: " << diff.count() 
                  << " index ntotal: " << m_ivfflat_gpu_index->ntotal << "\n"; 
    }
    else if (m_indexParams.indexType == 1)
    {
        faiss::gpu::GpuIndexFlatConfig config_gpu_flat;
        config_gpu_flat.device = 0;
        faiss::gpu::StandardGpuResources resources;
        resources.noTempMemory();
        m_flat_gpu_index.reset(new faiss::gpu::GpuIndexFlat(&resources,  m_indexParams.patch_dim, faiss::METRIC_L2, config_gpu_flat));
        
        auto s = std::chrono::high_resolution_clock::now();
        m_flat_gpu_index->add(patch_number, patchesDatabase.data());
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "gpu patch flat add time: " << diff.count() 
                  << " index ntotal: " << m_flat_gpu_index->ntotal << "\n"; 
    }
    else
    {
        m_flat_index.reset( new faiss::IndexFlatL2( m_indexParams.patch_dim));

        auto s = std::chrono::high_resolution_clock::now();
        m_flat_index->add(patch_number, patchesDatabase.data());
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "patch flat add time: " << diff.count() 
                << " index ntotal: " << m_flat_index->ntotal << "\n"; 

    }

    {
        m_research_index.reset( new faiss::IndexFlatL2( m_indexParams.patch_dim));
        m_research_index->add(patch_number, patchesDatabase.data());


        // faiss::gpu::GpuIndexFlatConfig config_gpu_flat;
        // config_gpu_flat.device = 0;
        // faiss::gpu::StandardGpuResources resources;
        // resources.noTempMemory();
        // m_research_index.reset(new faiss::gpu::GpuIndexFlat(&resources,  m_indexParams.patch_dim, faiss::METRIC_L2, config_gpu_flat));
        
        // auto s = std::chrono::high_resolution_clock::now();
        // m_research_index->add(patch_number, patchesDatabase.data());
        // auto e = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> diff = e - s;
        // std::cout << "research gpu patch flat add time: " << diff.count() 
        //           << " index ntotal: " << m_research_index->ntotal << "\n"; 
    }



    return true;

}


std::vector<std::tuple<float, int, int, int>> AnomalyDetectionImpl::applyTreeDetection(
        const cv::Mat& srcImage, float nfa_thresh, int nfa_method, int pow_layer,
        int R, int l)
{
    auto start = std::chrono::high_resolution_clock::now();
    assert(!srcImage.empty());

    cv::Mat query_image = srcImage.clone();
    ImageSize imSize;
    std::vector<int> scale_indexes;
    if (!m_indexes.empty()) scale_indexes = m_indexes[0];
    std::vector<float> candidate = get_candidate(query_image, m_layer_num, m_modelParams.sizePatch, scale_indexes, imSize, 
                                                 m_modelParams.protoPath, m_modelParams.modelPath);
    m_imSize = imSize;

    m_dbpm->setCurrentImage(candidate, imSize);

    m_fm->updatePM(m_dbpm.get());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "===>tree search update time: " << diff.count() << std::endl;


    const unsigned patch_num = m_modelParams.nSimilarPatches;
    std::vector<std::tuple<float, unsigned, float> > index(patch_num);
	const unsigned sPx = m_modelParams.sizePatch;
    const int stride = m_modelParams.patchStride;

    // find most similar patch for every patch
    LocalRefinement lpp(*m_dbpm, *m_fm, patch_num, m_modelParams);
    std::vector<float> model(imSize.whc);

#ifdef _OPENMP
#pragma omp parallel for num_threads(12) schedule(dynamic) \
        shared(lpp, model) \
        firstprivate(index)
#endif
    for (unsigned py = 0; py < imSize.height - sPx + 1; py+=sPx)
        for (unsigned px = 0; px < imSize.width - sPx + 1; px+=sPx)
        {
            const unsigned ij  = px + imSize.width*py;
            const unsigned ij3 = (px + imSize.width*py)*imSize.nChannels;

            //! Search for similar patches around the reference one
            unsigned nSimP = lpp.estimateSimilarPatches(index, ij3, m_modelParams.same);

            float norm_factor = 0.f;
            for (unsigned k = 0; k < nSimP; ++k)
            {
                std::get<0>(index[k]) = exp(-std::get<0>(index[k])/m_modelParams.h);
                norm_factor += std::get<0>(index[k]);
            }
            if(norm_factor > 0)
                norm_factor = 1.f / norm_factor;
            else
            {
                for(int i = 0; i < nSimP; ++i)
                    std::get<0>(index[i]) = 1;
                norm_factor = 1.f / nSimP;
            }
            for (unsigned k = 0; k < nSimP; ++k)
            {
                std::get<0>(index[k]) *= norm_factor;
            }
            
            // Compute the average patch while aggregating it.
            int offset;
            for (unsigned y = 0; y < sPx; y++)
            for (unsigned x = 0; x < sPx; x++)
            {
                for (unsigned c = 0; c < imSize.nChannels; c++)
                {
                    offset = y*imSize.width*imSize.nChannels + x*imSize.nChannels + c;
                    for (unsigned k = 0; k < nSimP; ++k)
                    model[ij3 + offset] += std::get<0>(index[k]) * m_reference[std::get<1>(index[k]) + offset];
                        
                }

            }
            
        }


    computeDiff(candidate, model, m_residual); 

    auto end1 = std::chrono::high_resolution_clock::now();
    diff = end1 - end;
    std::cout << "===>get model time: " << diff.count() << std::endl;

    std::vector<float> pixelNFA(m_imSize.width*m_imSize.height);
    std::vector<float> radiusNFA(m_imSize.width*m_imSize.height);
    ImageSize nfaSize;
    nfaSize.width = m_imSize.width;
    nfaSize.height = m_imSize.height;
    nfaSize.nChannels = 1;
    nfaSize.wh = nfaSize.width * nfaSize.height;
    nfaSize.whc = nfaSize.width * nfaSize.height * nfaSize.nChannels;

    static NFADetectionImpl m_nfa_detection(m_imSize, R, l, query_image.cols, query_image.rows, nfa_method);
    auto end3 = std::chrono::high_resolution_clock::now();
    m_nfa_detection.applyDetection(m_residual, m_modelParams.sizePatch/2, pixelNFA, radiusNFA);
    auto end4 = std::chrono::high_resolution_clock::now();
    diff = end4 - end3;
    std::cout << "===>nfa detection time: " << diff.count() << std::endl;


    std::vector<std::tuple<float, int, int, int>> detections;
    detectNFA(pixelNFA, radiusNFA, nfaSize, detections, 
                1, pow_layer, nfa_thresh, query_image.size());

    auto end2 = std::chrono::high_resolution_clock::now();
    diff = end2 - end1;
    std::cout << "===>detection time: " << diff.count() << std::endl;

    return detections;
}

std::vector<std::tuple<float, int, int, int>> AnomalyDetectionImpl::applyIndexDetection(
    const cv::Mat& src_image, float nfa_thresh, int nfa_method, int pow_layer, 
    float max_dis, int R, int l)
{
    auto start = std::chrono::high_resolution_clock::now();
    assert(!src_image.empty());
    cv::Mat query_image = src_image.clone();

    std::vector<int> scale_indexes;
    if (!m_indexes.empty()) scale_indexes = m_indexes[0];
    std::vector<float> query_datas = get_candidate(query_image, m_layer_num, m_indexParams.patchSize, scale_indexes,  
                                                   m_imSize, m_indexParams.protoPath, m_indexParams.modelPath);
    std::cout << "process info: " << query_image.size() << ", " 
              << query_image.channels() << std::endl;

    std::vector<float> queryDatabase;
    std::vector<std::tuple<int, int, int>> queryPatchIndex;
    int pp = 0;
    int nn = 0;
	std::map<int, int> patchesPixelIndex;   
    for (int j = 0; j < query_image.rows - m_indexParams.patchSize + 1; j += m_indexParams.patchSize)
    {
        for (int i = 0; i < query_image.cols - m_indexParams.patchSize + 1; i += m_indexParams.patchSize)
        {

            int ij = (i + j*query_image.cols);
            int ij3 = ij*query_image.channels();
            for (unsigned hy = 0; hy < m_indexParams.patchSize; hy++)
            {
                pp = ij3 + hy*query_image.channels()*query_image.cols;
                queryDatabase.insert(queryDatabase.end(), query_datas.begin() + pp, 
                            query_datas.begin() + pp + m_indexParams.continue_length);

            }
            queryPatchIndex.emplace_back(std::tuple<int, int, int>(i, j, ij3));
            int pixelIndex = query_image.channels()*(i + j*query_image.cols);
            patchesPixelIndex.insert({query_image.channels()*(i + j*query_image.cols), nn});

            ++nn;
        }
    }



    // find most similar patch for every patch
    int query_number = (query_image.cols / m_indexParams.patchSize ) * 
                       (query_image.rows / m_indexParams.patchSize);
    std::vector<faiss::idx_t> patch_nns(m_indexParams.nSimilarPatches * query_number);
    std::vector<float> patch_dis(m_indexParams.nSimilarPatches * query_number);

    if (m_indexParams.indexType == 0)
    {
        m_ivfflat_gpu_index->search(query_number, queryDatabase.data(), m_indexParams.nSimilarPatches, 
            patch_dis.data(), patch_nns.data());

    }
    else if (m_indexParams.indexType == 1)
    {
        m_flat_gpu_index->search(query_number, queryDatabase.data(), m_indexParams.nSimilarPatches, 
            patch_dis.data(), patch_nns.data());
    }
    else
    {
        m_flat_index->search(query_number, queryDatabase.data(), m_indexParams.nSimilarPatches, 
            patch_dis.data(), patch_nns.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "===>index search time: " << diff.count() << std::endl;

    std::vector<float> model(query_image.cols*query_image.rows*query_image.channels(), 0.f);
    
    #pragma omp parallel for num_threads(12) schedule(dynamic) \
        shared(model, patch_dis)    
    for (int j = 0; j < query_image.rows - m_indexParams.patchSize + 1; j += m_indexParams.patchSize)
    for (int i = 0; i < query_image.cols - m_indexParams.patchSize + 1; i += m_indexParams.patchSize)
    {
        // 检测图像patch编码
        int n = patchesPixelIndex[query_image.channels()*(i + j*query_image.cols)];
        int kk = 0;
        int offset;
        std::vector<faiss::idx_t> ids;
        std::vector<int> exclude;
        ids.clear();
        for (int m = kk; m < m_indexParams.nSimilarPatches; m++)
        {
            // 获取搜索到的临近patch的index 
            if ( patch_nns[m + n * m_indexParams.nSimilarPatches] >= 0  && (patch_dis[m + n * m_indexParams.nSimilarPatches] < max_dis ||
                std::abs(std::get<1>(m_patchesWithIndex[patch_nns[m + n * m_indexParams.nSimilarPatches]]) - i) < m_indexParams.patchSize ||
                std::abs(std::get<2>(m_patchesWithIndex[patch_nns[m + n * m_indexParams.nSimilarPatches]]) - j) < m_indexParams.patchSize ))
                {
                    ids.emplace_back(patch_nns[m + n * m_indexParams.nSimilarPatches]);

                } 
        }

        // 搜索到的patch均不符合要求则重新搜索
        if (ids.size() > m_indexParams.nSimilarPatches - 1 && m_indexParams.same)
        {
            unsigned rangex[2];
            unsigned rangey[2];
            
            offset = 40;
            rangex[0] = std::max(0, i - offset);
            rangey[0] = std::max(0, j - offset);

            rangex[1] = std::min(int(m_refSizes[0].width - m_indexParams.patchSize), i + offset);
            rangey[1] = std::min(int(m_refSizes[0].height - m_indexParams.patchSize), j + offset);

            int min = offset/4;
            int max = offset*2 - min;
            ids.clear();
            // 搜索临域外的patch
            for (unsigned qy = rangey[0], dy = 0; qy <= rangey[1]; qy++, dy++)
            for (unsigned qx = rangex[0], dx = 0; qx <= rangex[1]; qx++, dx++)
            {
                if ( (dy > max && dx > max) || (dy > max && dx < min) ||
                    (dy < min && dx > max) || (dy < min && dx < min) )
                    ids.emplace_back(m_patchesPixelIndex[m_refSizes[0].nChannels*(qx + qy*m_refSizes[0].width)]);
            }

            faiss::IDSelectorBatch sel(ids.size(), ids.data());
            faiss::IDSelectorNot nsel(&sel);
            faiss::SearchParameters search_params;
            faiss::IVFSearchParameters ivf_params;
            search_params.sel = &sel;
            ivf_params.quantizer_params = &search_params;

            int nsp = m_indexParams.nSimilarPatches/2;
            std::vector<faiss::idx_t> patch_nns1(nsp * 1);
            std::vector<float> patch_dis1(nsp * 1);
            auto s = std::chrono::high_resolution_clock::now();
            m_research_index->search(1, queryDatabase.data() + n* m_indexParams.patch_dim, nsp, 
                patch_dis1.data(), patch_nns1.data(), &search_params);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cout << "flat research time: " << diff.count() << std::endl;

            std::vector<int> exclude1;
            int nn = 0;
            for (int m = kk; m < nsp; m++)
            {
                
                if ( patch_nns1[m + nn * nsp] >= 0 && (patch_dis1[m + nn * nsp] < max_dis ||
                    std::abs(std::get<1>(m_patchesWithIndex[patch_nns1[m + nn * nsp]]) - i) < m_indexParams.patchSize ||
                    std::abs(std::get<2>(m_patchesWithIndex[patch_nns1[m + nn * nsp]]) - j) < m_indexParams.patchSize ))
                    {
                        exclude1.emplace_back(m);
                        ids.emplace_back(patch_nns1[m + nn * nsp]);
                    }

            }

            updateModel(patch_nns1, patch_dis1, exclude1, nsp, m_indexParams.patch_dim, m_indexParams.h, n, 
                m_indexParams.patchSize, query_image.size(), query_image.channels(), nn, queryPatchIndex, model);
 
            continue;

        }

        updateModel(patch_nns, patch_dis, exclude, m_indexParams.nSimilarPatches, m_indexParams.patch_dim, 
                   m_indexParams.h, n, m_indexParams.patchSize, query_image.size(), query_image.channels(), 
                   n, queryPatchIndex, model);
    }

    computeDiff(query_datas, model, m_residual); 

    auto end1 = std::chrono::high_resolution_clock::now();
    diff = end1 - end;
    std::cout << "===>get model time: " << diff.count() << ", " << query_datas.size()
              << ", " << model.size() << ", " << m_residual.size() << std::endl;
    
    // convert_distribution(m_residual, m_imSize);
    auto end6 = std::chrono::high_resolution_clock::now();
    diff = end6 - end1;
    std::cout << "===>convert_distribution time: " << diff.count() << std::endl;


    std::vector<float> pixelNFA(m_imSize.width*m_imSize.height);
    std::vector<float> radiusNFA(m_imSize.width*m_imSize.height);
    ImageSize nfaSize;
    nfaSize.width = m_imSize.width;
    nfaSize.height = m_imSize.height;
    nfaSize.nChannels = 1;
    nfaSize.wh = nfaSize.width * nfaSize.height;
    nfaSize.whc = nfaSize.width * nfaSize.height * nfaSize.nChannels;
    
    
    auto end5 = std::chrono::high_resolution_clock::now();
    static NFADetectionImpl nfa_detection(m_imSize, R, l,  query_image.cols, query_image.rows, nfa_method);
    auto end3 = std::chrono::high_resolution_clock::now();
    diff = end3 - end5;
    std::cout << "===>nfa init time: " << diff.count() << std::endl;

    nfa_detection.applyDetection(m_residual, m_indexParams.patchSize/2, pixelNFA, radiusNFA);
    auto end4 = std::chrono::high_resolution_clock::now();
    diff = end4 - end3;
    std::cout << "===>nfa detection time: " << diff.count() << std::endl;
    

    
    std::vector<std::tuple<float, int, int, int>> detections;
    detectNFA(pixelNFA, radiusNFA, nfaSize, detections, 1, 
              pow_layer, nfa_thresh, query_image.size());

    auto end2 = std::chrono::high_resolution_clock::now();
    diff = end2 - end1;
    std::cout << "===>detection time: " << diff.count() << std::endl;

    return detections;
}



ModelParamsPtr::ModelParamsPtr(int patch_size, int h, int nb_trees, int similar_number, bool same, 
	    int patch_stride, const std::string& model_path, const std::string& proto_path): 
		modelParams(std::make_shared<ModelParams>(patch_size, h, nb_trees, similar_number, 
		same, patch_stride, model_path, proto_path)) { }



IndexParamsPtr::IndexParamsPtr(int patch_size, int h, int similar_number, bool same, 
    int index_type, int patch_stride, const std::string& model_path, const std::string& proto_path): 
    indexParams(std::make_shared<IndexParams>(patch_size, h, similar_number, 
    same, index_type, patch_stride, model_path, proto_path)) { }



TreeParamsPtr::TreeParamsPtr(float epsilon, int nbVP, int subS, int rand, const ModelParamsPtr& modelParamsPtr): 
		treeParams(std::make_shared<TreeParams>(epsilon, nbVP, subS, rand, *(modelParamsPtr.modelParams))) { }



AnomalyDetection::AnomalyDetection(const ModelParamsPtr& modelParams): 
    anomalyDetection(std::make_shared<AnomalyDetectionImpl>(*(modelParams.modelParams)))   { }

AnomalyDetection::AnomalyDetection(const IndexParamsPtr& indexParams):  
    anomalyDetection(std::make_shared<AnomalyDetectionImpl>(*(indexParams.indexParams)))   { }

bool AnomalyDetection::buildTreeModel(const cv::Mat& ref_image, const TreeParamsPtr& treeParams,
    int layer_num, const std::vector<std::vector<int>> indexes)
{
    return anomalyDetection->buildTreeModel(ref_image, *(treeParams.treeParams), layer_num, indexes);
}

bool AnomalyDetection::addIndexModel(const cv::Mat& ref_image, int layer_num, 
    const std::vector<std::vector<int>> indexes)
{
    return anomalyDetection->addIndexModel(ref_image, layer_num, indexes);
}


std::vector<std::tuple<float, int, int, int>> AnomalyDetection::applyTreeDetection(const cv::Mat& image, 
    float nfa_thresh, int nfa_method, int pow_layer, int R, int l)
{
    return anomalyDetection->applyTreeDetection(image, nfa_thresh, nfa_method, pow_layer, R, l);
}
        
std::vector<std::tuple<float, int, int, int>> AnomalyDetection::applyIndexDetection(const cv::Mat& image,
    float nfa_thresh, int nfa_method, int pow_layer, float max_dis, int R, int l)
{
    return anomalyDetection->applyIndexDetection(image, nfa_thresh, nfa_method, pow_layer, max_dis, R, l);
    // return std::vector<std::tuple<float, int, int, int>>();
}


