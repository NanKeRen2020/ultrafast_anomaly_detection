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

#include "model.h"
#include "modelImpl.h"
#include "indexModelImpl.h" 


//------------------------------------------------- AnomalyDetectionImpl --------------------------------

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

    static NFADetectionImpl nfa_detection(m_imSize, R, l, query_image.cols, query_image.rows, nfa_method);
    auto end3 = std::chrono::high_resolution_clock::now();
    nfa_detection.applyDetection(m_residual, m_modelParams.sizePatch/2, pixelNFA, radiusNFA);
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


//----------------------------------------------------- AnomalyDetection --------------------------------------------------------

ModelParamsPtr::ModelParamsPtr(int patch_size, int h, int nb_trees, int similar_number, bool same, 
	    int patch_stride, const std::string& model_path, const std::string& proto_path): 
		modelParams(std::make_shared<ModelParams>(patch_size, h, nb_trees, similar_number, 
		same, patch_stride, model_path, proto_path)) { }


TreeParamsPtr::TreeParamsPtr(float epsilon, int nbVP, int subS, int rand, const ModelParamsPtr& modelParamsPtr): 
		treeParams(std::make_shared<TreeParams>(epsilon, nbVP, subS, rand, *(modelParamsPtr.modelParams))) { }

IndexParamsPtr::IndexParamsPtr(int patch_size, int h, int similar_number, bool same, 
    int index_type, int patch_stride, const std::string& model_path, const std::string& proto_path): 
    indexParams(std::make_shared<IndexParams>(patch_size, h, similar_number, 
    same, index_type, patch_stride, model_path, proto_path)) { }

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
}



