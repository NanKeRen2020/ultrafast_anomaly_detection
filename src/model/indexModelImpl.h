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

#ifndef INDEXMODELIMPL_H_INCLUDED
#define INDEXMODELIMPL_H_INCLUDED

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

#include "params.h"
#include "modelImpl.h"
#include "model.h"



/**
 * @brief Generic step of the NL-Bayes denoising (could be the first or the second).
 *
 * @param i_imNoisy: contains the noisy video;
 * @param io_imBasic: will contain the denoised image after the first step (basic estimation);
 * @param o_imFinal: will contain the denoised image after the second step;
 * @param p_params: parameters of the method, contains:
 *			- sigma: standard deviation of the noise;
 *			- sizePatch: size of patches (sizePatch x sizePatch);
 *			- nSimilarPatches: number of similar patches;
 *			- sizeSearchWindow: size of the neighbourhood searching window;
 *			- useHomogeneousArea: if true, the trick of using homogeneous area will be used;
 *			- gamma: parameter used to determine if we are in an homogeneous area;
 *			- maxAvoid: parameter used to stop the paste trick;
 *			- beta: parameter used during the estimate of the denoised patch;
 *			- coefBaricenter: parameter to determine if the covariance matrix inversion is correct;
 *			- isFirstStep: true if it's the first step of the algorithm which is needed;
 *			- verbose: if true, print some informations, do nothing otherwise.
 *
 * @return Percentage of processed groups over number of pixels.
 **/

struct IndexParams
{
	IndexParams()  { };
	IndexParams(int patch_size, int H, int similar_number, bool same_image, int index_type, 
	    int patch_stride, const std::string& model_path, const std::string& proto_path):
		patchSize(patch_size), nSimilarPatches(similar_number), h(H), same(same_image),
		indexType(index_type), patchStride(patch_stride), 
		modelPath(model_path), protoPath(proto_path) { }

	int patchStride;
	int indexType = 0;
	int nhsah = 2;
	int nprobe = 1;
	int continue_length;
	int patch_dim;

	unsigned patchSize;
	unsigned nSimilarPatches;
    bool same;
    float h;
	

	std::string protoPath;
	std::string modelPath;

};


/**
 * @brief Generic step of the NL-Bayes denoising (could be the first or the second).
 *
 * @param i_imNoisy: contains the noisy video;
 * @param io_imBasic: will contain the denoised image after the first step (basic estimation);
 * @param o_imFinal: will contain the denoised image after the second step;
 * @param p_params: parameters of the method, contains:
 *			- sigma: standard deviation of the noise;
 *			- sizePatch: size of patches (sizePatch x sizePatch);
 *			- nSimilarPatches: number of similar patches;
 *			- sizeSearchWindow: size of the neighbourhood searching window;
 *			- useHomogeneousArea: if true, the trick of using homogeneous area will be used;
 *			- gamma: parameter used to determine if we are in an homogeneous area;
 *			- maxAvoid: parameter used to stop the paste trick;
 *			- beta: parameter used during the estimate of the denoised patch;
 *			- coefBaricenter: parameter to determine if the covariance matrix inversion is correct;
 *			- isFirstStep: true if it's the first step of the algorithm which is needed;
 *			- verbose: if true, print some informations, do nothing otherwise.
 *
 * @return Percentage of processed groups over number of pixels.
 **/

class AnomalyDetectionImpl
{

public:

    AnomalyDetectionImpl(const ModelParams& model_params): m_layer_num(-1), 
		m_modelParams(model_params), m_dbpm(nullptr), m_fm(nullptr), m_pm(nullptr),
		m_ivfflat_gpu_index(nullptr), m_flat_gpu_index(nullptr), m_flat_index(nullptr),
		m_research_index(nullptr)  {  }

    AnomalyDetectionImpl(const IndexParams& index_params): m_layer_num(-1),
		m_indexParams(index_params), m_dbpm(nullptr), m_fm(nullptr), m_pm(nullptr),
		m_ivfflat_gpu_index(nullptr), m_flat_gpu_index(nullptr), m_flat_index(nullptr),
		m_research_index(nullptr) {  }

    // AnomalyDetection(const AnomalyDetection& anomalyDetection);
    // AnomalyDetection& operator=(const AnomalyDetection& anomalyDetection);
  
	bool buildTreeModel(const cv::Mat& ref_image,  const TreeParams& treeParams,
		                int layer_num = -1, const std::vector<std::vector<int>> indexes = {{}});

	bool addIndexModel(const cv::Mat& ref_image, int layer_num = -1, 
	                   const std::vector<std::vector<int>> indexes = {{}});


	std::vector<std::tuple<float, int, int, int>> applyTreeDetection(const cv::Mat& image, 
	    float nfa_thresh, int nfa_method, int pow_layer, int R = 2, int l = 2);
			
	std::vector<std::tuple<float, int, int, int>> applyIndexDetection(const cv::Mat& image,
	    float nfa_thresh, int nfa_method, int pow_layer, float max_dis, int R = 2, int l = 2);

    ~AnomalyDetectionImpl();
	
private:

    void updateModel(const std::vector<faiss::idx_t>& patch_nns,
        std::vector<float>& patch_dis, const std::vector<int>& excludes, 
        int similar_num, int patch_dim, int h, int n, int patch_size,
		cv::Size query_size, int query_channel, int nn,
		const std::vector<std::tuple<int, int, int>>& queryPatchIndex,
		std::vector<float>& model);


    std::vector<float> m_reference;
    std::vector<float> m_residual;
	std::vector<ImageSize> m_refSizes;
	ImageSize m_imSize;
	ModelParams m_modelParams;
	IndexParams m_indexParams;
	std::vector<std::vector<int>> m_indexes;
	int m_layer_num;

    std::vector<std::tuple<int, int, int>> m_patchesWithIndex;
    std::vector<int> m_patchesToPixel;
	std::map<int, int> m_patchesPixelIndex;

	std::vector<PartitionTree*> m_trees; 

	std::shared_ptr<DatabasePatchManager> m_dbpm;
	std::shared_ptr<ForestManager> m_fm;
	std::shared_ptr<ImagePatchManager> m_pm;

	std::shared_ptr<faiss::gpu::GpuIndexIVFFlat> m_ivfflat_gpu_index;
    std::shared_ptr<faiss::gpu::GpuIndexFlat> m_flat_gpu_index;
	std::shared_ptr<faiss::IndexFlatL2> m_flat_index;
	std::shared_ptr<faiss::Index> m_research_index;
	

};


#endif // MODELIMPL_H_INCLUDED
