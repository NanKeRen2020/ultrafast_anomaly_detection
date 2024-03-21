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

#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>


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

/**
 * @brief Compute the final weighted aggregation.
 *
 * result: contain the aggregation, will contain the result at the end.
 * weight: contains the weights used during the aggregation;
 *
 * @return none.
 **/


struct ModelParams;
struct IndexParams;
struct TreeParams;
class AnomalyDetectionImpl;

struct ModelParamsPtr
{
	ModelParamsPtr(int patch_size, int h, int nb_trees, int similar_number, bool same, 
	               int patch_stride, const std::string& model_path, const std::string& proto_path);

	std::shared_ptr<ModelParams> modelParams; 
};

struct IndexParamsPtr
{
	IndexParamsPtr(int patch_size, int h, int similar_number, bool same, int index_type, 
	               int patch_stride, const std::string& model_path, const std::string& proto_path);

	std::shared_ptr<IndexParams> indexParams; 
};


struct TreeParamsPtr
{
	TreeParamsPtr(float epsilon, int nbVP, int subS, int rand, const ModelParamsPtr& modelParamsPtr);

	std::shared_ptr<TreeParams> treeParams; 
};


class AnomalyDetection
{

public:

    AnomalyDetection(const ModelParamsPtr& modelParams);

    AnomalyDetection(const IndexParamsPtr& indexParams);

  
	bool buildTreeModel(const cv::Mat& ref_image,  const TreeParamsPtr& treeParams,
		                int layer_num = -1, const std::vector<std::vector<int>> indexes = {{}});


	bool addIndexModel(const cv::Mat& ref_image, int layer_num = -1, 
                       const std::vector<std::vector<int>> indexes = {{}});

	std::vector<std::tuple<float, int, int, int>> applyTreeDetection(const cv::Mat& image, 
	    float nfa_thresh, int nfa_method, int pow_layer, int R = 2, int l = 2);
			
	std::vector<std::tuple<float, int, int, int>> applyIndexDetection(const cv::Mat& image,
	    float nfa_thresh, int nfa_method, int pow_layer, float max_dis, int R = 2, int l = 2);


    ~AnomalyDetection() { };
	
private:

	std::shared_ptr<AnomalyDetectionImpl> anomalyDetection;

};


#endif // MODEL_HPP_INCLUDED
