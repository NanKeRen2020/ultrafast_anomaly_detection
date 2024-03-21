/*
 * Copyright (c) 2017, <shengiang8814@qq.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include "model/model.h"


using namespace std;


/*

./bin/anomaly_detection  /home/seeking/test_projects/anomaly_detection/models/deploy.prototxt /home/seeking/test_projects/anomaly_detection/models/VGG_normalised.caffemodel 1 8 16 /home/seeking/test_projects/anomaly_detection/datas/test1.png -1 0 /home/seeking/test_projects/anomaly_detection/results/detections_cpp.png /home/seeking/test_projects/anomaly_detection/datas/test1.png 2 2


./bin/anomaly_detection  /home/seeking/test_projects/anomaly_detection/models/deploy.prototxt /home/seeking/test_projects/anomaly_detection/models/VGG_normalised.caffemodel 1 8 16 /home/seeking/Downloads/dgb_data/ref.png -1 0 /home/seeking/test_projects/anomaly_detection/results/detections_cpp.png /home/seeking/Downloads/dgb_data/test0 2 2


*/


int main(int argc, char **argv)
{
   
    std::vector<int> nnlayer{0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 6};  

    int scale_num = atoi(argv[3]);
    int patch_size = atoi(argv[4]);
    int similar_num = atoi(argv[5]);   
    int layer_num = atoi(argv[7]);
    int nfa_thresh = atoi(argv[8]);
    int nfa_method = 2;
    if (argc > 11) nfa_method = atoi(argv[11]);
    int index = 0;  
    if (argc > 12) index = atoi(argv[12]);
    bool index_search = true;
    if (argc > 13) index = atoi(argv[13]);

    int pow_layer;
    if (layer_num >= 0)
    {
        pow_layer = std::pow(2, nnlayer[layer_num]);
    }
    else
       pow_layer = std::pow(2, nnlayer[0]);
    

    std::string out_path = std::string(argv[9]);
    std::string image_path = std::string(argv[10]);
    std::string ref_path = std::string(argv[6]);

    // define model parameters
    ModelParamsPtr modelParams(patch_size, 100, 4, similar_num, true, 1,  
                               std::string(argv[2]), std::string(argv[1]));
    // modelParams.sizePatch = patch_size;
    // modelParams.nbTrees = 4;
    // modelParams.h = 100;
    // modelParams.nSimilarPatches = num_patches;
    // modelParams.same = true;
    // modelParams.excR = 2*patch_size;    
    // modelParams.patchStride = 1;
    // modelParams.protoPath = std::string(argv[3]);
    // modelParams.modelPath = std::string(argv[4]);

    TreeParamsPtr treeParams(0, 1, 1000, 5, modelParams);
	// treeParams.epsilon = 0.; 
	// treeParams.nbVP = 1;
	// treeParams.subS = 1000;
	// treeParams.rand = 5;
    // treeParams.mprms = modelParams;

    IndexParamsPtr indexParams(patch_size, 100, similar_num, true, 0, 1,  
                               std::string(argv[2]), std::string(argv[1]));
    // indexParams.patchSize = patch_size;
    // indexParams.h = 100;
    // indexParams.nSimilarPatches = num_patches;
    // indexParams.same = true;
    // //indexParams.excR = 2*patch_size;    
    // indexParams.patchStride = 1;
    // indexParams.indexType = 0;
    // indexParams.protoPath = std::string(argv[3]);
    // indexParams.modelPath = std::string(argv[4]);

    float max_distance = 100;
    std::vector<std::vector<int>> indexes{{index}};
    cv::Mat ref_image = cv::imread(ref_path);

    // define detection object
    AnomalyDetection anomaly_detection(modelParams);
    anomaly_detection.buildTreeModel(ref_image, treeParams, layer_num, indexes);
    
    // AnomalyDetection anomaly_detection(indexParams);
    // anomaly_detection.addIndexModel(ref_image, layer_num, indexes);

    std::vector<cv::String> fn;
    boost::filesystem::path path(image_path);
    if (boost::filesystem::is_directory(path))
    {
        cv::glob(image_path, fn, false);
        std::string image_name = path.filename().string();
    }
    else
    {
        fn.emplace_back(image_path);
    }
    cv::Mat src_image, show_image;
    for (size_t i = 0; i < fn.size(); i++)
    {
        
        src_image = cv::imread(fn[i]);
        if (src_image.empty()) continue;
        std::cout << "===>process image name: " << fn[i] << std::endl;
        show_image = src_image.clone();
        // apply detection
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::tuple<float, int, int, int>> detections = 
            anomaly_detection.applyTreeDetection(src_image, nfa_thresh, nfa_method, pow_layer);
            // anomaly_detection.applyIndexDetection(src_image, nfa_thresh, nfa_method, pow_layer, max_distance);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "===>total detection time: " << diff.count() << std::endl;

        int run_count = 10;
        for (int i = 0; i < run_count; ++i)
        {
            // detections = anomaly_detection.applyTreeDetection(src_image, nfa_thresh, nfa_method, pow_layer);
            // detections = anomaly_detection.applyIndexDetection(src_image, nfa_thresh, nfa_method, pow_layer, max_distance);        
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        diff = end1 - end;
        std::cout << "===>average detection time: " << diff.count()/run_count 
                  << ", result number: " << detections.size() << std::endl;


        // show detection results
        if (!detections.empty())
        {
            std::vector<cv::Scalar> colors{cv::Scalar(0, 166, 255), cv::Scalar(0, 255, 0),
                                        cv::Scalar(255, 255, 0), cv::Scalar(255, 255, 255)};
            std::sort(detections.begin(), detections.end(), [](const std::tuple<float, int, int, int>& t1, 
                    const std::tuple<float, int, int, int>& t2) { return std::get<0>(t1) < std::get<0>(t2); });

            double nfa_best = std::get<0>(detections[0]);
            for (int i = 0; i < detections.size(); ++i)
            {
                double nfa = std::get<0>(detections[i]);
                
                if (nfa == nfa_best || std::isinf(nfa) || std::isnan(nfa) )
                {
                    cv::circle(show_image, cv::Point(std::get<1>(detections[i]), std::get<2>(detections[i])), 
                            std::get<3>(detections[i]), cv::Scalar(0, 0, 139), 1);
                }
                else
                {
                    double nfa = std::get<0>(detections[i]);
                    int color_index = colors.size() - 1 - int(std::min(std::floor(std::log(std::max(-1*nfa, 1.0))), colors.size() - 1.0));
                
                    
                    cv::circle(show_image, cv::Point(std::get<1>(detections[i]), std::get<2>(detections[i])), 
                            std::get<3>(detections[i]), colors[color_index], 1);

                }


            }
            cv::circle(show_image, cv::Point(std::get<1>(detections[0]), std::get<2>(detections[0])), 
                        std::get<3>(detections[0]), cv::Scalar(0, 0, 139), 2);
            
            // cv::imshow("results", show_image);
            // cv::waitKey(0);

            cv::imwrite(out_path, show_image);

            std::cout << std::endl;
        }
        else
        {
            std::cout << "===> NOT FOUND !!!" << std::endl << std::endl;
        }

    }




	return 0;
}
