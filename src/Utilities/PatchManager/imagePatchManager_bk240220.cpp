/*
 * Copyright (c) 2016, Thibaud Ehret <ehret.thibaud@gmail.com>
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
 * @file imagePatchManager.cpp
 * @brief Patch manager using the patches from the image
 *
 * @author Thibaud Ehret <ehret.thibaud@gmail.com>
 **/
#include "faiss/utils/distances.h"

#include "imagePatchManager.h"
#include <iostream>

ImagePatchManager::ImagePatchManager(std::vector<float> const& image, ImageSize ims, int sizeP, int patch_stride)
{
	im = &image;
	imSize = ims;
	sizePatch = sizeP;
	stride = patch_stride;

	continue_size = sizePatch*imSize.nChannels;
	nw = imSize.nChannels*imSize.width;
	patch_dim = continue_size*sizePatch;
}

void ImagePatchManager::getAllPatches(std::vector<unsigned>& allPatches)
{
	// for(unsigned x = 0, k = 0; x < imSize.width - sizePatch + 1; ++x)
	// for(unsigned y = 0; y < imSize.height - sizePatch + 1; ++y, ++k)
	for(unsigned x = 0, k = 0; x < imSize.width - sizePatch + 1; x = x + stride)
	for(unsigned y = 0; y < imSize.height - sizePatch + 1; y = y + stride, ++k)
	{
		allPatches[k] = imSize.nChannels*(x + y*imSize.width);
        if (stride > 1)
		{
		// 	allPatches.clear();
		// 	allPatches.emplace_back(imSize.nChannels*(x + y*imSize.width));
			allPatches.shrink_to_fit();
		}
		// if (stride == 1)
        // allPatches[k] = imSize.nChannels*(x + y*imSize.width);
		// else
		// {
		// 	allPatches.clear();
		// 	allPatches.emplace_back(imSize.nChannels*(x + y*imSize.width));
		// }
	}

	// std::cout << "allPatches: " << allPatches.size() << ", " << kk << std::endl;	

}

int ImagePatchManager::getNbPatches()
{
	if (stride == 1)
	return (imSize.width - sizePatch + 1) * (imSize.height - sizePatch + 1);
	else
	{
		//return (imSize.width - sizePatch + 1)/stride * (imSize.height - sizePatch + 1)/stride;
		return ((imSize.width - sizePatch)/stride + 1) * ((imSize.height - sizePatch)/stride + 1);
	}
	
}

float ImagePatchManager::distance(unsigned patch1, unsigned patch2)
{

	// std::vector<float> pv1;
	// std::vector<float> pv2;
	// int pp; 
	// for (unsigned hy = 0; hy < sizePatch; ++hy)
	// {
	// 	pp = patch1 + hy*imSize.nChannels*imSize.width;
    //     pv1.insert(pv1.end(), im->begin() + pp, 
    //                           im->begin() + pp + sizePatch*imSize.nChannels);	

    //     pp = patch2 + hy*imSize.nChannels*imSize.width;
    //     pv2.insert(pv2.end(), im->begin() + pp, 
    //                           im->begin() + pp + sizePatch*imSize.nChannels);	
	// }
	// float faiss_dis = faiss::fvec_L2sqr(pv1.data(), pv2.data(), sizePatch * sizePatch * imSize.nChannels);

	
    float faiss_dis = 0.f;
	int pp; 
    for (unsigned hy = 0; hy < sizePatch; ++hy)
	{
		pp = hy*nw;
        faiss_dis += faiss::fvec_L2sqr(im->data() + patch1 + pp, im->data() + patch2 + pp, continue_size);
	}
	return std::sqrt(faiss_dis / (patch_dim)) / 255.f;

	// float dist = 0.f, dif;
	// float dis = 0.f;
	// for (unsigned hy = 0; hy < sizePatch; ++hy)
	// for (unsigned hx = 0; hx < sizePatch; ++hx)
	// for (unsigned hc = 0; hc < imSize.nChannels; ++hc)
	// {
	// 	dist += (dif = (*im)[patch1 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] - 
	// 	        (*im)[patch2 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc]) * dif;
    //     // std::cout << (*im)[patch1 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] << " "
	// 	//           << (*im)[patch2 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] << ", dis: ";
	// 	// dis += ((*im)[patch1 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] - (*im)[patch2 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] ) * 
	// 	//         ((*im)[patch1 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] - (*im)[patch2 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] );
	//     // std::cout << dist << " " << dis << ",  ";
	// }
	    
	// //std::cout << "ImagePatchManager dist: " << dist << ", " << faiss_dis << std::endl;

	// return std::sqrt(dist / (patch_dim)) / 255.f;
}

float ImagePatchManager::distance(unsigned patch1, unsigned patch2, float& raw_dist)
{
	// std::vector<float> pv1;
	// std::vector<float> pv2;
	// int pp; 
	// for (unsigned hy = 0; hy < sizePatch; ++hy)
	// {
	// 	pp = patch1 + hy*imSize.nChannels*imSize.width;
    //     pv1.insert(pv1.end(), im->begin() + pp, 
    //                           im->begin() + pp + sizePatch*imSize.nChannels);	

    //     pp = patch2 + hy*imSize.nChannels*imSize.width;
    //     pv2.insert(pv2.end(), im->begin() + pp, 
    //                           im->begin() + pp + sizePatch*imSize.nChannels);	
	// }
	// float faiss_dis = faiss::fvec_L2sqr(pv1.data(), pv2.data(), sizePatch * sizePatch * imSize.nChannels);

    float faiss_dis = 0.f;
	int pp; 
    for (unsigned hy = 0; hy < sizePatch; ++hy)
	{
		pp = hy*nw;
        faiss_dis += faiss::fvec_L2sqr(im->data() + patch1 + pp, im->data() + patch2 + pp, continue_size);
	}
	raw_dist = faiss_dis / (patch_dim);


	// float dist = 0.f, dif;
	// for (unsigned hy = 0; hy < sizePatch; ++hy)
	// for (unsigned hx = 0; hx < sizePatch; ++hx)
	// for (unsigned hc = 0; hc < imSize.nChannels; ++hc)
	// 	dist += (dif = (*im)[patch1 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc] - 
	// 	        (*im)[patch2 + hx*imSize.nChannels + hy*imSize.nChannels*imSize.width + hc]) * dif;
	
	
	// raw_dist = dist / (patch_dim);

	// std::cout << "ImagePatchManager dist1: " << dist << ", " << faiss_dis << std::endl;

	return std::sqrt(raw_dist) / 255.f;
}


ImagePatchManager::~ImagePatchManager()
{

}
