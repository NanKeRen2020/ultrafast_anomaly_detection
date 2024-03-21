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
 * @file databasePatchManager.cpp
 * @brief Patch manager using patches from a given database different from the query
 *
 * @author Thibaud Ehret <ehret.thibaud@gmail.com>
 **/

#include "databasePatchManager.h"
#include <iostream>
#include "faiss/utils/distances.h"


DatabasePatchManager::DatabasePatchManager(std::vector<float> const& image, ImageSize ims, int sizeP, int patch_stride)
{
	im = &image;
	current = nullptr;
	
	sizePatch = sizeP;
	imSize = ims;
	stride = patch_stride;

	continue_size = sizePatch*imSize.nChannels;
	nw = imSize.nChannels*imSize.width;
	patch_dim = continue_size*sizePatch;

}

DatabasePatchManager& DatabasePatchManager::operator=(const DatabasePatchManager& dbpm)
{
	auto new_im = new std::vector<float>(*dbpm.im);
	auto new_current = new std::vector<float>(*dbpm.current);
	delete im;
	delete current;
	im = new_im;
    current = new_current;

	sizePatch = dbpm.sizePatch; imSize = dbpm.imSize; curSize = dbpm.curSize;
	stride = dbpm.stride; continue_size = dbpm.continue_size; nw = dbpm.nw; patch_dim = dbpm.patch_dim;

    return *this;
}

void DatabasePatchManager::getAllPatches(std::vector<unsigned>& allPatches)
{

	for(unsigned x = 0, k = 0; x < imSize.width - sizePatch + 1; x = x + stride - 1)
	for(unsigned y = 0; y < imSize.height - sizePatch + 1; y = y + stride - 1, ++k)
	{
		allPatches[k] = imSize.nChannels*(x + y*imSize.width);
	}
	// maybe donot need
	if (stride > 1)
	{
		allPatches.shrink_to_fit();
	}			
}

int DatabasePatchManager::getNbPatches()
{
	if (stride == 1)
	return (imSize.width - sizePatch + 1) * (imSize.height - sizePatch + 1);
	else
	{
		return ((imSize.width - sizePatch)/stride + 1) * ((imSize.height - sizePatch)/stride + 1);
	}

}

float DatabasePatchManager::distance(unsigned patch1, unsigned patch2)
{

    float faiss_dis = 0.f;
	int pp; 
    for (unsigned hy = 0; hy < sizePatch; ++hy)
	{
		pp = hy*nw;
		// faiss vector distance compute
        faiss_dis += faiss::fvec_L2sqr(current->data() + patch1 + pp, im->data() + patch2 + pp, continue_size);
	}
    return std::sqrt(faiss_dis / patch_dim) / 255.f;

}

float DatabasePatchManager::distance(unsigned patch1, unsigned patch2, float& raw_dist)
{

    float faiss_dis = 0.f;
	int pp; 
    for (unsigned hy = 0; hy < sizePatch; ++hy)
	{
		pp = hy*nw;
		// faiss vector distance compute
        faiss_dis += faiss::fvec_L2sqr(current->data() + patch1 + pp, im->data() + patch2 + pp, continue_size);
	}
    raw_dist = faiss_dis / (patch_dim);

	return std::sqrt(raw_dist) / 255.f;
}

DatabasePatchManager::~DatabasePatchManager()
{

}

void DatabasePatchManager::setCurrentImage(const std::vector<float>& cur, ImageSize sz)
{
	current = &cur;
	curSize = sz;
}
