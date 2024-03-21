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

#include <string>


#ifndef PARAMS_H_INCLUDED
#define PARAMS_H_INCLUDED

/**
 * @brief Structures of parameters dedicated to the creation of the model
 *
 * @param sizePatch: size of patches (sizePatch x sizePatch);
 * @param nSimilarPatches: minimum number of similar patches wanted;
 * @param excR: size of the exclusion region;
 * @param nbTrees: Number of trees;
 * @param same: Uses the exclusion region.
 **/
struct Params
{
	unsigned sizePatch;
	unsigned nSimilarPatches;
	unsigned excR;
	int nbTrees;
	bool same;
    float h;
};

struct ModelParams
{
	ModelParams()  { };
	ModelParams(int patch_size, int H, int nb_trees, int similar_number, bool same_image, 
	            int patch_stride, const std::string& model_path, const std::string& proto_path): 
        sizePatch(patch_size), nSimilarPatches(similar_number), h(H), same(same_image),
		patchStride(patch_stride), nbTrees(nb_trees), excR(2*sizePatch),
		modelPath(model_path), protoPath(proto_path) { }

	unsigned sizePatch;
	unsigned nSimilarPatches;
	unsigned excR;
	int nbTrees;
	bool same;
    float h;
	int patchStride;
	std::string protoPath;
	std::string modelPath;
};

struct TreeParams
{
	TreeParams()  { };
	TreeParams(float epsilon_, int nbVP_, int subS_, int rand_, 
	           const ModelParams& modelParams):
        epsilon(epsilon_), nbVP(nbVP_), subS(subS_), rand(rand_),
		mprms(modelParams) {  }


    ModelParams mprms;
	float epsilon;
	int nbVP;
	int subS;	
	int rand;
};

#endif
