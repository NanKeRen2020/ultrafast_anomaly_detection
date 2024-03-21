
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

find_package(BLAS REQUIRED)

find_package(LAPACK REQUIRED)
find_package(CUDA REQUIRED)

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/src/Utilities ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

include_directories(${CMAKE_SOURCE_DIR}/src/faisslib/include/faiss)
include_directories(${CMAKE_SOURCE_DIR}/src/faisslib/include)
link_directories(${CMAKE_SOURCE_DIR}/src/faisslib/lib)


if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

# add_library(model
# 	indexmodelImpl.cpp
# 	../Utilities/iio.c
# 	../Utilities/LibImages.cpp
# 	../Utilities/comparators.cpp
# 	../Utilities/mt19937ar.c
# 	../Utilities/PatchManager/imagePatchManager.cpp
# 	../Utilities/PatchManager/databasePatchManager.cpp
# 	../Utilities/PartitionTree/vptree.cpp
# 	../Utilities/PartitionTree/forestManager.cpp
# 	../Utilities/PatchSearch/localRefinement.cpp)


add_library(modelLibrary
	modelImpl.cpp
	# indexmodelImpl.cpp
	../Utilities/nfa.cpp)


# add_library(modelLibrary
# 	modelImpl.cpp
# 	# indexmodelImpl.cpp
# 	../Utilities/iio.c
# 	../Utilities/LibImages.cpp
# 	../Utilities/comparators.cpp
# 	../Utilities/mt19937ar.c
# 	../Utilities/PatchManager/imagePatchManager.cpp
# 	../Utilities/PatchManager/databasePatchManager.cpp
# 	../Utilities/PartitionTree/vptree.cpp
# 	../Utilities/PartitionTree/forestManager.cpp
# 	../Utilities/PatchSearch/localRefinement.cpp
# 	../Utilities/nfa.cpp)


set_target_properties(modelLibrary PROPERTIES
	SOVERSION ${PROJECT_VERSION})
target_link_libraries(modelLibrary  -lpthread -fopenmp -lfftw3_omp -lfftw3 -lm faiss CUDA::cufft CUDA::cublas CUDA::cudart  ${CUDA_CUBLAS_LIBRARIES}  ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} )

# target_link_libraries(model -lpthread -fopenmp -lm faiss CUDA::cufft CUDA::cublas CUDA::cudart  ${CUDA_CUBLAS_LIBRARIES}  ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} )

install(TARGETS modelLibrary
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    COMPONENT libs)
# install(TARGETS model
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     COMPONENT libs)