
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)

find_package(BLAS REQUIRED)

find_package(LAPACK REQUIRED)
find_package(CUDA REQUIRED)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utilities)
include_directories(${CMAKE_SOURCE_DIR}/model)


include_directories(${LAPACK_INCLUDES})
include_directories(${CBLAS_INCLUDE_DIR})
include_directories(${FFTW_INCLUDEs})
SET(COMMON_COMPILE_FLAGS "-fopenmp ")
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

link_directories(${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/src/faisslib/include)
link_directories(${CMAKE_SOURCE_DIR}/src/faisslib/lib)
link_directories(${CMAKE_SOURCE_DIR}/src/modellib)


include_directories(${CMAKE_SOURCE_DIR}/src/library)
add_executable(anomaly_detection anomaly_detection.cpp)
# target_link_libraries(anomaly_detection modelLibrary png ${CUBLAS_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${CBLAS_LIBRARIES} ${FFTW_LIBRARIES} ${OpenCV_LIBS} -lstdc++fs -lpthread -fopenmp  -lboost_serialization -lboost_system -lboost_filesystem)
target_link_libraries(anomaly_detection model modelLibrary -lm faiss CUDA::cufft CUDA::cublas CUDA::cudart ${CUDA_CUBLAS_LIBRARIES}  ${CUBLAS_LIBRARIES}  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${CBLAS_LIBRARIES} ${FFTW_LIBRARIES} ${OpenCV_LIBS} -lstdc++fs -lpthread -fopenmp  -lboost_serialization -lboost_system -lboost_filesystem)


