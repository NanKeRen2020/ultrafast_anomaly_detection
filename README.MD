An ultrafast nice quasi-generic image anomaly detection method, which 

reduce anomaly detection in images to anomaly detection in noise.

This approach is unsupervised and works on arbitrary images,

but also can apply on residuals extracted from dense low- and mid-level features 

of the popular pre-trained deep neural network, such as VGG, Resnet, etc.



# Overview


This source code provides an ultrafast implementation of the anomaly detection method 

developped in [1][2], which has been studied in details in [3]. The running time is ~17ms

for 320*480 image on ubuntu1604 Core i7-8570H 2.2Ghz*12 8G gtx1050 cuda10.0.

This project only open source the original optimizing implementation, the faster version is 

another total different patch search approach. 


detection result image
![image](https://github.com/NanKeRen2020/anomaly_detection/blob/main/detections.png)
![image](https://github.com/NanKeRen2020/ultrafast_anomaly_detection/blob/main/detections1.png)

# Usage


The code is compilable on ubuntu1604

requires the cmake and make programs, gcc/g++7.5, OpenCV3.4, fftw3

and faiss library with gpu. You need to check install intruction to 

compile the faiss library[4]. 


$ mkdir build; cd build
$ cmake ..
$ make

Binaries will be created in build/bin folder.

NOTE: By default, the code is compiled with OpenMP multithreaded

parallelization enabled (if your system supports it). 


running example:

./bin/anomaly_detection  /home/seeking/test_projects/anomaly_detection/models/deploy.prototxt /home/seeking/test_projects/

anomaly_detection/models/VGG_normalised.caffemodel 1 8 16 /home/seeking/test_projects/anomaly_detection/datas/test1.png -1 

0 /home/seeking/test_projects/anomaly_detection/results/detections_cpp.png /home/seeking/test_projects/anomaly_detection/

datas/test1.png 2 2

NOTE: Repalce with your own path.

computes the result using the method and parameters from [2], the caffe model file can download from [3].


# References

[1] https://github.com/NanKeRen2020/anomaly_detection.

[2] Davy, Axel, et al. Reducing Anomaly Detection in Images to Detection in Noise. 
    IEEE International Conference on Image Processing, 2018.

[3] http://www.ipol.im/.

[4] https://github.com/facebookresearch/faiss.
