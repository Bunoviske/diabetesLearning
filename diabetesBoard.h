#ifndef DIABETES_BOARD
#define DIABETES_BOARD

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "string.h"

using namespace cv;
using namespace std;


namespace db { //diabetesBoard

    extern Mat src,labels;
    extern cv::RotatedRect plateBox;
    extern Point newRoiOrig;
    extern vector<int> numPixeis; //numPixeis de cada regiao

}

#endif
