#include "diabetesBoard.h"

#define K 10 //number of initial clusters
#define RUIDO -2

typedef struct{
    int i,j,fi,fj;
}anchor;

typedef struct{
    Mat clusterMat;
    vector<int> neighbours;
    vector<int> unions; //representa quais cluster vao ser unidos
    Point center;
    int labelNumber;
    int numPixeis;
}region;


class clusterDetector{
public:

    clusterDetector();
    Mat run(Mat plate);

private:

    Mat plate, convertSrc, matLabels, grayPlate;
    vector<region> clusters; //cada feature é representada em uma coluna da matriz
    Mat points;
    int Kupdated;
    vector<Vec3b> colorTab;


    /*
     * Métodos
     */
    Mat filter();
    Mat cropImage(Mat src);
    Mat drawClusters();
    void getFeatures();

    Mat getLBPTexture();
    Mat getGaborTexture();
    void localBinaryPattern(Mat& src, Mat& dst, int radius, int neighbors);
    Mat textureRoi(int i,int j,Mat image);

    void applyKmeans();
    void applyMerge();
    void applyKmeansFilter();
    void getMahalanobisFeatures();
    int getFinalLabels();
    void getNeighbours(Mat binary, int idx);
    void checkPixelNeighbours(int newIdx, int idx);
    void getNewClusters(Mat binary, int idx);
    void connectedComponents(Mat binary, int idx);
    int connectedPixel(int *check, int *count, Mat *regionNumberMat);
    int mergeConnectedRegions(vector<int>& regionNumber, Mat *regionNumberMat);
    void labels2matLabels(Mat labels);
    float distance(int Xi, int Xf, int Yi, int Yf);


};
