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

struct kmeansFeatures{
    Mat lbpHist;
    float val0;
    float val1;
    float val2;
    float i, j;
    kmeansFeatures(Mat lbpHist, float val0, float val1, float val2, float i, float j){ //construtor
        this->lbpHist = lbpHist;
        this->val0 = val0;
        this->val1 = val1;
        this->val2 = val2;
        this->i = i;
        this->j = j;

    }
};


class clusterDetector{
public:

    clusterDetector();
    Mat run(Mat plate);

private:

    Mat plate, convertSrc, matLabels, gray, lbpMat;
    vector<region> clusters; //cada feature é representada em uma coluna da matriz
    //std::vector<Vec6f> points;
    Mat points;
    int Kupdated;
    //vector<Vec3b> colorTab;


    /*
     * Métodos
     */
    Mat filter();
    Mat cropImage(Mat src);
    Mat drawClusters();
    void getFeatures();
    Mat getTexture();
    int localBinaryPattern(int i, int j);
    Mat textureRoi(int i,int j);

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
