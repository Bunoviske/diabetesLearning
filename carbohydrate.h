#ifndef CARBO_DETECTOR
#define CARBO_DETECTOR

#include "diabetesBoard.h"

struct foodRegion{
    int regionPixeis;
    float relation;
    float density;
    float weigh;
    int carbo;
    foodRegion(int regionP, float carboRelation, float foodDensity){ //construtor
        regionPixeis = regionP;
        relation = carboRelation;
        density = foodDensity;
    }
};

class carboDetector{

public:

    /*
     * Variaveis
     */
    int totalWeigh;


    /*
     * Métodos
     */

    carboDetector();
    void saveRegionPixeis(int regionPixeis, string name);
    int calculateCarbo();

private:

    float constant;
    int totalCarbo;

    /*
     * Métodos
     */

    std::map<string,pair<float,float>> foods;
    vector<foodRegion> foodFeatures;



};

#endif
