#include "carbohydrate.h"

carboDetector::carboDetector(){
    /*********
     *
     * Definicao das relacoes de carboidrato. Foi usado estrutura map (dicionario)
     *
     *********/

    foods["batata"] = std::make_pair(0.2, 1); //relacao e densidade
    foods["arroz"] = std::make_pair(0.165, 1);
    foods["feijao"] = std::make_pair(0.165, 0.75);
    foods["banana"] = std::make_pair(0.23, 1);
    foods["abobora"] = std::make_pair(0.2, 1);
    foods["lasanha"] = std::make_pair(0.12, 1);
    foods["macarrao"] = std::make_pair(0.26, 1);
    foods["inhame"] = std::make_pair(0.27, 1);
    foods["baroa"] = std::make_pair(0.25, 1);
    foods["pao"] = std::make_pair(0.5, 1);
    foods["doce"] = std::make_pair(0.5, 1);
    foods["carne"] = std::make_pair(0.0, 1);
    foods["verdura"] = std::make_pair(0.0, 1);


    totalCarbo = 0;
    foodFeatures.clear();


    cout << "Clique em cada regiao que representa uma comida e indique qual é o alimento" << endl;
    cout << "Escreva conforme as opcoes listadas a seguir:" << endl;
    for(auto it = foods.cbegin(); it != foods.cend(); ++it)
    {
        std::cout << it->first << endl;
    }
    cout << endl;


}

void carboDetector::saveRegionPixeis(int regionPixeis, string name){

    auto it = foods.find(name);
    if(it == foods.end())
        cout << "Nome do alimento invalido" << endl;
    else{
        foodNames.push_back(name);
        foodFeatures.push_back(foodRegion(regionPixeis,it->second.first,it->second.second));
    }

}

float carboDetector::calculateCarbo(){


    float aux = 0;
    for(int i = 0; i < foodFeatures.size();i++){
        aux += (foodFeatures[i].density * foodFeatures[i].regionPixeis);
    }
    constant = totalWeigh/aux;
    //cout << constant << ' ' << foodFeatures.size() << endl;

    for(int i = 0; i < foodFeatures.size();i++){
        foodFeatures[i].weigh = foodFeatures[i].density * foodFeatures[i].regionPixeis * constant;
        foodFeatures[i].carbo = foodFeatures[i].weigh * foodFeatures[i].relation;
        cout << foodNames[i] << "  Peso: " << foodFeatures[i].weigh << " Carbo: " << foodFeatures[i].carbo << endl;
        totalCarbo += foodFeatures[i].carbo;
    }

    return totalCarbo;

}
