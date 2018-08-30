#include "carbohydrate.h"

carboDetector::carboDetector(){
    /*********
     *
     * Definicao das relacoes de carboidrato. Foi usado estrutura map (dicionario)
     *
     *********/

    //relacao de CHO e densidade mapeados pelo nome

    //carboidratos em geral
    foods["arroz"] = std::make_pair(0.165, 0.73);
    foods["feijao"] = std::make_pair(0.165, 0.70);
    foods["arroz&feijao"] = std::make_pair(0.165, 0.75);
    foods["pasta"] = std::make_pair(0.26, 0.55);
    foods["batata"] = std::make_pair(0.2, 0.59);
    foods["pureBatata"] = std::make_pair(0.165, 1.048);
    foods["batataDoce"] = std::make_pair(0.23, 0.65);
    foods["mandioca"] = std::make_pair(0.25, 0.63);
    foods["inhame"] = std::make_pair(0.27, 0.79);
    foods["baroa"] = std::make_pair(0.25, 1); //densidade incerta
    foods["lasanha"] = std::make_pair(0.12, 1.04);

    //frutas
    foods["banana"] = std::make_pair(0.23, 0.634);

    //paes e doces
    foods["pao"] = std::make_pair(0.5, 0.29); // ou 0.42 para pao branco!
    foods["bolo"] = std::make_pair(0.5, 0.415);
    foods["doce"] = std::make_pair(0.5, 1);  //densidade incerta

    //proteina
    foods["carne"] = std::make_pair(0.0, 0.96);
    foods["carneMoida"] = std::make_pair(0.0, 0.95); //ainda incerto
    foods["peixe"] = std::make_pair(0.0, 0.96);
    foods["frango"] = std::make_pair(0.0, 0.96);
    foods["ovo"] = std::make_pair(0.0, 0.6);
    foods["queijoCottage"] = std::make_pair(0.0, 1);


    //legumes e verduras
    foods["salada"] = std::make_pair(0.0, 0.06);
    foods["tomateCereja"] = std::make_pair(0.0, 0.63);
    foods["tomate"] = std::make_pair(0.0, 0.76);
    foods["brocolis"] = std::make_pair(0.0, 0.65);
    foods["cenoura"] = std::make_pair(0.0, 0.54);
    foods["abobora"] = std::make_pair(0.0, 1); //densidade de abobora amassada no cup
    foods["espinafre"] = std::make_pair(0.0, 0.76);
    foods["sopa"] = std::make_pair(0.06, 1);


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
