#include "ellipse.h"
#include "cluster.h"
#include "carbohydrate.h"

void CallBackFunc(int event, int x, int y, int flags, void*objeto);

carboDetector carbo; //global para ser visto pela funcao de callback

int main(int argc, char** argv)
{
    VideoCapture cap;
    ellipseDetector ellipse;
    clusterDetector cluster;

    // Le imagem/video do terminal
    if (argc > 1){
        cap.open(argv[1]);
    }
    // Se nao tiver argumentos de entrada
    else
    {
        cout << "Insira uma imagem" << endl;
        return -1;
    }

    cap.read(db::src);
    resize(db::src,db::src,Size(250,250));

    Mat plate = ellipse.run(db::src.clone());

    //cout << "Insira o peso total dos alimentos: " << endl;
    //cin >> carbo.totalWeigh;

    imshow("labels", cluster.run(plate));

    setMouseCallback("labels", CallBackFunc, &carbo);

    while(1){
        char c = waitKey();
        //se desejar fechar video/streaming, apertar esc
        if( (char)c == 27 ) //esc
        {
            break;
        }
    }


    cout << "Total de carboidrato: " << carbo.calculateCarbo() << endl;
    return 0;

}

void CallBackFunc(int event, int x, int y, int flags, void*objeto)
{

    if ( event == EVENT_FLAG_LBUTTON )
    {
        //cout << "Posicao do mouse (" << x << ", " << y << ")    " << endl;
        int idx = db::labels.at<int>(y,x);

        if (idx == -1 || idx == 50){
            cout << "Regiao invalida! clique novamente" << endl;
        }
        else{
            string name;
            cout << "Digite o nome do alimento da regiao " << idx << ":" << flush;
            cin >> name;
            cout << endl;
            carbo.saveRegionPixeis(db::numPixeis[idx], name);
        }

    }
}




