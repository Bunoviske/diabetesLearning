#include "cluster.h"

//K = 18
Vec3b colorTab[] =
{

    Vec3b(0, 0, 255),
    Vec3b(0,255,0),
    Vec3b(255, 0, 0),

    Vec3b(255,0,255),
    Vec3b(0,255,255),
    Vec3b(255,255,0),

    Vec3b(255,100,100),
    Vec3b(100,0,255),
    Vec3b(100,255,0),
    Vec3b(255,0,100),

    Vec3b(100,255,100),
    Vec3b(100,100,255),


    Vec3b(255,255,100),
    Vec3b(255,100,255),
    Vec3b(100,255,255),


    Vec3b(0,255,100),
    Vec3b(255,100,0),
    Vec3b(0,100,255),

};


clusterDetector::clusterDetector(){
    //construtor
    //    RNG rng( 0xFFFFFFFF );

    //    for (int i = 0; i < K; i++){
    //        colorTab.push_back(Vec3b((uchar)rng.uniform(0,255),(uchar)rng.uniform(0,255),(uchar)rng.uniform(0,255)));
    //    }
    this->Kupdated = K;
}

Mat clusterDetector::run(Mat plate){
    this->plate = plate;
    cvtColor(this->cropImage(this->plate),this->grayPlate,CV_BGR2GRAY);
    this->convertSrc = this->cropImage(this->filter());
    this->getFeatures();
    this->applyKmeans();

    imshow("labelsBeforeMerge", drawClusters());
    this->applyMerge();

    db::labels = matLabels.clone();
    for(int i = 0; i < clusters.size();i++){
        db::numPixeis.push_back(clusters[i].numPixeis);
    }

    return drawClusters();

    //    cv::Mat cimage = cv::Mat::zeros(showClusters.size(), CV_8UC3);
    //    Point auxPoint = Point(db::plateBox.center.x - db::newRoiOrig.x,db::plateBox.center.y - db::newRoiOrig.y);
    //    cv::ellipse(cimage, auxPoint, db::plateBox.size*0.5f, db::plateBox.angle, 0, 360, cv::Scalar(255,255,255), -1, CV_AA);
    //    cv::bitwise_and(showClusters,cimage,showClusters);




}

void clusterDetector::applyKmeans(){

    Mat bestLabels, bestCenters;
    int attempts = 10;

    double bestCompactness = kmeans(points, K, bestLabels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 100,0.01),
                                    attempts,KMEANS_PP_CENTERS, bestCenters);

    labels2matLabels(bestLabels);
    applyKmeansFilter();
}

void clusterDetector::applyKmeansFilter(){
    Mat binary = Mat::zeros(convertSrc.size(), CV_8UC1);
    for (int it = 0; it < 2; it++){
        if (it == 1) { //apenas inicializa o vetor de clusters vazio
            for(int ii = 0; ii < Kupdated; ii++){
                clusters.push_back(region());
            }
        }

        //pega todas as regioes antes de pegar os vizinhos! Usar parte da funcao showClusters
        int Kaux = Kupdated;
        for (int idx = 0; idx < Kaux; idx++){
            for (int i=0; i<convertSrc.rows; i++) {
                // get the address of row j
                uchar* data2write = binary.ptr<uchar>(i);
                Vec3b* data = convertSrc.ptr<Vec3b>(i);
                int* pointer = matLabels.ptr<int>(i);

                for (int j=0; j<convertSrc.cols; j++) {
                    // process each pixel ---------------------
                    if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
                        if (pointer[j] == idx){
                            data2write[j] = 255;
                        }
                        else
                            data2write[j] = 0;
                    }
                }
            }
            //na primeira iteracao remove os ruidos e pega novos clusters e na segunda pega os vizinhos de cada cluster
            if (it == 0) getNewClusters(binary,idx);
            else getNeighbours(binary,idx);
        }
    }
}

void clusterDetector::getNeighbours(Mat binary, int idx){

    int low = 50;
    Mat canny;
    cv::Canny(binary,canny,low,2*low);
    //imshow("canny", canny);
    //waitKey();


    int offset = 5;
    int count = 0;
    for (int i=0; i<canny.rows; i++) {
        // get the address of row j
        uchar* boarderData = canny.ptr<uchar>(i);
        int* labelData = matLabels.ptr<int>(i);

        for (int j=0; j<canny.cols; j++) {
            // process each pixel ---------------------
            if (boarderData[j] == 255){ //pixel de borda
                checkPixelNeighbours(labelData[j+offset >= matLabels.cols ? matLabels.cols-1 : j + offset],idx);
                checkPixelNeighbours(labelData[j-offset < 0 ? 0 : j - offset],idx);
                checkPixelNeighbours(matLabels.at<int>(i+offset >= matLabels.rows ? matLabels.rows-1 : i + offset,j),idx);
                checkPixelNeighbours(matLabels.at<int>(i-offset < 0 ? 0 : i - offset,j),idx);

            }
            count++;
        }
    }
    //    cout << endl << "Regiao " << idx << " tem vizinhos: " << flush;
    //    for (int j = 0; j < clusters[idx].neighbours.size(); j++){
    //        cout << clusters[idx].neighbours[j] << ' ' << flush;
    //    }
}

void clusterDetector::checkPixelNeighbours(int newIdx, int idx){

    //so adiciona vizinhos novos se eles nao tiverem sido adicionados ja ou se for ruido
    if (newIdx == idx || newIdx == -1 || newIdx == RUIDO) return;
    if (clusters[idx].neighbours.size() != 0){
        int cont = 0;
        for (int j = 0; j < clusters[idx].neighbours.size(); j++){
            if(clusters[idx].neighbours[j] != newIdx){
                cont++;
            }
        }
        if (cont == clusters[idx].neighbours.size()){
            clusters[idx].neighbours.push_back(newIdx);
        }

    }
    else{
        clusters[idx].neighbours.push_back(newIdx);
    }
}



void clusterDetector::getNewClusters(Mat binary, int idx){

    //remove ruido!
    //imshow("binary cru", binary);
    cv::erode(binary, binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::dilate(binary, binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    //imshow("binary filtered", binary);

    connectedComponents(binary, idx);

    //adiciona ruido a matriz de labels
    for (int i=0; i<convertSrc.rows; i++) {
        // get the address of row j
        uchar* dataBin = binary.ptr<uchar>(i);
        Vec3b* data = convertSrc.ptr<Vec3b>(i);
        int* pointer = matLabels.ptr<int>(i);

        for (int j=0; j<convertSrc.cols; j++) {
            // process each pixel ---------------------
            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
                if (pointer[j] == idx){
                    if(dataBin[j] == 0){
                        pointer[j] = RUIDO; //ruido - nao pertence a nenhum grupo
                    }
                }
            }
        }
    }
}

void clusterDetector::connectedComponents(Mat binary, int idx){

    int offset = 1;
    Mat regionNumber = Mat::zeros(binary.size(), CV_32SC1);
    int count = 1;

    //percorre a imagem e aplica o algoritmo connected component labeling de conectividade = 8
    for (int i=0; i<binary.rows; i++) {
        // get the address of row j
        uchar* dataBin = binary.ptr<uchar>(i);
        int* pointer = regionNumber.ptr<int>(i);

        for (int j=0; j<binary.cols; j++) {
            if (dataBin[j] == 255){
                int check[4] = {0};
                int *auxRegion = regionNumber.ptr<int>(i-offset < 0 ? 0 : i - offset);
                check[0] = pointer[j-offset < 0 ? 0 : j - offset];
                check[1] = auxRegion[j-offset < 0 ? 0 : j - offset];
                check[2] = auxRegion[j];
                check[3] = auxRegion[j+offset >= regionNumber.cols ? regionNumber.cols-1 : j + offset];

                pointer[j] = connectedPixel(check, &count, &regionNumber);

            }
        }
    }

    //atribuir a matriz labels (matLabels) os novos clusters encontrados no vetor regionNumber.
    //o Kupdated vai sendo incrementado para cada regiao nova encontrada
    vector<int> newClusterNumber, newLabelNumber;
    for (int i=0; i<regionNumber.rows; i++) {
        // get the address of row j
        int* pointer = regionNumber.ptr<int>(i);
        int* label = matLabels.ptr<int>(i);

        for (int j=0; j<regionNumber.cols; j++) {

            if(pointer[j] != 0){

                if(newClusterNumber.size() == 0) {
                    newClusterNumber.push_back(pointer[j]);
                    newLabelNumber.push_back(idx);
                    //cout << pointer[j] << ' ' << flush;
                }
                else{
                    int cont = 0;
                    for(int k = 0; k < newClusterNumber.size(); k++){
                        if(pointer[j] != newClusterNumber[k]) cont++;
                        else if (pointer[j] == newClusterNumber[k]) label[j] = newLabelNumber[k];

                    }
                    if(cont == newClusterNumber.size()){
                        newClusterNumber.push_back(pointer[j]);
                        newLabelNumber.push_back(Kupdated);
                        label[j] = Kupdated;
                        Kupdated++;
                        //cout << pointer[j] << ' ' << flush;
                    }
                }

            }

        }
    }
    //cout << endl;

}

int clusterDetector::connectedPixel(int *check, int *count, Mat *regionNumberMat){
    if(check[0]+check[1]+check[2]+check[3] == 0)//pixel desconectado de qualquer regiao!
    {
        *count = *count + 1;
        return *count - 1;
    }

    vector<int> regionNumber;

    for(int i = 0; i < 4; i++){
        if(check[i] != 0){
            if(regionNumber.size() == 0) regionNumber.push_back(check[i]);
            else{
                int cont = 0;
                for(int j = 0; j < regionNumber.size(); j++){
                    if(check[i] != regionNumber[j]) cont++;
                }
                if(cont == regionNumber.size()){
                    regionNumber.push_back(check[i]);
                }
            }
        }
    }
    if (regionNumber.size() == 1){
        return regionNumber[0]; //apenas uma regiao foi encontrada
    }
    else {
        return mergeConnectedRegions(regionNumber, regionNumberMat); //dá merge nas regioes que são conectadas
    }
}

int clusterDetector::mergeConnectedRegions(vector<int>& regionNumber,Mat *regionNumberMat){

    int min = *std::min_element(regionNumber.begin(), regionNumber.end());
    for (int i=0; i<regionNumberMat->rows; i++) {
        // get the address of row j
        int* pointer = regionNumberMat->ptr<int>(i);

        for (int j=0; j<regionNumberMat->cols; j++) {
            for(int k = 0; k < regionNumber.size(); k++){
                if(pointer[j] == regionNumber[k]){
                    pointer[j] = min;
                }
            }
        }

    }
    return min;
}

void clusterDetector::getMahalanobisFeatures(){
    int pointsSumI[Kupdated];
    int pointsSumJ[Kupdated];
    for(int i =0; i < Kupdated;i++){ //inicializa os vetores zerados
        pointsSumJ[i] = 0;
        pointsSumI[i] = 0;
    }

    int cont = 0;
    for (int i=0; i<convertSrc.rows; i++) {
        // get the address of row j
        Vec3b* data = convertSrc.ptr<Vec3b>(i);
        int* pointer = matLabels.ptr<int>(i);

        for (int j=0; j<convertSrc.cols; j++) {
            // process each pixel ---------------------
            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
                int idx = pointer[j];
                if (idx != RUIDO){ //representa os ruidos filtrados

                    //pega as features para realizar a distancia de mahalanobis
//                    cv::Mat sample(1, 5, CV_32FC1);
//                    sample.at<float>(0, 0) = data[j].val[0];
//                    sample.at<float>(0, 1) = data[j].val[1];
//                    sample.at<float>(0, 2) = data[j].val[2];
                    //sample.at<float>(0, 3) = i; //i      //mudar tamanho do vetor sample logo acima!!!!
                    //sample.at<float>(0, 4) = j; //j      //mudar tamanho do vetor sample logo acima!!!!
                    //sample.at<float>(0, 3) = points[cont].val[5];

                    // exclui as features i e j na hora de atribuir mahalanobis features!
                    Mat sample = points.colRange(2,points.cols).rowRange(cont,cont+1);

                    clusters[idx].clusterMat.push_back(sample);

                    //para descobrir centro geometrico de cada cluster
                    pointsSumI[idx] += i;
                    pointsSumJ[idx] += j;
                }
                cont++;
            }
        }
    }

    //pega o centro geometrico de cada regiao e o num total de pixeis
    for(int i =0; i < Kupdated;i++){
        clusters[i].center.x = pointsSumJ[i]/clusters[i].clusterMat.rows;
        clusters[i].center.y = pointsSumI[i]/clusters[i].clusterMat.rows;
        clusters[i].numPixeis = clusters[i].clusterMat.rows;

    }
}

void clusterDetector::applyMerge(){


    getMahalanobisFeatures();


    Mat covar[Kupdated], mean[Kupdated];

    for(int i = 0; i < Kupdated; i++){
        calcCovarMatrix(clusters[i].clusterMat, covar[i],mean[i], CV_COVAR_NORMAL + CV_COVAR_ROWS);
    }

    Mat invertCovar1, invertCovar2;
    //cout << endl;
    for(int i  = 0; i < Kupdated; i++){
        for(int j = 0; j < clusters[i].neighbours.size(); j++){

            invert(covar[i],invertCovar1, DECOMP_SVD);
            invert(covar[clusters[i].neighbours[j]],invertCovar2, DECOMP_SVD);
            if (Mahalanobis(mean[i],mean[clusters[i].neighbours[j]],invertCovar1 + invertCovar2) < 0.1){
                clusters[i].unions.push_back(clusters[i].neighbours[j]);
                //cout << "Junção das regiões " << i << " e " << clusters[i].neighbours[j] << endl;
            }
        }
    }

    //modifica matLabels com as novas regioes unidas
    Kupdated = getFinalLabels();
    clusters.clear();
    for(int ii = 0; ii < Kupdated; ii++){
        clusters.push_back(region());
    }
    getMahalanobisFeatures(); //altera o vetor de clusters (vector<region> clusters com as novas regioes unidas)
}

int clusterDetector::getFinalLabels(){

    int cont = 0;

    for(int i = 0; i < Kupdated; i++){

        clusters[i].unions.push_back(i); //add o proprio valor no vetor
        int minLabel = *std::min_element(clusters[i].unions.begin(), clusters[i].unions.end());
        if (minLabel == i){ //quer dizer que tem que incrementar o contador pois um novo label deve ser criado
            clusters[i].labelNumber = cont;
            cont++; //contador incrementa quando nao ha unioes
        }
        else{
            clusters[i].labelNumber = clusters[minLabel].labelNumber;
        }
        //cout << i << ' ' << clusters[i].labelNumber << endl;
    }
    for (int i=0; i<matLabels.rows; i++) {
        // get the address of row j
        int* pointer = matLabels.ptr<int>(i);
        for (int j=0; j<matLabels.cols; j++) {
            //se for diferente de preto e de ruido, add o novo valor do label
            if (pointer[j] != -1 && pointer[j] != RUIDO) pointer[j] = clusters[pointer[j]].labelNumber;
        }
    }
    return cont;
}


void clusterDetector::localBinaryPattern(Mat& src, Mat& dst, int radius, int neighbors) {
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<uchar>(i,j)) && (abs(t-src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

Mat clusterDetector::getLBPTexture(){

    Mat lbp;
    int radius = 3; int neighbours = 10; //3 10   ---- 2 8
    localBinaryPattern(grayPlate,lbp,radius,neighbours);
    normalize(lbp,lbp,0,255,CV_MINMAX);
    lbp.convertTo(lbp,CV_8UC1);
    imshow("lbppppp",lbp);
    return lbp;

}

Mat clusterDetector::getGaborTexture(){

    Mat gray_f, gabor;
    grayPlate.convertTo(gray_f,CV_32F);
    int kernel_size = 11;
    double sig = 6, theta = 0, lambda = 30.0, gama = 0.5, psi = 0;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, theta, lambda, gama, psi);
    cv::filter2D(gray_f, gabor, CV_32F, kernel);

    normalize(gabor,gabor,0,255,CV_MINMAX);
    gabor.convertTo(gabor,CV_8UC1);
    imshow("k",kernel);
    imshow("d",gabor);
    return gabor;
}


void clusterDetector::getFeatures(){

    //variavel 'points' contem as features para o kmeans

    Mat lbpMat = getLBPTexture();
    Mat gabor = getGaborTexture();

    int nl = convertSrc.rows;
    int nc = convertSrc.cols;

    for (int i=0; i<nl; i++) {
        // get the address of row j
        Vec3b* data = convertSrc.ptr<Vec3b>(i);

        for (int j=0; j<nc; j++) {
            // process each pixel ---------------------
            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){

                int histSize = 16;
                float range[] = { 0, 256 } ;
                const float* histRange = { range };
                bool uniform = true; bool accumulate = false;
                Mat textureHist;
                Mat ROI = textureRoi(i,j, lbpMat);

                //Compute the histogram
                calcHist(&ROI, 1, 0, Mat(), textureHist, 1, &histSize, &histRange, uniform, accumulate );
                transpose(textureHist,textureHist);

                const int FeatSize = 6;

                //vetor de features que representa a posicao i,j e os valores cieLAB
                float dataFeat[FeatSize] = {(float)i,(float)j,
                                            (float)data[j].val[0],(float)data[j].val[1],(float)data[j].val[2],
                                            //(float)gabor.at<uchar>(i,j) //gabor feature
                                           };

                cv::Mat features = cv::Mat(1, FeatSize, CV_32F, dataFeat);
                hconcat(features,textureHist,features); // lbp histogram feature!!!

                points.push_back(features);

            }
            //end of pixel processing ----------------

        }
    }
}

Mat clusterDetector::textureRoi(int i, int j, Mat image){ //pega roi em torno do ponto da imagem texturizada
    anchor roi;
    int offset = 10; // 5 - 10

    roi.fi = i + offset <= image.rows-1 ? i + offset : image.rows-1;
    roi.fj = j + offset <= image.cols-1 ? j + offset : image.cols-1;
    roi.i = i - offset >= 0 ? i - offset : 0;
    roi.j = j - offset >= 0 ? j - offset : 0;

    return image(Rect(Point(roi.j,roi.i),Point(roi.fj,roi.fi)));
}

Mat clusterDetector::filter(){
    Mat convertSrc = Mat::zeros(plate.size(), CV_8UC3);

    cvtColor(plate, convertSrc,CV_BGR2Lab);
    medianBlur(convertSrc,convertSrc,7);

    //pyrMeanShiftFiltering(convertSrc,convertSrc,10,10);

    imshow("lab", convertSrc);

    //cvtColor(plate, graySrc,CV_BGR2GRAY);
    //medianBlur(graySrc,graySrc,7);
    //Canny(graySrc,graySrc,50,2*50);
    //imshow("hsv", graySrc);

    return convertSrc;

}

void clusterDetector::labels2matLabels(Mat labels){
    //transforma vetor labels em matriz
    int count = 0;
    int* pointer = labels.ptr<int>(count);
    matLabels = Mat(convertSrc.size(),CV_32SC1);
    for(int i = 0; i < matLabels.rows; i++){
        int* dataLabel = matLabels.ptr<int>(i);
        Vec3b* data = convertSrc.ptr<Vec3b>(i);
        for (int j=0; j<matLabels.cols; j++){
            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
                dataLabel[j] = pointer[count];
                count++;
            }
            else{
                dataLabel[j] = -1; //representa regiao fora do prato (em preto)
            }
        }
    }
}

Mat clusterDetector::cropImage(Mat src){
    int nl= src.rows;
    int nc= src.cols;
    anchor roi;
    roi.fi = 0;    roi.fj = 0;    roi.i = nl;    roi.j = nc;

    for (int i=0; i<nl; i++) {

        // get the address of row i
        Vec3b* data = src.ptr<Vec3b>(i);

        for (int j=0; j<nc; j++) {
            // process each pixel ----------------
            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
                if(i < roi.i) roi.i = i;
                else if(i > roi.fi) roi.fi = i;

                if(j < roi.j) roi.j = j;
                else if(j > roi.fj) roi.fj = j;
            }
            //end of pixel processing ----------------

        }
    }
    db::newRoiOrig = Point(roi.j,roi.i);
    return src(Rect(Point(roi.j,roi.i),Point(roi.fj,roi.fi)));
}

float clusterDetector::distance(int Xi, int Xf, int Yi, int Yf){
    return sqrt(pow(Xf - Xi, 2) + pow(Yf - Yi, 2));
}

Mat clusterDetector::drawClusters(){
    Mat showClusters = convertSrc.clone();

    //agrupa cada cluster em uma Mat diferente
    for (int i=0; i<convertSrc.rows; i++) {
        // get the address of row j
        Vec3b* data = showClusters.ptr<Vec3b>(i);
        int* pointer = matLabels.ptr<int>(i);

        for (int j=0; j<convertSrc.cols; j++) {
            // process each pixel ---------------------
            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
                int idx = pointer[j];
                if (idx == RUIDO){ //idx = -2 representa os ruidos filtrados
                    data[j] = Vec3b(255,255,255);
                }
                else{
                    data[j] = colorTab[idx];
                }
            }
        }
    }

    for(int i = 0; i < Kupdated; i++){
        circle(showClusters,clusters[i].center,2, Scalar(0,0,0));
        String msg = std::to_string(i);
        putText(showClusters,msg,clusters[i].center,
                CV_FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,0,0));
    }
    return showClusters;
}

//void clusterDetector::applyKmeansFilter(Mat binary, Mat labels, int idx){
//    //imshow("binary cru", binary);
//    cv::erode(binary, binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
//    cv::dilate(binary, binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
//    //imshow("binary filtered", binary);


//    int low = 50;
//    Mat canny;
//    cv::Canny(binary,canny,low,2*low);

//    //cv::dilate(binary, binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
//    //cv::erode(binary, binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

//    //imshow("binary canny", canny);

//    cv::vector<cv::vector<Point> > contours;
//    std::vector<RotatedRect> box_vector;
//    std::vector<cv::Vec4i> hierarchy;

//    cv::findContours(canny, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

//    cv::Mat cimage = cv::Mat::zeros(canny.size(), CV_8UC1);

//    for(size_t i = 0; i < contours.size(); i++)
//    {
//        size_t count = contours[i].size();
//        if( count < 6 ){
//            continue;
//        }
//        cv::RotatedRect box = cv::fitEllipse(contours[i]);

//        //Exclui candidatos muito pequenos.
//        //        if ((box.size.area()) < 10.0){
//        //            cv::Mat aux = cv::Mat::ones(binary.size(), CV_8UC1);
//        //            aux.setTo(Scalar(255));
//        //            cv::ellipse(aux, box.center, box.size*0.5f, box.angle, 0, 360, cv::Scalar(0,0,0), -1, CV_AA);
//        //            cv::bitwise_and(binary,aux,binary);
//        //            continue;
//        //        }

//        //cv::drawContours(canny,contours,i, Scalar(255,255,255));
//        if (box_vector.size() != 0){
//            int cont = 0;
//            for (int j = 0; j < box_vector.size(); j++){
//                if(this->distance(box.center.x, box_vector[j].center.x, box.center.y, box_vector[j].center.y) > 10){
//                    cont++;
//                }
//            }
//            if (cont == box_vector.size()){
//                box_vector.push_back(box);
//            }
//        }
//        else{
//            box_vector.push_back(box);
//        }

//    }
//    //cout << box_vector.size() << endl;
//    for (int i = 0; i < box_vector.size(); i++){
//        cv::ellipse(cimage, box_vector[i].center, box_vector[i].size*0.5f, box_vector[i].angle,
//                    0, 360, cv::Scalar(255,255,255), 1, CV_AA);
//    }
//    //imshow("binary apos AND", binary);
//    //imshow("elipses", cimage);

//    int count = 0;
//    int* pointer = labels.ptr<int>(count);
//    for (int i=0; i<convertSrc.rows; i++) {
//        // get the address of row j
//        uchar* dataBin = binary.ptr<uchar>(i);
//        Vec3b* data = convertSrc.ptr<Vec3b>(i);

//        for (int j=0; j<convertSrc.cols; j++) {
//            // process each pixel ---------------------
//            if (data[j].val[0] != 0 && data[j].val[1] != 0 && data[j].val[2] != 0){
//                if (pointer[count] == idx){
//                    if(dataBin[j] == 0){
//                        pointer[count] = 50; //ruido - nao pertence a nenhum grupo
//                    }
//                    else{

//                    }
//                }
//                count++;
//            }
//        }

//    }
//}
