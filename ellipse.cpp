#include "ellipse.h"


Mat ellipseDetector::run(Mat srcContours){

    cv::Mat canny_output;
    int low = 50;
    blur( srcContours, srcContours, Size(5,5) );
    cv::Canny(srcContours,canny_output,low,2*low);
    //imshow("canny cru", canny_output);

    //Morphological Transformation
    cv::dilate(canny_output, canny_output, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::erode(canny_output, canny_output, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    //cv::imshow("contours.jpg",canny_output);

    //FIND ELLIPSES
    cv::vector<cv::vector<Point> > contours;
    std::vector<RotatedRect> box_vector;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat cimage = cv::Mat::zeros(canny_output.size(), CV_8UC3);


    for(size_t i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        if( count < 6 )
            continue;

        cv::RotatedRect box = cv::fitEllipse(contours[i]);

        //Exclui candidatos pequenos.
        if ((box.size.area()) < 5000.0)
            continue;

        //Exclui candidatos desproporcionais
        float relation = box.size.width/box.size.height;
        const float PROPORCAO = 2.0;
        if (relation >= PROPORCAO || 1/relation >= PROPORCAO)
            continue;

        //cout<<box.center<< endl;
        //cout<<box.size.width << ' ' << box.size.height<<endl;
        //cout << cimage.cols << ' ' << cimage.rows << endl;
        //if(box.center.x + (box.size.width/2) >= cimage.cols || box.center.x - (box.size.width/2) < 0 ||
        //   box.center.y + (box.size.height/2) >= cimage.rows || box.center.y - (box.size.height/2) < 0 )
        //    continue;
        //Exclui candidatos com centro fora da imagem. Fazer dps um metodo que detecte se o rotatedRect está fora da imagem
        if(box.center.x >= cimage.cols || box.center.x  < 0 ||
                box.center.y >= cimage.rows || box.center.y < 0 )
            continue;

        box_vector.push_back(box);

    }

    //pega a maior elipse
    cv::RotatedRect bestBox = box_vector[0];
    for(size_t i = 1; i < box_vector.size(); i++){
        if(box_vector[i].size.area() > bestBox.size.area()){
            bestBox = box_vector[i];
        }
    }

    //segmenta imagem original
    Mat plate;
    db::plateBox = bestBox;
    cv::ellipse(cimage, bestBox.center, bestBox.size*0.5f, bestBox.angle, 0, 360, cv::Scalar(255,255,255), -1, CV_AA);
    cv::bitwise_and(db::src,cimage,plate);
    imshow("plate", plate);
    return plate;
}


