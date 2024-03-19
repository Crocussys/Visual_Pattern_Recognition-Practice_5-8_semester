#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("../Visual_Pattern_Recognition-Practice_5-8_semester/image.png", IMREAD_COLOR);
    if (img.empty()){
        cout << "Изображение не загружено" << endl;
        return -1;
    }
    copyMakeBorder(img, img, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));
    Mat new_img, edges;
    namedWindow("Original", WINDOW_NORMAL);
    imshow("Original", img);
    GaussianBlur(img, new_img, Size(0, 0), 1);
    cvtColor(new_img, new_img, COLOR_BGR2GRAY);
    namedWindow("Processed", WINDOW_NORMAL);
    imshow("Processed", new_img);
    Canny(new_img, edges, 0, 0);
    namedWindow("Edges", WINDOW_NORMAL);
    imshow("Edges", edges);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> approx;
    findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
        if (fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx))
            continue;
        drawContours(img, contours, (int)i, Scalar(0, 255, 0), 1, LINE_8, hierarchy, 0);
        Moments m = moments(approx, true);
        Point p(m.m10/m.m00, m.m01/m.m00);
        if(approx.size() == 3){
            putText(img, "Triangle", p, 0, 0.5, Scalar(0, 255, 0));
        }else if(approx.size() > 3 && approx.size() < 7){
            putText(img, "Rectangle", p, 0, 0.5, Scalar(0, 255, 0));
        }else{
            putText(img, "Circle", p, 0, 0.5, Scalar(0, 255, 0));
        }
    }
    namedWindow("Contours", WINDOW_NORMAL);
    imshow("Contours", img);
    waitKey(0);
    return 0;
}
