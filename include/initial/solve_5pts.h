#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
//#include <ros/console.h>

using namespace Eigen;
using namespace std;

/* Reference: An efficient solution to the five-point relative pose problem */
/* Reference: https://zhuanlan.zhihu.com/p/374399877 */
class MotionEstimator
{
public:
  bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);

private:
  double testTriangulation(const vector<cv::Point2f> &l,
                           const vector<cv::Point2f> &r,
                           cv::Mat_<double> R, cv::Mat_<double> t);
  void decomposeE(cv::Mat E,
                  cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                  cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};
