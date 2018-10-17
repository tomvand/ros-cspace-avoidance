#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <vector>
#include <cmath>

namespace
{
  template <typename T>
  T bound(const T& val, const T& min, const T& max) {
    if(val < min) return min;
    if(val > max) return max;
    return val;
  }

  struct ImageParams {
    int width;
    int height;
    int ndisp;
    double f; // Focal length of downscaled(!) image
    double f_disp; // Focal length for disparity calculation
    double B;
  };

  class LUT
  {
  public:
    LUT(ImageParams ip, double rv) :
      ip(ip),
      rv(rv),
      lut_x1(ip.width, std::vector<int>(ip.ndisp)),
      lut_x2(ip.width, std::vector<int>(ip.ndisp)),
      lut_y1(ip.height, std::vector<int>(ip.ndisp)),
      lut_y2(ip.height, std::vector<int>(ip.ndisp)),
      lut_dnew(ip.ndisp)
    {
      // For LUT generation, refer to Matthies et al., 2014 "Stereo vision..."
      // x1, x2 LUT
      for(int x = 0; x < ip.width; ++x) {
        for(int d = 1; d < ip.ndisp; ++d) { // Caution: d = 0 unhandled!
          double cx = ip.width / 2.0;
          double u = x - cx;
          double zw = ip.f_disp * ip.B / d;
          double xw = u * zw / ip.f;
          double alpha = std::atan(zw / xw);
          double alpha1 = std::asin(rv / std::sqrt(zw * zw + xw * xw));
          double r1x = zw / std::tan(alpha + alpha1);
          double r2x = zw / std::tan(alpha - alpha1);
          double x1 = bound<double>(cx + ip.f * r1x / zw, 0, ip.width - 1);
          double x2 = bound<double>(cx + ip.f * r2x / zw, 0, ip.width - 1);
          lut_x1[x][d] = static_cast<int>(x1);
          lut_x2[x][d] = static_cast<int>(x2);
//          ROS_INFO("(%i,%i): x1 = %i, x2 = %i", x, d, lut_x1[x][d], lut_x2[x][d]);
        }
      }
      // y1, y2 LUT
      for(int y = 0; y < ip.height; ++y) {
        for(int d = 1; d < ip.ndisp; ++d) { // Caution: d = 0 unhandled!
          double cy = ip.height / 2.0;
          double v = y - cy;
          double zw = ip.f_disp * ip.B / d;
          double yw = v * zw / ip.f;
          double beta = std::atan(zw / yw);
          double beta1 = std::asin(rv / std::sqrt(zw * zw + yw * yw));
          double r3y = zw / std::tan(beta + beta1);
          double r4y = zw / std::tan(beta - beta1);
          double y1 = bound<double>(cy + ip.f * r3y / zw, 0, ip.height - 1);
          double y2 = bound<double>(cy + ip.f * r4y / zw, 0, ip.height - 1);
          lut_y1[y][d] = static_cast<int>(y1);
          lut_y2[y][d] = static_cast<int>(y2);
//          ROS_INFO("(%i,%i): y1 = %i, y2 = %i, zw = %f", y, d, lut_y1[y][d], lut_y2[y][d], zw);
        }
      }
      // dnew LUT
      for(int d = 1; d < ip.ndisp; ++d) { // Caution: d = 0 unhandled!
        double zw = ip.f_disp * ip.B / d;
        double znew = zw - rv;
        if(znew < NEAR_CLIP) znew = NEAR_CLIP;
        lut_dnew[d] = std::ceil(ip.f_disp * ip.B / znew);
      }
    }

    int x1(int x, int d) {
      assert(x >= 0 && x < ip.width && d > 0 && d < ip.ndisp);
      return lut_x1[x][d];
    }

    int x2(int x, int d) {
      assert(x >= 0 && x < ip.width && d > 0 && d < ip.ndisp);
      return lut_x2[x][d];
    }

    int y1(int y, int d) {
      assert(y >= 0 && y < ip.height && d > 0 && d < ip.ndisp);
      return lut_y1[y][d];
    }

    int y2(int y, int d) {
      assert(y >= 0 && y < ip.height && d > 0 && d < ip.ndisp);
      return lut_y2[y][d];
    }

    int dnew(int d) {
      assert(d > 0 && d < ip.ndisp);
      return lut_dnew[d];
    }

  private:
    std::vector<std::vector<int>> lut_x1;
    std::vector<std::vector<int>> lut_x2;
    std::vector<std::vector<int>> lut_y1;
    std::vector<std::vector<int>> lut_y2;
    std::vector<int> lut_dnew;
    ImageParams ip;
    double rv;
    const double NEAR_CLIP = 0.1;
  };


  class CSpaceExpander {
  public:
    CSpaceExpander(ImageParams ip, double rv) :
      ip(ip),
      lut(ip, rv)
      {}

    void expand(const cv::Mat_<float>& disp, cv::Mat_<float>& cspace) {
      assert(disp.cols == ip.width && disp.rows == ip.height);
      cv::Mat_<float> temp;
      disp.copyTo(temp);
      // For C-Space expansion procedure, refer to Matthies et al., 2014
      // Row-wise expansion
      for(int y = 0; y < disp.rows; ++y) {
        for(int x = 0; x < disp.cols; ++x) {
          int d = disp(y, x);
          if (d > 0 && d < ip.ndisp) {
            int x1 = lut.x1(x, d);
            int x2 = lut.x2(x, d);
            for(int x_write = x1; x_write <= x2; ++x_write) {
              if(!(d <= temp(y, x_write))) { // Negative test to handle NaNs!
                temp(y, x_write) = d;
              }
            }
          }
        }
      }
      // Column-wise expansion
      temp.copyTo(cspace);
      for(int x = 0; x < temp.cols; ++x) {
        for(int y = 0; y < temp.rows; ++y) {
          int d = temp(y, x);
          if(d > 0 && d < ip.ndisp) {
            int dnew = lut.dnew(d);
            int y1 = lut.y1(y, d);
            int y2 = lut.y2(y, d);
            for(int y_write = y1; y_write <= y2; ++y_write) {
              if(!(dnew <= cspace(y_write, x))) { // Negative test to handle NaNs!
                cspace(y_write, x) = dnew;
              }
            }
          }
        }
      }
    }
  private:
    ImageParams ip;
    LUT lut;
  };

  class CSpaceNode {
  public:
    CSpaceNode(ImageParams ip, double rv) :
      ce(ip, rv),
      it(nh)
    {
      sub = it.subscribe("/disp_map/image", 1, &CSpaceNode::image_callback, this);
    }
//  private:
    void image_callback(const sensor_msgs::ImageConstPtr& msg) {
      cv::Mat_<float> disp(msg->height, msg->width, (float*)(&(msg->data[0])));
      cv::Mat_<float> cspace;

      ce.expand(disp, cspace);

      cv::Mat_<float> debug;
      cv::vconcat(disp, cspace, debug);

      cv::imshow("cspace", debug / 63.0);
      cv::waitKey(1);
    }

    CSpaceExpander ce;
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber sub;
    image_transport::Publisher pub;
  };
} // namespace

int main(int argc, char **argv) {
  ros::init(argc, argv, "cspace");
  ImageParams ip = {
      .width = 96,
      .height = 96,
      .ndisp = 64,
      .f = 425 / 6, // Note: get from param for now, as camera_info does not arrive before construction...
      .f_disp = 425,
      .B = 0.20,
  };
  CSpaceNode c(ip, 1.0);
  ros::spin();
}
