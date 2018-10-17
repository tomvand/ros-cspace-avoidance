#include <vector>
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <cstdio>

namespace
{
  template <typename T>
  T bound(const T& val, const T &min, const T& max) {
    if(val < min) return min;
    if(val > max) return max;
    return val;
  }

  struct ImageParams {
    int width;
    int height;
    int ndisp;
    double f;
    double B;
  };

  class LUT
  {
  public:
    LUT(ImageParams ip, float rv) :
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
          double zw = ip.f * ip.B / d;
          double xw = u * zw / ip.f;
          double alpha = std::atan(zw / xw);
          double alpha1 = std::asin(rv / std::sqrt(zw * zw + xw * xw));
          double r1x = zw / std::tan(alpha + alpha1);
          double r2x = zw / std::tan(alpha - alpha1);
          double x1 = bound<double>(cx + ip.f * r1x / zw, 0, ip.width - 1);
          double x2 = bound<double>(cx + ip.f * r2x / zw, 0, ip.width - 1);
          lut_x1[x][d] = static_cast<int>(x1);
          lut_x2[x][d] = static_cast<int>(x2);
        }
      }
      // y1, y2 LUT
      for(int y = 0; y < ip.height; ++y) {
        for(int d = 1; d < ip.ndisp; ++d) { // Caution: d = 0 unhandled!
          double cy = ip.height / 2.0;
          double v = y - cy;
          double zw = ip.f * ip.B / d;
          double yw = v * zw / ip.f;
          double beta = std::atan(zw / yw);
          double beta1 = std::asin(rv / std::sqrt(zw * zw + yw * yw));
          double r3y = zw / std::tan(beta + beta1);
          double r4y = zw / std::tan(beta - beta1);
          double y1 = bound<double>(cy + ip.f * r3y / zw, 0, ip.height - 1);
          double y2 = bound<double>(cy + ip.f * r4y / zw, 0, ip.height - 1);
          lut_y1[y][d] = static_cast<int>(y1);
          lut_y2[y][d] = static_cast<int>(y2);
        }
      }
      // dnew LUT
      for(int d = 1; d < ip.ndisp; ++d) { // Caution: d = 0 unhandled!
        double zw = ip.f * ip.B / d;
        double znew = zw - rv;
        if(znew < NEAR_CLIP) znew = NEAR_CLIP;
        lut_dnew[d] = std::ceil(ip.f * ip.B / znew);
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

} // namespace

int main(int argc, char **argv) {
  ros::init(argc, argv, "cspace");
  ros::NodeHandle nh;
  ImageParams ip = {
      .width = 96,
      .height = 96,
      .ndisp = 64,
      .f = 425,
      .B = 0.20,
  };
  LUT lut(ip, 1.0);
}
