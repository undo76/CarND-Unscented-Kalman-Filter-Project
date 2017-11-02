#include "../tools.h"
#include <iostream>
#include "./catch.hpp"

TEST_CASE("RMSE working") {
  std::vector<VectorXd> estimations;
  std::vector<VectorXd> ground_truth;

  SECTION("One measure without error") {
    VectorXd measure = VectorXd(6);
    measure << 1, 1, 1, 1, 1, 1;

    estimations.push_back(measure);
    ground_truth.push_back(measure);

    VectorXd rmse = Tools::calculateRMSE(estimations, ground_truth);

    REQUIRE(rmse.size() == measure.size());
    REQUIRE(rmse(0) == 0);
  }
}

TEST_CASE("Angle normalization") {
  double angle;
  SECTION("Out of bounds positive") {
    angle = M_PI + 1;
    Tools::normalizeAngle(angle);
    REQUIRE(angle > -M_PI);
    REQUIRE(angle < M_PI);
  }

  SECTION("Out of bounds negative") {
    angle = -M_PI - 10;
    Tools::normalizeAngle(angle);
    REQUIRE(angle > -M_PI);
    REQUIRE(angle < M_PI);
  }

  SECTION("In bounds") {
    angle = M_PI - 1;
    Tools::normalizeAngle(angle);
    REQUIRE(angle > -M_PI);
    REQUIRE(angle < M_PI);
    REQUIRE(angle == M_PI - 1);
  }
}

TEST_CASE("Sigma points") {
  int n_x = 5;
  VectorXd x = VectorXd(n_x);
  x << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;

  // set example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);

  // clang-format off
  P <<  0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
       -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
        0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
       -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
       -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  // clang-format on

  SECTION("No augmentation") {
    double lambda = 3 - n_x;
    MatrixXd Xsig = Tools::generateSigmaPoints(x, P, lambda);

    // clang-format off
    MatrixXd Xsig_expected(n_x, 2 * n_x + 1);
    Xsig_expected <<
      5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,
        1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,
      2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,
      0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,
      0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879;
    // clang-format on

    REQUIRE((Xsig - Xsig_expected).norm() < 1e-3);
  }

  SECTION("With augmentation") {
    VectorXd sigma(2);
    sigma << .2, .2;

    int n_aug = x.size() + sigma.size();
    double lambda = 3 - n_aug;

    MatrixXd Xsig = Tools::generateAugmentedSigmaPoints(x, P, lambda, sigma);

    // clang-format off
    MatrixXd Xsig_expected(n_aug, 2 * n_aug + 1);
    Xsig_expected <<
      5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
        1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
      2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
      0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
      0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
           0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
           0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;
    // clang-format on

    REQUIRE((Xsig - Xsig_expected).norm() < 1e-4);
  }
}
