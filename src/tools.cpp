#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

namespace Tools {

VectorXd calculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth) {
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return VectorXd(1);
  }

  VectorXd rmse(estimations[0].size());
  rmse.fill(0);

  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array().square();
    rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd generateSigmaPoints(const VectorXd &x, const MatrixXd &P,
                             double lambda) {
  int n_x = x.size();
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

  // calculate square root of P
  MatrixXd A = P.llt().matrixL();

  Xsig.col(0) = x;
  for (int i = 0; i < n_x; i++) {
    VectorXd dx = sqrt(lambda + n_x) * A.col(i);
    Xsig.col(i + 1) = x + dx;
    Xsig.col(i + 1 + n_x) = x - dx;
  }

  return Xsig;
}

MatrixXd generateAugmentedSigmaPoints(const VectorXd &x, const MatrixXd &P,
                                      double lambda, const VectorXd &sigma) {
  int n_x = x.size();
  int n_sigma = sigma.size();
  int n_aug = n_x + n_sigma;
  VectorXd x_aug = VectorXd(n_aug);
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  // MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  // create augmented mean state [...x, ...sigma]
  x_aug.head(x.size()) = x;
  x_aug.tail(sigma.size()).fill(0.);

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x, n_x) = P;

  for (int i = 0; i < n_sigma; i++) {
    P_aug(n_x + i, n_x + i) = sigma(i) * sigma(i);
  }

  return generateSigmaPoints(x_aug, P_aug, lambda);
}

void normalizeAngle(double &angle) {
  while (angle > M_PI) angle -= 2. * M_PI;
  while (angle < -M_PI) angle += 2. * M_PI;
}

double NIS(const VectorXd &z_diff, const MatrixXd &Sinv) {
  return z_diff.transpose() * Sinv * z_diff;
}

}