#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace Tools {

/**
 * A helper method to calculate RMSE.
 */
VectorXd calculateRMSE(const std::vector<VectorXd> &estimations,
                       const std::vector<VectorXd> &ground_truth);

/**
 * Generate sigma points.
 */
MatrixXd generateSigmaPoints(const VectorXd &x, const MatrixXd &P,
                             double lambda);

/**
 * Generate sigma points.
 */
MatrixXd generateAugmentedSigmaPoints(const VectorXd &x, const MatrixXd &P,
                                      double lambda, const VectorXd &sigma);
}

#endif /* TOOLS_H_ */