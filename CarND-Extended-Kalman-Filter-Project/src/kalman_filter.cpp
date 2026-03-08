#include "kalman_filter.h"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
#include <math.h>

#define PI 3.14159265

using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * convert current state x_ from cartesian to polar 
    * update the state by using Extended Kalman Filter equations
  */
  float rho = sqrt(x_[0]*x_[0] + x_[1]*x_[1]);
  float phi;
  if(fabs(x_[0]) < 0.00001) {
    cout << "Division by zero: fabs(x_[0]) < 0.00001" << endl;
  } else {
    phi = atan2(x_[1], x_[0]);
  }

  if(rho < 0.00001) {
    rho = 0.00001;
  }
  float rho_dot = (x_[0]*x_[2]+x_[1]*x_[3])/rho;

  VectorXd hx = VectorXd(3);
  hx << rho, phi, rho_dot;
  VectorXd y = z - hx;

  // angle normalization (-pi, pi)
  if(y[1] > 2*PI) {
    y[1] = fmod(y[1], 2*PI);
  }
  if(y[1] < -2*PI) {
    y[1] = fmod(y[1], -2*PI);
  }
  // std::cout << "phi: " << y[1] << std::endl;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
