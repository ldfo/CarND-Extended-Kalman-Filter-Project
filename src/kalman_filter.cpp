#include <math.h>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

static VectorXd _cartesian_to_polar(const VectorXd &x_state);
static VectorXd _innovate_polar(const VectorXd &z, const VectorXd &z_pred);

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
	// predict state & covariance
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;

	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd PHt = P_ * Ht;
	MatrixXd S = H_ * PHt + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //update state with kalman filter equations
  // convert radar measurements from cartesian to polar
  VectorXd z_pred = _cartesian_to_polar(x_);
  VectorXd y = _innovate_polar(z, z_pred);

  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();

  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}


static VectorXd _cartesian_to_polar(const VectorXd &x_state)
{
	const float EPSILON	= 0.00001;
	float px, py, vx, vy;
  float rho, phi, rho_dot;

	px = x_state[0];
	py = x_state[1];
	vx = x_state[2];
	vy = x_state[3];
  phi = atan2(py, px);
	rho = sqrt(px*px + py*py);

	if(rho < EPSILON)
		rho = EPSILON;

	rho_dot = (px * vx + py * vy) / rho;

	VectorXd z_pred = VectorXd(3);
	z_pred << rho, phi, rho_dot;

	return z_pred;
}

static VectorXd _innovate_polar(const VectorXd &z, const VectorXd &z_pred)
{
	  VectorXd y = z - z_pred;
	  while(y(1) > M_PI){
	    y(1) -= M_PI*2;
	  }

	  while(y(1) < -M_PI){
	    y(1) += M_PI*2;
	  }
	  return y;
}
