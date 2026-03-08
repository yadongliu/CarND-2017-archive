#ifndef PID_H
#define PID_H

#include <vector>

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  double pre_cte;
  double sum_cte;

  // for twiddle
  int n_twiddle;
  bool twiddle_initialized; 
  int time_steps;
  double error; 
  double best_err; 
  std::vector<double> dp;
  bool skip_once;

  /*
  * Coefficients: use vector for its ease of use in Twiddle
  */ 
  std::vector<double> K;

  //double Kp;
  //double Ki;
  //double Kd;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  void Reset();

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

  /*
  * Calculate the total PID error.
  */
  void Twiddle();

};

#endif /* PID_H */
