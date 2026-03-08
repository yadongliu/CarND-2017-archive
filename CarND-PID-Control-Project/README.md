# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= 3.5
* make >= 4.1
* gcc/g++ >= 5.4
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

## Project Reflection

By using the provided PID header file, it is fairly straightforward to implement the PID controller. 
Essentially you need to set the three error coefficients in Init() function, update internal errors by 
passing CTE (cross track error) to the function updateError(), the third function TotalError() returns the total error
from which you would adjust your steering angle.  

The most important aspect of this project is to come up with the right coefficients Kp, Kd and Ki. 
Firt I used a manual process to find a reasonable Kp, which I basically started at 0.0 and gradually
increased to 0.06, while keeping Kd and Ki at 0.0. This alone would get the simulator to drive over 
the bridge but the car oscillated a lot. That means we need to add a differential error to offset the gyration in 
errors. I manually tuned the PID to use 0.6 for Kd (10 times the Kp) and also use a very small Ki to 
take into consideration of accumulated CTE. This was able to get the car finish the entire track but the tires still
touched the non-drivable portions of the track at the most curvy places. This tells me that Kd is sufficiently large 
enough to turn the vehicle in those cases. I needed to further tune the error coefficients.

At this point, I started adding code to implement Twiddle. The basic idea is try to find an optimal set
of error coefficients such that accumulated sqaured CTE would be the smallest. Through this process, I came up with
the final coefficients.