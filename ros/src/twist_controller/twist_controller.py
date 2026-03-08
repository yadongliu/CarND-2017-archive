import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH     = 0.44704
MAX_VEL     = 40.0

#any suggestion with these parameters?
Kp = 1
Ki = 0.003
Kd = 0.25

class Controller(object):
    def __init__(self, *args, **kwargs):

        self.vehicle_mass    = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity   = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband  = rospy.get_param('~brake_deadband', .1)
        self.decel_limit     = rospy.get_param('~decel_limit', -5)
        self.accel_limit     = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius    = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base      = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio     = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel   = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        
        self.low_pass = LowPassFilter(0.1,1.0) # (?)
        self.pid_controller  = PID(Kp,Ki,Kd)
        self.yaw_controller  = YawController(self.wheel_base, self.steer_ratio, 0.0, self.max_lat_accel, self.max_steer_angle)  
        self.timestamp       = rospy.get_time()
        MAX_VEL = rospy.get_param('/waypoint_loader/velocity') # km/h

    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_status_enabled, dt):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        throttle = 0.0
        brake    = 0.0
        steering = 0.0

        #How can we improve this part?
        if dbw_status_enabled: 
        
            # compute throttle considering the speed limit constraint
            cte = min(linear_velocity, MAX_VEL) - current_velocity
            throttle = self.pid_controller.step(cte, dt)
            
            #throttle can not be greater than 1
            if(throttle>1):
                throttle = 1
                
            brake = 0.0 #we have to implement this when the tl detection is done...
            if linear_velocity < 0.01 and current_velocity < 0.1:
                # full stop
                throttle = 0.0
                brake = 500
            else:
                if throttle >= 0.001:
                    brake = 0.0
                elif throttle < -self.brake_deadband:
                    brake = -throttle * self.vehicle_mass * self.wheel_radius
                    throttle = 0.0
                else:
                    brake = 0.0
                    throttle = 0.0
            #rospy.loginfo('### Throttle %.3f, Brake: %.3f ', throttle, brake)
            
            steering = self.low_pass.filt(self.yaw_controller.get_steering(linear_velocity,angular_velocity,current_velocity))
            
        else:
            self.pid_controller.reset()
            
        return throttle, brake, steering
