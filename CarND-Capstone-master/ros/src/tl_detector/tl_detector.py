#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

import numpy as np
from scipy.spatial import distance, cKDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):

        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # define minimal distance we will consider traffic lights
        self.config['traffic_light_min_distance'] = 100

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # use case='sim' to load the classifier for the simulated scenario
        # use case='real' to load the classifier for the real scenario
        sim_or_site = rospy.get_param("~graph_case")
        self.light_classifier = TLClassifier(case=sim_or_site)
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # create a cKDTree from waypoints to increase a lookup speed
        feed = [self.extract_pose(waypoint.pose) for waypoint in waypoints.waypoints]
        self.waypoints = cKDTree(np.array(feed))

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # in case traffic light changed from what we sent previously reset counter
        if (self.last_wp, self.last_state) != (light_wp, state):
            self.state_count = 0
            self.last_wp, self.last_state = light_wp, state
            return

        # if we got same state < STATE_COUNT_THRESHOLD times - just increase the counter
        if self.state_count <= STATE_COUNT_THRESHOLD:
            self.state_count += 1

            # return if it's still not enough attempts
            if self.state_count < STATE_COUNT_THRESHOLD:
                return

        # in case we are getting same sate / light constantly no need to re-send info again
        if self.state_count > STATE_COUNT_THRESHOLD:
            return

        # if traffic light is not red - send -1
        if state != TrafficLight.RED:

            light_wp = -1

        # send info to channel
        self.upcoming_red_light_pub.publish(Int32(light_wp))

    def extract_pose(self, pose):
        """Extract coordinates from Pose object

        Args:
            pose (Pose): object for extraction

        Returns:
            dict: list of coordinates from the object

        """
        return [pose.pose.position.x, pose.pose.position.y]

    def get_closest_waypoint(self, pose_coord):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem

        Args:
            pose_coord (list): position coordinates to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        return self.waypoints.query(pose_coord, k=1)[1]

    def get_closest_light(self):
        """Identifies the closest light in front of a car

        Returns:
            int: index of the closest light in front of the car

        """
        car_coord = self.extract_pose(self.pose)
        car_wp_idx = self.get_closest_waypoint(car_coord)

        closest_idx = None
        for i, light in enumerate(self.lights):
            light_coord = self.extract_pose(light.pose)

            # check if traffic light is close enough to consider it
            dist = distance.euclidean(car_coord, light_coord)
            if dist > self.config['traffic_light_min_distance']:
                continue

            # in case car waypoint followed by light waypoint then car is in front so skip.
            light_wp_idx = self.get_closest_waypoint(light_coord)
            if car_wp_idx > light_wp_idx:
                continue

            # check if we should get this light as closest
            if closest_idx is None or light_wp_idx < closest_idx:
                closest_idx = i

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        res_unknown = (-1, TrafficLight.UNKNOWN)

        # if we have no waypoints - no info for traffic lights
        if self.waypoints is None:
            return res_unknown

        # if no lights defined - no info
        if not self.lights:
            return res_unknown

        # if no current position defined - no info
        if self.pose is None:
            return res_unknown

        # get closest light index
        light_idx = self.get_closest_light()
        if light_idx is None:
            return res_unknown

        # get stop line related to light 
        stop_line_coord = self.config['stop_line_positions'][light_idx]
        stop_line_wp = self.get_closest_waypoint(stop_line_coord)
        state = self.get_light_state(self.lights[light_idx])

        return stop_line_wp, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
