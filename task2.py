import heapq
import math
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

#Write description of GridWorldG7 class
class GridWorldG7:
    """
    This class is used to create a grid world with obstacles, robot, and goal position. 
    It can be used to find the shortest path from the robot to the goal position using the A* search algorithm,
    display the map, and decode the motions into a set of motions."""
    def __init__(
        self, map_size, grid_size, obstacle_positions, robot_position, goal_position
    ):
        self.map_size = map_size
        self.grid_size = grid_size
        self.obstacle_size = (self.grid_size, self.grid_size)
        self.grid_rows = int(self.map_size / self.grid_size)
        self.grid_cols = int(self.map_size / self.grid_size)
        self.obstacle_positions = obstacle_positions 
        self.new_obstacle_positions = [] # used to store the new obstacle positions (to make them have different color)
        self.robot_position = robot_position
        self.robot_direction = 90 # 0: right, 90: up, 180: left, 270: down. The default direction is up.
        self.goal_position = goal_position
        self.path = [] # used to store the path
        self.path_color = (0, 255, 0) # used to set the color of the path
        self.map_image = (
            np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 200
        )
        self.motions = [] # used to store the set of motions. Which can be used to control the robot.

    def draw_grid(self):
        color = (100, 100, 100)
        for x in range(0, self.map_image.shape[1], self.grid_size):
            cv2.line(self.map_image, (x, 0), (x, self.map_image.shape[0]), color, 1)
        for y in range(0, self.map_image.shape[0], self.grid_size):
            cv2.line(self.map_image, (0, y), (self.map_image.shape[1], y), color, 1)

    #This method is used to draw a rectangle on the map image based on the obstacle positions and color provided.
    def draw_rectangle(self, obstacle_positions, color):
        for obstacle in obstacle_positions:
            row, col = obstacle
            x1 = col * self.grid_size
            y1 = row * self.grid_size
            x2 = x1 + self.obstacle_size[1]
            y2 = y1 + self.obstacle_size[0]
            cv2.rectangle(self.map_image, (x1, y1), (x2, y2), color, -1)

    #This method is used to add obstacles to the map image.
    def add_obstacles(self):
        self.draw_rectangle(self.obstacle_positions, (0, 0, 200)) # red color for the known obstacles
        self.draw_rectangle(self.new_obstacle_positions, (200, 0, 0)) # blue color for the new obstacles

    def add_robot(self):
        row, col = self.robot_position
        center_x = int((col + 0.5) * self.grid_size)
        center_y = int((row + 0.5) * self.grid_size)
        radius = int(self.grid_size / 3)
        cv2.circle(self.map_image, (center_x, center_y), radius, (0, 0, 0), -1)
        line_length = int(radius * 0.8)
        angle_radians = math.radians(self.robot_direction)
        end_x = int(center_x + line_length * math.cos(angle_radians))
        end_y = int(center_y - line_length * math.sin(angle_radians))
        cv2.line(
            self.map_image, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2
        )

    def add_goal(self):
        row, col = self.goal_position
        center_x = int((col + 0.5) * self.grid_size)
        center_y = int((row + 0.5) * self.grid_size)
        radius = int(self.grid_size / 4)
        cv2.circle(self.map_image, (center_x, center_y), radius, (0, 255, 0), -1)

    #This method is used to calculate the heuristic value between two points.
    #We use the Manhattan distance as the heuristic value.
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, current):
        list_of_obstacles = self.obstacle_positions + self.new_obstacle_positions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        result = []
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if (
                0 <= nx < self.map_size
                and 0 <= ny < self.map_size
                and (nx, ny) not in list_of_obstacles
            ):
                result.append((nx, ny))
        return result

    """This method is used to find the shortest path from the robot position to the goal position using the A* search algorithm.
    It returns the came_from dictionary and the cost_so_far dictionary.
    - The came_from dictionary is used to store the parent of each node in the path.
    - The cost_so_far dictionary is used to store the cost to reach each node.
    We implemented the A* search algorithm using the heapq module to create a priority queue.
    The priority queue is used to store the nodes to be explored based on the priority value.
    The priority value is calculated based on the cost to reach the node and the heuristic value.
    We used the Manhattan distance as the heuristic value.
    We also used the came_from dictionary to store the parent of each node in the path.
    The cost_so_far dictionary is used to store the cost to reach each node.
    We explored the nodes in the priority queue and updated the cost_so_far and came_from dictionaries accordingly.
    When the goal position is reached, we break the loop and reconstruct the path using the came_from dictionary."""
    
    def a_star_search(self):
        frontier = []
        heapq.heappush(frontier, (0, self.robot_position))
        came_from = {self.robot_position: None}# parent of the robot position is None
        cost_so_far = {self.robot_position: 0}# cost to reach the robot position is 0

        while frontier:
            _, current = heapq.heappop(frontier)# get the node with the lowest priority

            if current == self.goal_position:# if the goal position is reached, break the loop
                break

            for next in self.neighbors(current):# explore the neighbors of the current node
                new_cost = cost_so_far[current] + 1# cost to reach the next node is the cost to reach the current node + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:# if the new cost is less than the cost to reach the next node
                    cost_so_far[next] = new_cost# update the cost to reach the next node
                    priority = new_cost + self.heuristic(self.goal_position, next)# calculate the priority value
                    heapq.heappush(frontier, (priority, next))# add the next node to the priority queue
                    came_from[next] = current# update the parent of the next node

        return came_from, cost_so_far

    #This method is used to reconstruct the path from the goal position to the robot position using the came_from dictionary.
    def get_path(self, came_from):
        path = []
        current = self.goal_position# start from the goal position
        while current != self.robot_position:# until the robot position is reached
            path.append(current)# add the current node to the path
            current = came_from[current]# move to the parent of the current node
        path.append(self.robot_position)# add the robot position to the path
        self.path = path[::-1]# reverse the path to get the path from the robot position to the goal position

    #This method is used to add the path to the map image.
    def add_path(self, update_path = False):
        for i in range(1, len(self.path)):
            current = self.path[i - 1]
            next = self.path[i]
            x1 = int((current[1] + 0.5) * self.grid_size)
            y1 = int((current[0] + 0.5) * self.grid_size)
            x2 = int((next[1] + 0.5) * self.grid_size)
            y2 = int((next[0] + 0.5) * self.grid_size)
            if update_path:# if update_path is True, set the path color to blue
                self.path_color = (255, 0, 0)
            cv2.line(self.map_image, (x1, y1), (x2, y2), self.path_color, 2)

    #This method is used to decode the motions into a set of motions.
    def decode_motions(self):
        self.motions = []
        direction_map = {
            90: "up",
            270: "down",
            0: "right",
            180: "left",
        }
        current_direction = direction_map[self.robot_direction]
        # motion_map is used to map the current direction and the next direction to the motion
        motion_map = {
            ("up", "right"): "turn_right",
            ("up", "left"): "turn_left",
            ("down", "right"): "turn_left",
            ("down", "left"): "turn_right",
            ("right", "down"): "turn_right",
            ("right", "up"): "turn_left",
            ("left", "down"): "turn_left",
            ("left", "up"): "turn_right",
            ("up", "up"): "move_forward",
            ("down", "down"): "move_forward",
            ("right", "right"): "move_forward",
            ("left", "left"): "move_forward",
        }
        # direction_delta_map is used to map the change in x and y to the direction 
        direction_delta_map = {
            (0, 1): "right",
            (0, -1): "left",
            (1, 0): "down",
            (-1, 0): "up",
        }
        
        for i in range(1, len(self.path)):
            current = self.path[i - 1]
            next = self.path[i]
            dx = next[0] - current[0]
            dy = next[1] - current[1]
            next_direction = direction_delta_map[(dx, dy)]# get the next direction

            motion = motion_map[(current_direction, next_direction)]# get the motion based on the current and next direction
            self.motions.append(motion)
            current_direction = next_direction# update the current direction

    #This method is used to update the robot position based on the path.
    def update_robot_position(self):
        self.robot_position = self.path.pop(0) # move to the next position in the path
        self.update_map() # update the map after updating the robot position

    #This method is used to update the obstacle positions and update the map.
    def update_obstacle_positions(self, new_obstacle_position):
        self.new_obstacle_positions.append(new_obstacle_position)
        self.update_map()

    def display_map(self):
        cv2.startWindowThread()
        cv2.imshow("Gridworld", self.map_image)
        cv2.waitKey(1)

    #This method is used to update the map image with the grid, obstacles, robot, goal, and path.
    def update_map(self):
        self.draw_grid()
        self.add_obstacles()
        self.add_robot()
        self.add_goal()
        self.add_path()
        self.display_map()

    #This method is used to run the grid world simulation.
    #It draws the grid, adds obstacles, robot, and goal to the map image.
    #It then runs the A* search algorithm to find the shortest path from the robot to the goal.
    #If a path is found, it displays the path and decodes the motions into a set of motions.
    def run(self, update_path = False):
        self.draw_grid()
        self.add_obstacles()
        self.add_robot()
        self.add_goal()
        came_from, cost_so_far = self.a_star_search()
        if self.goal_position in came_from:# if the goal position is reachable
            print("Solution found!")
            print("Cost:", cost_so_far[self.goal_position])
            self.get_path(came_from)# get the path from the came_from dictionary
            for i in self.path:
                print(i)
            self.add_path(update_path)
            self.decode_motions()
            print(f"Set of motions: {self.motions}")
        else:
            print("No path found!")
        self.display_map()

#This class is used to represent an action with a state and action time.
#It has methods to start the action and check if the action is done.
class Action:
    def __init__(self, state, action_time):
        self.state = state
        self.start_time = None# when the action starts
        self.action_time = action_time# time to perform the action

    def start(self):
        self.start_time = time.time() # set the start time to the current time

    def is_done(self):
        # check if the current time is greater than the start time + action time
        # if yes, the action is done
        return time.time() > (self.start_time + self.action_time) 

class TurtlebotController(Node):
    def __init__(self):
        super().__init__("turtlebot_controller")

        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.state = None

        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.bridge = CvBridge()

        # Thresholds for detecting obstacles and walls
        self.red_threshold = 300000
        self.brown_threshold = 240000
        self.sum_threshold = 220000

        # Parameters for controlling the robot
        # To get the best performance, the value of these parameters need to be adjusted based on system
        # Note: the performace will be different on different systems and even the case you are recording the video or not
        # In case of recording the video, there will be some delay in the video stream, so you need to increase the time for actions
        self.moving_speed = 0.4 # speed of the robot
        self.moving_time = 1.15 # time to move forward
        self.turning_time = 4.2 # time to turn right or left
        self.moving_after_turn_time = 1.4 # time to move forward after turning
        self.idle_time = 0.75 # time to wait after each action

        obstacle_positions = [(1, 1), (0, 9), (4, 1), (4, 7), (8, 3), (8, 6)]
        robot_position = (4, 5)
        goal_position = (8, 8)
        self.grid_world = GridWorldG7(
            470, 47, obstacle_positions, robot_position, goal_position
        )
        self.grid_world.run() # run the grid world simulation
        self.action_stack = None # used to store the set of actions
        self.convert_motion_to_action() # convert the motions into a set of actions
        self.current_action = None

    #This method is used to convert the set of motions into a set of actions.
    #It iterates through the set of motions and adds the corresponding actions to the action stack.
    def convert_motion_to_action(self):
        self.action_stack = []
        moving_path = self.grid_world.motions.copy() # get the set of motions. Need to copy to avoid changing the original set
        while len(moving_path) > 0:# while there are motions in the set
            motion = moving_path.pop(0) # get the first motion
            match motion:
                case "move_forward":
                    self.action_stack.append(Action("move_forward", self.moving_time))
                    self.action_stack.append(Action("idle", self.idle_time))
                case "turn_right":
                    self.action_stack.append(Action("turn_right", self.turning_time))
                    self.action_stack.append(Action("idle", self.idle_time))
                    self.action_stack.append(Action("move_forward", self.moving_after_turn_time))
                    self.action_stack.append(Action("idle", self.idle_time))
                case "turn_left":
                    self.action_stack.append(Action("turn_left", self.turning_time))
                    self.action_stack.append(Action("idle", self.idle_time))
                    self.action_stack.append(Action("move_forward", self.moving_after_turn_time))
                    self.action_stack.append(Action("idle", self.idle_time))

    #This method is used to update the direction of the robot based on the action.
    def update_robot_direction(self, action):
        if action == "turn_right":
            self.grid_world.robot_direction -= 90 # turn right
        elif action == "turn_left":
            self.grid_world.robot_direction += 90 # turn left

    #This method is used to update the position of the robot in the grid world.
    def update_robot_position(self):
        self.grid_world.update_robot_position()
        
    #This method is used to update the state of the robot based on the action.
    def update_state(self):
        msg = Twist()

        if self.state == "move_forward":
            msg.linear.x = self.moving_speed
            msg.angular.z = 0.0
        elif self.state == "turn_right":
            msg.linear.x = 0.0
            msg.angular.z = -1 * self.moving_speed
        elif self.state == "turn_left":
            msg.linear.x = 0.0
            msg.angular.z = self.moving_speed
        elif self.state == "idle":
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.publisher_.publish(msg)

    #This method is used to detect obstacles based on the largest area of red color in the image.
    def detect_obstacle(self, largest_area_red):
        if largest_area_red > self.red_threshold: # if the largest area of red color is greater than the threshold
            return True # obstacle detected
        return False
    
    #This method is used to get the position of the obstacle based on the robot position and direction.
    def get_obstacle_position(self):
        x, y = self.grid_world.robot_position
        direction = self.grid_world.robot_direction
        if direction == 0:
            return (x, y + 1)
        elif direction == 90:
            return (x + 1, y)
        elif direction == 180:
            return (x, y - 1)
        elif direction == 270:
            return (x - 1, y)
        return None
    
    #This method is used to check if the obstacle position is a new obstacle.
    def check_new_obstacle(self, obstacle_position):
        if obstacle_position in self.grid_world.obstacle_positions:
            return False
        return True
    
    #This method is used to check if the robot is blocked by an obstacle.
    def check_blocked_by_obstacle(self, obstacle_position):
        if obstacle_position in self.grid_world.path:
            return True
        return False
    #This method is used to display the camera image and detect obstacles.
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error("Failed to convert image: " + str(e))
            return
       
        # Simple Image Processing tasks:
        # - Convert the image to HSV
        # - Define the lower and upper bounds for red and brown colors
        # - Create masks for red and brown colors
        # - Find contours for red and brown areas
        # - Get the largest area for red and brown colors
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        red_areas = cv2.bitwise_and(cv_image, cv_image, mask=mask_red)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_areas = cv2.bitwise_and(cv_image, cv_image, mask=mask_brown)
        contours_red, _ = cv2.findContours(
            mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_brown, _ = cv2.findContours(
            mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_area_red = (
            max(cv2.contourArea(contour) for contour in contours_red)
            if contours_red
            else 0
        )
        largest_area_brown = (
            max(cv2.contourArea(contour) for contour in contours_brown)
            if contours_brown
            else 0
        )

        # Robot control logic:
        # - If the robot is not moving and there are actions in the action stack, pop the first action and start it
        # - If the robot is moving and the current action is done, check for obstacles
        # - If a new obstacle is detected, update the grid world and check if the robot is blocked by the obstacle
        # - If the robot is blocked by the obstacle, update the path in the grid world and convert the motions to actions
        # - If there are actions in the action stack, pop the first action and start it
        # - If there are no actions in the action stack, set the current action to None and the state to idle
        if self.current_action is None and len(self.action_stack) > 0:
            self.current_action = self.action_stack.pop(0)
            self.current_action.start()
            self.state = self.current_action.state
            self.update_robot_direction(self.state)
            self.update_robot_position()
            print("Moving: ", self.state)

        if self.current_action is not None and self.current_action.is_done():
            if self.detect_obstacle(largest_area_red):# if an obstacle is detected
                    obstacle_position = self.get_obstacle_position()# get the position of the obstacle
                    if self.check_new_obstacle(obstacle_position):# if the obstacle is new
                        print(f"New obstacle detected at {obstacle_position}!")
                        print("Update grid world!")
                        self.grid_world.update_obstacle_positions(obstacle_position)# update the grid world with the new obstacle
                        if self.check_blocked_by_obstacle(obstacle_position):# if the robot is blocked by the obstacle
                            print(f"Blocked by obstacle at {obstacle_position}, finding new path...")
                            self.grid_world.run(update_path=True)# find a new path
                            self.convert_motion_to_action()# convert the motions to actions
                            self.grid_world.path.pop(0)# remove the first position in the path (current robot position)
                    else:
                        print(f"Known obstacle at {obstacle_position}, continue!")# if the obstacle is known, continue
            if self.state == "move_forward":# if the robot is moving forward
                self.update_robot_position()# update the robot position
            if len(self.action_stack) > 0:# if there are actions in the action stack
                self.current_action = self.action_stack.pop(0)# pop the first action
                self.current_action.start()# start the action
                self.state = self.current_action.state# update the state
                self.update_robot_direction(self.state)# update the robot direction
                if self.state != "idle":
                    print("Moving: ", self.state)
            else:
                self.current_action = None
                self.state = "idle"

        self.update_state()

        #Simple Image Processing tasks to display the camera image
        height, width, _ = cv_image.shape

        new_width = int(width * 0.5)
        new_height = int(height * 0.5)

        cv_image = cv2.resize(cv_image, (new_width, new_height))
        red_areas = cv2.resize(red_areas, (new_width, new_height))
        brown_areas = cv2.resize(brown_areas, (new_width, new_height))

        dual_images = cv2.hconcat([cv_image, red_areas, brown_areas])

        cv2.putText(
            dual_images,
            f"State: {self.state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Camera Image", dual_images)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    turtlebot_controller = TurtlebotController()
    rclpy.spin(turtlebot_controller)
    turtlebot_controller.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
