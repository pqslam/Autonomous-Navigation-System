import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
import time
import heapq


class GridWorldG7:
    def __init__(
        self, map_size, grid_size, obstacle_positions, robot_position, goal_position
    ):
        self.map_size = map_size
        self.grid_size = grid_size
        self.obstacle_size = (self.grid_size, self.grid_size)
        self.grid_rows = int(self.map_size / self.grid_size)
        self.grid_cols = int(self.map_size / self.grid_size)
        self.obstacle_positions = obstacle_positions
        self.robot_position = robot_position
        self.robot_direction = 90
        self.goal_position = goal_position
        self.path = []
        self.map_image = (
            np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 200
        )
        self.motions = []

    def draw_grid(self):
        color = (100, 100, 100)
        for x in range(0, self.map_image.shape[1], self.grid_size):
            cv2.line(self.map_image, (x, 0), (x, self.map_image.shape[0]), color, 1)
        for y in range(0, self.map_image.shape[0], self.grid_size):
            cv2.line(self.map_image, (0, y), (self.map_image.shape[1], y), color, 1)

    def add_obstacles(self):
        for obstacle in self.obstacle_positions:
            row, col = obstacle
            x1 = col * self.grid_size
            y1 = row * self.grid_size
            x2 = x1 + self.obstacle_size[1]
            y2 = y1 + self.obstacle_size[0]
            cv2.rectangle(self.map_image, (x1, y1), (x2, y2), (0, 0, 200), -1)

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

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, current):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        result = []
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if (
                0 <= nx < self.map_size
                and 0 <= ny < self.map_size
                and (nx, ny) not in self.obstacle_positions
            ):
                result.append((nx, ny))
        return result

    def a_star_search(self):
        frontier = []
        heapq.heappush(frontier, (0, self.robot_position))
        came_from = {self.robot_position: None}
        cost_so_far = {self.robot_position: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == self.goal_position:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(self.goal_position, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        return came_from, cost_so_far

    def get_path(self, came_from):
        path = []
        current = self.goal_position
        while current != self.robot_position:
            path.append(current)
            current = came_from[current]
        path.append(self.robot_position)
        self.path = path[::-1]


    def add_path(self):
        for i in range(1, len(self.path)):
            current = self.path[i - 1]
            next = self.path[i]
            x1 = int((current[1] + 0.5) * self.grid_size)
            y1 = int((current[0] + 0.5) * self.grid_size)
            x2 = int((next[1] + 0.5) * self.grid_size)
            y2 = int((next[0] + 0.5) * self.grid_size)
            cv2.line(self.map_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def decode_motions(self):
        current_direction = "up"
        direction_map = {
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

        for i in range(1, len(self.path)):
            current = self.path[i - 1]
            next = self.path[i]
            dx = next[0] - current[0]
            dy = next[1] - current[1]
            if dx == 0 and dy == 1:
                next_direction = "right"
            elif dx == 0 and dy == -1:
                next_direction = "left"
            elif dx == 1 and dy == 0:
                next_direction = "down"
            elif dx == -1 and dy == 0:
                next_direction = "up"

            motion = direction_map[(current_direction, next_direction)]
            self.motions.append(motion)
            current_direction = next_direction

    def update_robot_position(self):
        self.robot_position = self.path.pop(0)
        self.update_map()

    def display_map(self):
        cv2.startWindowThread()
        cv2.imshow("Gridworld", self.map_image)
        cv2.waitKey(1)

    def update_map(self):
        self.draw_grid()
        self.add_obstacles()
        self.add_robot()
        self.add_goal()
        self.add_path()
        self.display_map()

    def run(self):
        self.draw_grid()
        self.add_obstacles()
        self.add_robot()
        self.add_goal()
        came_from, cost_so_far = self.a_star_search()
        if self.goal_position in came_from:
            print("Solution found!")
            print("Cost:", cost_so_far[self.goal_position])
            self.get_path(came_from)
            for i in self.path:
                print(i)
            self.add_path()
            self.decode_motions()
            print(f"Set of motions: {self.motions}")
        else:
            print("No path found!")
        self.display_map()

class Action:
    def __init__(self, state, action_time):
        self.state = state
        self.start_time = None
        self.action_time = action_time

    def start(self):
        self.start_time = time.time()

    def is_done(self):
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

        obstacle_positions = [(1, 1), (0, 9), (4, 1), (4, 7), (8, 3), (8, 6)]
        robot_position = (4, 5)
        goal_position = (8, 8)
        self.grid_world = GridWorldG7(
            470, 47, obstacle_positions, robot_position, goal_position
        )
        self.grid_world.run()

        self.moving_speed = 0.4
        self.moving_time = 1.15
        self.turning_time = 4.1
        self.moving_after_turn_time = 1.4
        self.idle_time = 0.4

        self.action_stack = []
        moving_path = self.grid_world.motions.copy()
        while len(moving_path) > 0:
            motion = moving_path.pop(0)
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
        self.current_action = None

    def update_robot_direction(self, action):
        if action == "turn_right":
            self.grid_world.robot_direction -= 90
        elif action == "turn_left":
            self.grid_world.robot_direction += 90

    def update_robot_position(self):
        self.grid_world.update_robot_position()

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

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error("Failed to convert image: " + str(e))
            return

        if self.current_action is None and len(self.action_stack) > 0:
            self.current_action = self.action_stack.pop(0)
            self.current_action.start()
            self.state = self.current_action.state
            self.update_robot_direction(self.state)
            self.update_robot_position()
            print("Moving: ", self.state)

        if self.current_action is not None and self.current_action.is_done():
            if self.state == "move_forward":
                self.update_robot_position()
            if len(self.action_stack) > 0:
                self.current_action = self.action_stack.pop(0)
                self.current_action.start()
                self.state = self.current_action.state
                self.update_robot_direction(self.state)
                if self.state != "idle":
                    print("Moving: ", self.state)
            else:
                self.current_action = None
                self.state = "idle"

        self.update_state()


def main(args=None):
    rclpy.init(args=args)
    turtlebot_controller = TurtlebotController()
    rclpy.spin(turtlebot_controller)
    turtlebot_controller.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
