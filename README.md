# Autonomous Navigation System: Assignment of Robotics course (Master of Computational Science)

This repository contains the implementation of an autonomous navigation system for a TurtleBot3 Waffle Pi. The system was developed as part of Assignment 4 for a Robotics course and includes features such as A* path planning and Simultaneous Localization and Mapping (SLAM).

## Files:
- task1.py : A* Path Planning on Customized Gazebo World
- world_task1.xml : the Gazebo world file designed for task 1
- task2.py : Visual SLAM Implementation
- world_task2.xml : the Gazebo world file designed for task 2

To run my code, you can simply create new world files (using the xml files) and launch files in Gazebo. Open the world and the corresponding Python file in separate consoles.

## Introduction
In this README, I discuss the progress made on Assignment 4, which aimed to improve the autonomous navigation system developed in the previous assignment. This project incorporated two key features, building upon the vision module developed in Assignment 3: A* path planning and Simultaneous Localization and Mapping (SLAM).

## Features
- **A* Path Planning**: This algorithm was implemented to enable efficient navigation in a customized Gazebo world populated with multiple obstacles. It generates the shortest collision-free path from the robot's current location to a goal position while avoiding any obstacles.
- **Visual SLAM (Simultaneous Localization and Mapping)**: This system allows the robot to create a real-time map of its environment. It subscribes to the TurtleBot3's camera topic to receive images from its surroundings and uses OpenCV to detect obstacles and walls.

## Approach
The assignment was divided into three main tasks:

1. **Customizing the Gazebo World**: I customized the existing world using built-in editors, adding multiple red-colored cube obstacles and creating a flat blue area to serve as the destination for the path-finding problem.
2. **Implementing A* Path Planning**: I created a grid-based map of the environment, with each cell representing a possible state for the autonomous agent. The A* algorithm was initialized with a start state and a goal state, and it maintains a priority queue of states to explore.
3. **Developing a Visual SLAM (VSLAM) System**: My VSLAM implementation is based on a grid world. The robot uses image processing techniques to detect obstacles and update its path.

## Challenges
I encountered two main challenges during this assignment:

- **Mapping the Robot's Movement**: One of the challenges was mapping the movement of the robot in the world environment and the grid world (created for A* path planning and VSLAM). To overcome this challenge, I defined three basic actions for the robot: move forward, turn left, and turn right. Each action had a specific movement and moving time. After each action, I updated the grid world based on the robot's movement.
- **Unstable Behavior of Gazebo**: The instability of Gazebo when running on different hardware/operating systems was a challenge. I defined the robot's action based on movement and moving time. Optimal values for moving parameters were obtained through practical experiments.

## Results and Observations
My implementation of the assignment tasks yielded impressive results. My system successfully analyzed the grid world, applied the A* algorithm, and provided a list of necessary actions for the robot to reach the target destination. While moving, the robot continuously detected new obstacles and updated the grid world. When the current moving path was blocked by new obstacles, my solution reran the A* algorithm with the updated grid world and provided a new path for the robot.

## Conclusion
Through this assignment, I gained valuable insights into robotics, particularly in autonomous navigation systems. I successfully implemented an A* path planning algorithm and a Visual SLAM system to enhance the capabilities of my TurtleBot3 Waffle Pi. Despite some challenges, I was able to overcome them through practical experiments and defining basic actions for the robot. The system yielded impressive results, and I aim to refine it further to handle more complex environments and improve the robot's movement accuracy. The experience gained from this assignment has reinforced my understanding of autonomous navigation systems and sparked my curiosity to explore more advanced concepts in robotics.
