#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vggt_mapper',
            executable='vggt_mapper_node',
            name='vggt_mapper_node',
            output='screen',
            parameters=[
                {'config_path': 'config/vggt_mapper_robot.yaml'}
            ]
        ),
    ])
