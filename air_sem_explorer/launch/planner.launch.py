import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    planner_config = PathJoinSubstitution([
        FindPackageShare('air_sem_explorer'),
        'config',
        'planner_ros_sim.yaml'
    ])
    # Create the planner node
    planner_node = Node(
        package='air_sem_explorer',
        executable='planner_node',
        name='planner_node',
        output='screen',
        parameters=[planner_config,
                    {'use_sim_time': use_sim_time}],
        remappings=[]
    )
    
    return LaunchDescription([use_sim_time_arg, planner_node])
