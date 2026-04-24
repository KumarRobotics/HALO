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
    tf_handler_node_config = PathJoinSubstitution([
        FindPackageShare('air_sem_explorer'),
        'config',
        'tf_handler_ros_sim.yaml'
    ])
    print(f"TF Handler config path: {tf_handler_node_config}")
    # Create the tf handler node
    tf_handler_node = Node(
        package='air_sem_explorer',
        executable='tf_handler_node',
        name='tf_handler_node',
        output='screen',
        parameters=[tf_handler_node_config,
                    {'use_sim_time': use_sim_time}],
        remappings=[]
    )
    
    return LaunchDescription([use_sim_time_arg, tf_handler_node])
