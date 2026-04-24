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
    mapper_config = PathJoinSubstitution([
        FindPackageShare('air_sem_explorer'),
        'config',
        'mapper_ros.yaml'
    ])
     
    # Create the mapper node
    mapper_node = Node(
        package='air_sem_explorer',
        executable='mapper_node',
        name='mapper_node',
        output='screen',
        parameters=[
            mapper_config,
        {
            'use_sim_time': use_sim_time
        }],
        remappings=[]
    )
    
    return LaunchDescription([use_sim_time_arg, mapper_node])
