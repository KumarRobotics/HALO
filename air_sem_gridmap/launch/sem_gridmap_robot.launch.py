import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )    
    
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    mapper_node = Node(
        package='air_sem_gridmap',
        executable='sem_gridmap_node',
        name='sem_gridmap_node',
        output='screen',
        remappings=[]
    )
    
    return LaunchDescription([use_sim_time_arg, mapper_node])
