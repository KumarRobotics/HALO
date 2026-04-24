import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
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
    
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('air_sem_explorer'),
            'config',
            'mapper.rviz'
        ]),
        description='Path to RViz config file'
    )
    
    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config = LaunchConfiguration('rviz_config')

    mapper_config = PathJoinSubstitution([
        FindPackageShare('air_sem_explorer'),
        'config',
        'mapper_ros_sim.yaml'
    ])
    print("Mapper config path:", mapper_config)

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
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    print("RViz node created with config:", rviz_config)
    
    return LaunchDescription([
        use_sim_time_arg,
        rviz_config_arg,
        mapper_node,
        rviz_node
    ])

