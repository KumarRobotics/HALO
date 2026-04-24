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
        parameters=[{'use_sim_time': use_sim_time, 
                     'rgb_topic': '/quadrotor/color/image',
                     'depth_topic': '/quadrotor/depth/image',
                     'odom_topic': '/quadrotor/pose',
                     'intrinsic_topic': '/quadrotor/color/info'}],
        remappings=[]
    )
    
    from launch.actions import ExecuteProcess
    set_prompt_client = ExecuteProcess(
        cmd=['python3', os.path.join(os.path.dirname(__file__), 'set_prompt_client.py')],
        output='screen',
        shell=False
    )

    return LaunchDescription([
        use_sim_time_arg,
        mapper_node,
        set_prompt_client
    ])
