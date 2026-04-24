import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def get_workspace_path():
    # This works for both installed and source workspaces
    prefix_path = os.getenv('COLCON_PREFIX_PATH', '').split(':')[0]
    if prefix_path:
        return os.path.dirname(prefix_path)  # Goes up from install/<pkg>
    return None

def generate_launch_description():
    ws_path = get_workspace_path()
    if not ws_path:
        raise RuntimeError("Could not determine workspace path!")

    sim_path = os.path.join(ws_path, 'src', 'aerial_sim_binaries', 'aerial_sim.x86_64')
    print(f"Using simulation binary at: {sim_path}")
    config_path = os.path.join(ws_path, 'src', 'HALO', 'aerial_sim_ros2', 'unity_interface', 'config', 'unity_sim_config.yaml')

    return LaunchDescription([
        ExecuteProcess(
            cmd=[sim_path, '-config', config_path],
            output='screen'
        )
    ])


