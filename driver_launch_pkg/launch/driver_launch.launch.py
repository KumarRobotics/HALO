import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, OpaqueFunction, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer, PushRosNamespace, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.conditions import LaunchConfigurationEquals, IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, Command, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution as PJoin


def generate_launch_description():
    
    # bag configs
    kr_args = [
        DeclareLaunchArgument('record_bag', default_value='true'),
        DeclareLaunchArgument('bag', default_value='air_sem_bag'),
        DeclareLaunchArgument('bag_prefix', default_value='/bags/air_sem_bag'),
    ]

    # Load UBlox GPS parameters
    ublox_cfg = os.path.join(
        get_package_share_directory('ublox_gps'),
        'config', 'zed_f9p.yaml')
    with open(ublox_cfg, 'r') as f:
        ublox_params = yaml.safe_load(f)['ublox_gps_node']['ros__parameters']
    
    # ZED camera arguments
    default_xacro_path = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'urdf',
        'zed_descr.urdf.xacro'
    )
    zed_args = [
        DeclareLaunchArgument('zed_enable', default_value='true'),
        DeclareLaunchArgument('camera_name', default_value='zed'),
        DeclareLaunchArgument('camera_model', default_value='zed2i'),
        DeclareLaunchArgument('publish_urdf', default_value='true'),
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('publish_map_tf', default_value='true'),
        DeclareLaunchArgument('publish_imu_tf', default_value='true'),
        DeclareLaunchArgument('xacro_path', default_value=TextSubstitution(text=default_xacro_path)),
        DeclareLaunchArgument('custom_baseline', default_value='0.0'),
        DeclareLaunchArgument('enable_gnss', default_value='false'),
        DeclareLaunchArgument('publish_svo_clock', default_value='false'),
    ]

    ld = LaunchDescription(kr_args + zed_args)

    container = ComposableNodeContainer(
        name='driver_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(                
                condition=IfCondition(LaunchConfiguration('record_bag')),
                package='rosbag2_composable_recorder',
                plugin='rosbag2_composable_recorder::ComposableRecorder',
                name="recorder",
                parameters=[{'topics': [
                        "/zed/zed_node/rgb/camera_info ",
                        "/zed/zed_node/rgb/image_rect_color",
                        "/zed/zed_node/depth/camera_info",
                        "/zed/zed_node/depth/depth_registered",
                        "/zed/zed_node/odom",
                        "/zed/zed_node/pose",
                        "/zed/zed_node/pose/filtered",
                        "/ublox_gps_node/fix",
                        "/vectornav/imu",
                        "/vectornav/magnetic",
                        "/planner/exploration_path",
                        "/planner/goal",
                        "/planner/gps_waypoint",
                        "/planner/regions",
                        "/planner/waypoint",
                        "/pose_map",
                        ],
                            'storage_id': 'mcap',
                            'record_all': False,
                            'disable_discovery': False,
                            'serialization_format': 'cdr',
                            'start_recording_immediately': False,
                            "bag_prefix": '/bags/ase_'}],
                remappings=[],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # Ublox GPS node
            ComposableNode(
                package='ublox_gps',
                plugin='ublox_node::UbloxNode',
                name='ublox_gps_node',
                parameters=[ublox_params],
                remappings=[("/aidalm",  "/ublox_raw/aidalm"),
                            ("/timtm2", "/ublox_raw/timtm2"),
                            ("/rtcm", "/ublox_raw/rtcm"),
                            ("/nmea", "/ublox_raw/nmea"),
                            ("/navclock", "/ublox_raw/navclock"),
                            ("/navcov", "/ublox_raw/navcov"),
                            ("/navheading", "/ublox_raw/navheading"),
                            ("/navrelposned", "/ublox_raw/navrelposned"),
                            ("/navstate", "/ublox_raw/navstate"),
                            ("/navsvin", "/ublox_raw/navsvin"),
                            ("/navstatus", "/ublox_raw/navstatus"),
                            ("/aideph", "/ublox_raw/aideph"),
                            ("/diagnostics", "/ublox_raw/diagnostics"),
                            ("/monhw", "/ublox_raw/monhw"),
                            ("/navsin", "/ublox_raw/nmea"),
                            ("/rtcm", "/ublox_raw/rtcm"),
                            ("/rxmrtcm", "/ublox_raw/rxmrtcm")],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package="zed_components",
                plugin="stereolabs::ZedCamera",
                name="zed_node",
                namespace="zed",
                parameters=[
                    [FindPackageShare('zed_wrapper').find('zed_wrapper'), '/config/', LaunchConfiguration('camera_model'), '.yaml'],
                    [FindPackageShare('zed_wrapper').find('zed_wrapper'), '/config/common_stereo.yaml'],
                    # Finally apply launch-specific overrides
                    {
                        'general.camera_name': LaunchConfiguration('camera_name'),
                        'general.camera_model': LaunchConfiguration('camera_model'),
                        'pos_tracking.publish_tf': LaunchConfiguration('publish_tf'),
                        'pos_tracking.publish_map_tf': LaunchConfiguration('publish_map_tf'), 
                        'sensors.publish_imu_tf': LaunchConfiguration('publish_imu_tf', default='true'),
                    }
                ],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            # Vectornav
            ComposableNode(
                package='vectornav',
                plugin='vectornav::Vectornav',
                name='vectornav',
                parameters=[PJoin(
                    [FindPackageShare('vectornav'),
                     'config', 'vectornav_composable.yaml'])],
                remappings=[],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='vectornav',
                plugin='vectornav::VnSensorMsgs',
                name='vn_sensor_msgs',
                parameters=[PJoin(
                    [FindPackageShare('vectornav'),
                     'config', 'vn_sensor_msgs_composable.yaml'])],
                remappings=[],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),

        ],
        output='screen',
    )

    ld.add_action(container)

    # robot state publisher to publish URDF and static transforms for zed
    robot_state_publisher_node = Node(
        condition=IfCondition(LaunchConfiguration('zed_enable')),
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name=[LaunchConfiguration('camera_name'), '_state_publisher'],
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('publish_svo_clock'),
            'robot_description': Command([
                'xacro', ' ', LaunchConfiguration('xacro_path'),
                ' camera_name:=', LaunchConfiguration('camera_name'),
                ' camera_model:=', LaunchConfiguration('camera_model'),
                ' custom_baseline:=', LaunchConfiguration('custom_baseline')
            ])
        }]
    )

    ld.add_action(robot_state_publisher_node)

    return ld

# Add entry point for direct python execution
if __name__ == '__main__':
    # create and run a LaunchService using our description
    ls = LaunchService(argv=sys.argv[1:])
    ls.include_launch_description(generate_launch_description())
    sys.exit(ls.run())
