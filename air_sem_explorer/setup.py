from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'air_sem_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('models/*.pt')),
        (os.path.join('share', package_name, "config"), glob('config/*.yaml')),
        (os.path.join('share', package_name, "config"), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuezhan',
    maintainer_email='yuezhan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Update the path to your nodes
            'mapper_node = air_sem_explorer.node.mapper_node:main',
            'planner_node = air_sem_explorer.node.planner_node:main',
            'sim_tracker_node = air_sem_explorer.node.sim_tracker_node:main',
            'tf_handler_node = air_sem_explorer.node.tf_handler_node:main',
        ],
    },
)