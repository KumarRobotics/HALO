from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vggt_mapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dexter Ong',
    maintainer_email='dexterong94@gmail.com',
    description='ROS2 node for VGGT-SLAM that subscribes to camera images and performs SLAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vggt_mapper_node = vggt_mapper.vggt_mapper_node:main'
        ],
    },
)
