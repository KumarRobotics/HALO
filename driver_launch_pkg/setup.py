from setuptools import setup
import os
from glob import glob

package_name = 'driver_launch_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dexter Ong',
    maintainer_email='dexterong94@gmail.com',
    description='Launch drivers for Ublox GPS and ZED wrapper',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
