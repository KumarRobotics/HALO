from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'air_sem_gridmap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('config/*.yaml')),
        (os.path.join('share', package_name), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dexter Ong',
    maintainer_email='dexterong94@gmail.com',
    description='Semantic grid mapping for aerial robots',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sem_gridmap_node = air_sem_gridmap.node.sem_gridmap_node:main',
        ],
    },
)