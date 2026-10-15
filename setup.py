from setuptools import find_packages, setup

package_name = 'human_reid_ros'

setup(
    name=package_name,
    version='0.1.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],

    install_requires=['setuptools'],

    zip_safe=True,

    maintainer='roberto',
    maintainer_email='r.donate.2023@alumnos.urjc.es',

    description=(
        'Re-identificacion de personas con Qwen3-VL-8B en ROS 2 '
        '(servidor de modelo independiente + cliente ligero).'
    ),

    license='TODO: License declaration',

    extras_require={
        'test': [
            'pytest',
        ],
    },

    entry_points={
        'console_scripts': [
            'qwen3vl_reid_node = human_reid_ros.qwen3vl_reid_node:main',
            'image_publisher = human_reid_ros.image_publisher:main',
            'qwen3vl_server = human_reid_ros.qwen3vl_server:main',
            'qwen3vl_download = human_reid_ros.qwen3vl_download:main',
        ],
    },
)
