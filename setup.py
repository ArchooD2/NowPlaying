from setuptools import setup, find_packages

setup(
    name="nowplay",
    version="0.1.5",
    description="CLI tool for playing audio and visualizing waveform with metadata.",
    author="PJ H.",
    author_email="archood2@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sounddevice",
        "matplotlib",
        "mutagen",
        "blessed",
    ],
    entry_points={
        "console_scripts": [
            "nowplay-peak=nowplay.play_peak:main",
            "nowplay-pitch=nowplay.play_pitch:main",
        ]
    },
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)