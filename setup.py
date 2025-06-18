from setuptools import setup, find_packages

setup(
    name="nowplaying",
    version="0.1.0",
    description="CLI tool for playing audio and visualizing waveform with metadata.",
    author="PJ H.",
    author_email="archood2@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sounddevice",
        "matplotlib",
        "mutagen"
    ],
    entry_points={
        "console_scripts": [
            "nowplaying-peak=nowplaying.play-peak:main",
            "nowplaying-pitch=nowplaying.play-pitch:main"
        ]
    },
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)