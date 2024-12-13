# Lane Detection with OpenCV in C++

This repository contains a C++ application for detecting lanes in videos or images using OpenCV. The application processes input video frames to detect lane markers and visualize them.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- OpenCV 4
- HDF5
- VTK (Visualization Toolkit)

## Building the Application

To build the lane detection application, navigate to the directory containing `detector.cpp` and use the following command:

```bash
g++ detector.cpp -o detector `pkg-config --cflags --libs opencv4` -lhdf5 -lvtkCommonCore -lvtkRenderingCore -lvtkInteractionStyle -lvtkFiltersCore

