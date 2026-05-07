cmake_minimum_required(VERSION 3.10)
project(detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(detector src/main.cpp src/detector.cpp src/ellipse_filter.cpp src/contour_segmenter.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(detector ${OpenCV_LIBS})

install(TARGETS detector
    RUNTIME DESTINATION bin
)