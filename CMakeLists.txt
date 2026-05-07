cmake_minimum_required(VERSION 3.10)
project(detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(detector
    main.cpp
    detector.cpp
    contour_segmenter.cpp
    ellipse_filter.cpp
)

target_link_libraries(detector ${OpenCV_LIBS})

install(TARGETS detector
    RUNTIME DESTINATION bin
)