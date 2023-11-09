cmake_minimum_required (VERSION 2.9.0)
project (BarcodeDetectionHough)

find_package(OpenCV 4.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-std=c++11)

find_package(Boost COMPONENTS filesystem system chrono program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(${Boost_DEFINITIONS})

include_directories("./include")
include_directories("./utils-3rdparty")

file(GLOB SOURCES "./src/*.cpp")
file(GLOB SOURCES_3RD_PARTY "./utils-3rdparty/*.cpp")

add_library(Barcode3rdParty SHARED ${SOURCES_3RD_PARTY})
add_library(BarcodeLibrary SHARED ${SOURCES} ${SOURCES_3RD_PARTY})

target_include_directories(BarcodeLibrary PRIVATE "$./include/" "./utils-3rdparty/")
target_link_libraries(BarcodeLibrary ${OpenCV_LIBS} ${Boost_LIBRARIES})

install(TARGETS BarcodeLibrary
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable (barcodedetection main.cpp)

target_link_libraries(barcodedetection BarcodeLibrary Barcode3rdParty ${OpenCV_LIBS} ${Boost_LIBRARIES})





