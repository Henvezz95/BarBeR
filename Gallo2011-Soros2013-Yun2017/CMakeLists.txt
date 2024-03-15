cmake_minimum_required(VERSION 2.8)

project( BarcodeLocalization )

find_package(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui ml)
include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB_RECURSE COMP_METHOD
    "./gallo/*.cpp"
    "./gallo/*.h"
    "./soros/*.cpp"
    "./soros/*.h"
    "./yun/*.cpp"
    "./yun/*.h"
)

add_executable( iyBarcode main.cpp ${COMP_METHOD})
target_link_libraries( iyBarcode ${OpenCV_LIBS})

add_library(BarcodeLocalization SHARED "./gallo/gallo.cpp"
                                        "./soros/soros.cpp"
                                        "./yun/yun.cpp"
                                        "./wrapper.cpp")
target_include_directories(BarcodeLocalization PRIVATE ${OpenCV_INCLUDE_DIRS} "./gallo/" "./soros/" "./yun/" "./")
target_link_libraries(BarcodeLocalization ${OpenCV_LIBS})
install(TARGETS BarcodeLocalization
LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})







