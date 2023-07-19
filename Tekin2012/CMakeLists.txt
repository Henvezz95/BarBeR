project(blade)

cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wfatal-errors -pedantic -std=gnu99 -Wl,--no-undefined -fPIC -Wall -Wextra -Wconversion -Werror=return-type -Werror=implicit-function-declaration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -pedantic -std=c++11 -Wl,--no-undefined -fPIC -Wall -Wextra -Wconversion -Werror=return-type -Werror=implicit-function-declaration")


file(GLOB BLaDE "BLaDE/src/*.cpp")

include_directories("include")

add_definitions(-DDEBUG)
add_library(blade SHARED ${BLaDE})
target_link_libraries(blade)
target_link_libraries(blade ${OpenCV_LIBS})
set_target_properties(blade PROPERTIES
    OUTPUT_NAME "blade"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX "lib" )

    file(GLOB_RECURSE COMP_METHOD
        "./BLaDE/src/*.cpp"
        "./BLaDE/src/*.hpp"
        "./BLaDE/src/*.h"
        ".include/ski/*.h"
        ".include/ski/*/*.h"
    )

add_executable( blade_executable "BLaDE/main.cpp" ${COMP_METHOD})
include_directories( blade_executable "BLaDE/src/" "include/" "include/ski/" "include/ski/BLaDE/" "include/ski/sound/")

target_include_directories(blade_executable PRIVATE "BLaDE/src/" "include/" "include/ski/" "include/ski/BLaDE/" "include/ski/sound/")
target_link_libraries( blade_executable ${OpenCV_LIBS})



