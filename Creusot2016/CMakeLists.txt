cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE Release)
project(LCEBDFTB)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

find_package(OpenCV REQUIRED)

#find_package(MathGL REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FILE(GLOB_RECURSE LibFiles "*.h")
add_custom_target(headers SOURCES ${LibFiles})

add_library(LCEBDFTB SHARED barcode_localization.cpp)
#target_link_libraries(barcodelocalizer ${OpenCV_LIBS} ${MathGL_LIBRARIES} zbar)
target_link_libraries(LCEBDFTB ${OpenCV_LIBS})
file(GLOB SRC_FILES *.cpp)
add_executable(MyApp ${SRC_FILES})
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(MyApp ${OpenCV_LIBS})

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib/${PROJECT_NAME})

# Install library headers
file(GLOB HEADERS *.h)
install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME})
