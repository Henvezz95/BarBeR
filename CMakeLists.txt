project(Barcode_Localization_Benchmark)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(Gallo2011-Soros2013-Yun2017)
add_subdirectory(Zamberletti2013)
#add_subdirectory(Tekin2012)
#add_subdirectory(Creusot2016)