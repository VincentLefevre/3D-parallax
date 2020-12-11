cmake_minimum_required(VERSION 2.8.12)
project(cork)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_library(cork STATIC 
  src/file_formats/files.cpp
  src/file_formats/ifs.cpp
  src/file_formats/off.cpp
  src/isct/empty3d.cpp
  src/isct/quantization.cpp
  src/isct/triangle.c
  src/util/log.cpp
  src/util/timer.cpp
  src/cork.cpp)
add_executable(cork-bin src/main.cpp)
set(INCLUDEDIRS
  src/mesh
  src/accel/
  src/file_formats/
  src/isct/
  src/math/
  src/mesh/
  src/rawmesh/
  src/util/
)

if(UNIX)
    target_compile_options(cork PRIVATE -fPIC -fvisibility=hidden)
endif(UNIX)

target_include_directories(cork PRIVATE ${INCLUDEDIRS})
target_include_directories(cork-bin PRIVATE ${INCLUDEDIRS})
SET_TARGET_PROPERTIES(cork-bin PROPERTIES OUTPUT_NAME cork)
IF (NOT GMP_FOUND)
    find_package(GMP REQUIRED)
ENDIF (NOT GMP_FOUND)
include_directories(${GMP_INCLUDE_DIRS})
target_link_libraries(cork-bin PRIVATE cork ${GMP_LIBRARIES})
target_compile_definitions(cork PRIVATE 
  -DNO_TIMER -DREDUCED -DTRILIBRARY -DANSI_DECLARATORS )

INSTALL(TARGETS cork
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
INSTALL(FILES src/cork.h DESTINATION include)
