# Install script for directory: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/libvhacd.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/FloatMath.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/btAlignedAllocator.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/btAlignedObjectArray.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/btConvexHullComputer.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/btMinMax.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/btScalar.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/btVector3.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdCircularList.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdICHull.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdManifoldMesh.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdMesh.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdMutex.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdRaycastMesh.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdSArray.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdTimer.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdVHACD.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdVector.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdVolume.h"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/public/VHACD.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdCircularList.inl"
    "/home/ondrej/Research/code/v-hacd/src/VHACD_Lib/inc/vhacdVector.inl"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd/vhacd-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd/vhacd-targets.cmake"
         "/home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/CMakeFiles/Export/lib/cmake/vhacd/vhacd-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd/vhacd-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd/vhacd-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd" TYPE FILE FILES "/home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/CMakeFiles/Export/lib/cmake/vhacd/vhacd-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd" TYPE FILE FILES "/home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/CMakeFiles/Export/lib/cmake/vhacd/vhacd-targets-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/vhacd" TYPE FILE FILES
    "/home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/vhacd/vhacd-config.cmake"
    "/home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/vhacd/vhacd-config-version.cmake"
    )
endif()

