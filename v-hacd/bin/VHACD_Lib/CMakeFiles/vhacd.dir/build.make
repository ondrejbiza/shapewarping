# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1082/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1082/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ondrej/Research/code/v-hacd/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ondrej/Research/code/v-hacd/bin

# Include any dependencies generated for this target.
include VHACD_Lib/CMakeFiles/vhacd.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.make

# Include the progress variables for this target.
include VHACD_Lib/CMakeFiles/vhacd.dir/progress.make

# Include the compile flags for this target's objects.
include VHACD_Lib/CMakeFiles/vhacd.dir/flags.make

VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/FloatMath.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.o -MF CMakeFiles/vhacd.dir/src/FloatMath.cpp.o.d -o CMakeFiles/vhacd.dir/src/FloatMath.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/FloatMath.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/FloatMath.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/FloatMath.cpp > CMakeFiles/vhacd.dir/src/FloatMath.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/FloatMath.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/FloatMath.cpp -o CMakeFiles/vhacd.dir/src/FloatMath.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD-ASYNC.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o -MF CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o.d -o CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD-ASYNC.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD-ASYNC.cpp > CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD-ASYNC.cpp -o CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.o -MF CMakeFiles/vhacd.dir/src/VHACD.cpp.o.d -o CMakeFiles/vhacd.dir/src/VHACD.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/VHACD.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD.cpp > CMakeFiles/vhacd.dir/src/VHACD.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/VHACD.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/VHACD.cpp -o CMakeFiles/vhacd.dir/src/VHACD.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btAlignedAllocator.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o -MF CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o.d -o CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btAlignedAllocator.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btAlignedAllocator.cpp > CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btAlignedAllocator.cpp -o CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btConvexHullComputer.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o -MF CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o.d -o CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btConvexHullComputer.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btConvexHullComputer.cpp > CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/btConvexHullComputer.cpp -o CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdICHull.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o -MF CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o.d -o CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdICHull.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdICHull.cpp > CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdICHull.cpp -o CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdManifoldMesh.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o -MF CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o.d -o CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdManifoldMesh.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdManifoldMesh.cpp > CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdManifoldMesh.cpp -o CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdMesh.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o -MF CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o.d -o CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdMesh.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdMesh.cpp > CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdMesh.cpp -o CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdRaycastMesh.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o -MF CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o.d -o CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdRaycastMesh.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdRaycastMesh.cpp > CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdRaycastMesh.cpp -o CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.s

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/flags.make
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o: /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdVolume.cpp
VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o: VHACD_Lib/CMakeFiles/vhacd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o -MF CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o.d -o CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o -c /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdVolume.cpp

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.i"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdVolume.cpp > CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.i

VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.s"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ondrej/Research/code/v-hacd/src/VHACD_Lib/src/vhacdVolume.cpp -o CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.s

# Object files for target vhacd
vhacd_OBJECTS = \
"CMakeFiles/vhacd.dir/src/FloatMath.cpp.o" \
"CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o" \
"CMakeFiles/vhacd.dir/src/VHACD.cpp.o" \
"CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o" \
"CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o" \
"CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o" \
"CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o" \
"CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o" \
"CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o" \
"CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o"

# External object files for target vhacd
vhacd_EXTERNAL_OBJECTS =

VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/FloatMath.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD-ASYNC.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/VHACD.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/btAlignedAllocator.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/btConvexHullComputer.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdICHull.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdManifoldMesh.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdMesh.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdRaycastMesh.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/src/vhacdVolume.cpp.o
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/build.make
VHACD_Lib/libvhacd.a: VHACD_Lib/CMakeFiles/vhacd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ondrej/Research/code/v-hacd/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX static library libvhacd.a"
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && $(CMAKE_COMMAND) -P CMakeFiles/vhacd.dir/cmake_clean_target.cmake
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vhacd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
VHACD_Lib/CMakeFiles/vhacd.dir/build: VHACD_Lib/libvhacd.a
.PHONY : VHACD_Lib/CMakeFiles/vhacd.dir/build

VHACD_Lib/CMakeFiles/vhacd.dir/clean:
	cd /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib && $(CMAKE_COMMAND) -P CMakeFiles/vhacd.dir/cmake_clean.cmake
.PHONY : VHACD_Lib/CMakeFiles/vhacd.dir/clean

VHACD_Lib/CMakeFiles/vhacd.dir/depend:
	cd /home/ondrej/Research/code/v-hacd/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ondrej/Research/code/v-hacd/src /home/ondrej/Research/code/v-hacd/src/VHACD_Lib /home/ondrej/Research/code/v-hacd/bin /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib /home/ondrej/Research/code/v-hacd/bin/VHACD_Lib/CMakeFiles/vhacd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : VHACD_Lib/CMakeFiles/vhacd.dir/depend

