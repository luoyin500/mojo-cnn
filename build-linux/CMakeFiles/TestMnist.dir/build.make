# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspaces/mojo-cnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspaces/mojo-cnn/build-linux

# Include any dependencies generated for this target.
include CMakeFiles/TestMnist.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TestMnist.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TestMnist.dir/flags.make

CMakeFiles/TestMnist.dir/examples/test.cpp.o: CMakeFiles/TestMnist.dir/flags.make
CMakeFiles/TestMnist.dir/examples/test.cpp.o: ../examples/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspaces/mojo-cnn/build-linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TestMnist.dir/examples/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TestMnist.dir/examples/test.cpp.o -c /workspaces/mojo-cnn/examples/test.cpp

CMakeFiles/TestMnist.dir/examples/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TestMnist.dir/examples/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspaces/mojo-cnn/examples/test.cpp > CMakeFiles/TestMnist.dir/examples/test.cpp.i

CMakeFiles/TestMnist.dir/examples/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TestMnist.dir/examples/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspaces/mojo-cnn/examples/test.cpp -o CMakeFiles/TestMnist.dir/examples/test.cpp.s

# Object files for target TestMnist
TestMnist_OBJECTS = \
"CMakeFiles/TestMnist.dir/examples/test.cpp.o"

# External object files for target TestMnist
TestMnist_EXTERNAL_OBJECTS =

TestMnist: CMakeFiles/TestMnist.dir/examples/test.cpp.o
TestMnist: CMakeFiles/TestMnist.dir/build.make
TestMnist: CMakeFiles/TestMnist.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspaces/mojo-cnn/build-linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TestMnist"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TestMnist.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TestMnist.dir/build: TestMnist

.PHONY : CMakeFiles/TestMnist.dir/build

CMakeFiles/TestMnist.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TestMnist.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TestMnist.dir/clean

CMakeFiles/TestMnist.dir/depend:
	cd /workspaces/mojo-cnn/build-linux && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspaces/mojo-cnn /workspaces/mojo-cnn /workspaces/mojo-cnn/build-linux /workspaces/mojo-cnn/build-linux /workspaces/mojo-cnn/build-linux/CMakeFiles/TestMnist.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TestMnist.dir/depend

