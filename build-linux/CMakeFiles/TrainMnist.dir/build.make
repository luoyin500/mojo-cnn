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
include CMakeFiles/TrainMnist.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TrainMnist.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TrainMnist.dir/flags.make

CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.o: CMakeFiles/TrainMnist.dir/flags.make
CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.o: ../examples/train_mnist.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspaces/mojo-cnn/build-linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.o -c /workspaces/mojo-cnn/examples/train_mnist.cpp

CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspaces/mojo-cnn/examples/train_mnist.cpp > CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.i

CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspaces/mojo-cnn/examples/train_mnist.cpp -o CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.s

# Object files for target TrainMnist
TrainMnist_OBJECTS = \
"CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.o"

# External object files for target TrainMnist
TrainMnist_EXTERNAL_OBJECTS =

TrainMnist: CMakeFiles/TrainMnist.dir/examples/train_mnist.cpp.o
TrainMnist: CMakeFiles/TrainMnist.dir/build.make
TrainMnist: CMakeFiles/TrainMnist.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspaces/mojo-cnn/build-linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TrainMnist"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TrainMnist.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TrainMnist.dir/build: TrainMnist

.PHONY : CMakeFiles/TrainMnist.dir/build

CMakeFiles/TrainMnist.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TrainMnist.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TrainMnist.dir/clean

CMakeFiles/TrainMnist.dir/depend:
	cd /workspaces/mojo-cnn/build-linux && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspaces/mojo-cnn /workspaces/mojo-cnn /workspaces/mojo-cnn/build-linux /workspaces/mojo-cnn/build-linux /workspaces/mojo-cnn/build-linux/CMakeFiles/TrainMnist.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TrainMnist.dir/depend

