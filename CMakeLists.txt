cmake_minimum_required(VERSION 3.24)
project(llama_project)

set(CMAKE_CXX_STANDARD 17)

# Add llama.cpp as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# Set the target name
set(TARGET llama_app)

# Add your executable
add_executable(${TARGET} src/main.cpp)

# Install the target
install(TARGETS ${TARGET} RUNTIME)

# Link libraries
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

# Include directories
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp)

# Ensure C++17 is used
target_compile_features(${TARGET} PRIVATE cxx_std_17)
