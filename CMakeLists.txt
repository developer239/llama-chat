cmake_minimum_required(VERSION 3.24)
project(llama_project)

set(CMAKE_CXX_STANDARD 17)

# Add llama.cpp as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# Add your executable with a different name
add_executable(llama_app src/main.cpp)

# Link llama.cpp to your executable
target_link_libraries(llama_app PRIVATE llama)

# Include llama.cpp headers
target_include_directories(llama_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp)
