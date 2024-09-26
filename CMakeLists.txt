cmake_minimum_required(VERSION 3.24)
project(LlamaCPP VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIB_NAME LlamaCPP)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "The llama.cpp submodule is missing. Please run 'git submodule update --init --recursive'")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp)

set(SOURCES
        src/llama-chat.cpp
        src/llama-chat.h
)

add_library(${LIB_NAME} STATIC ${SOURCES})

target_link_libraries(${LIB_NAME} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${LIB_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp
)

install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES src/llama-chat.h DESTINATION include)
