cmake_minimum_required(VERSION 3.10)

project(MyApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")

include_directories(include)

add_executable(myapp ${SOURCES})

file(GLOB HEADERS "include/*.h" "include/*.hpp")
target_sources(myapp PRIVATE ${HEADERS})