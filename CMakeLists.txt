cmake_minimum_required(VERSION 3.10)

project(MyApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")

include_directories(include)



include(FetchContent)
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git)
FetchContent_MakeAvailable(cpr)

add_executable(myapp ${SOURCES})

file(GLOB HEADERS "include/*.h" "include/*.hpp")
target_sources(myapp PRIVATE ${HEADERS})
target_link_libraries(myapp PRIVATE cpr::cpr)
