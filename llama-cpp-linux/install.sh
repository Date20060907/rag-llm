git clone --depth=1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -Bbuild -DLLAMA_CURL=OFF
cmake --build build
cd build/bin