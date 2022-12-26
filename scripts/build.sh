mkdir build ; cd build
cmake ..
cmake --build . -j$(nproc)
