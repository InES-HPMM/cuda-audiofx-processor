wget https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-linux-aarch64.tar.gz -q --show-progress 
tar -zxvf cmake-3.29.2-linux-aarch64.tar.gz 
cd cmake-3.29.2-linux-aarch64/
sudo cp -rf bin/ doc/ share/ /usr/local/
sudo cp -rf man/* /usr/local/man
sync
cmake --version 
rm -r cmake-3.29.2-linux-aarch64.tar.gz cmake-3.29.2-linux-aarch64/