#!/bin/sh

wget 'http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/i386/cuda-repo-ubuntu1204_5.5-0_i386.deb' --quiet
dpkg -i cuda-repo-ubuntu1204_5.5-0_i386.deb
apt-get -y update
apt-get -y install cuda
rm cuda-repo-ubuntu1204_5.5-0_i386.deb

apt-get -y install flex bison scons build-essential subversion llvm-3.2-dev
apt-get -y install libboost-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev
apt-get -y install ocaml

if [ -d gpuocelot ]; then
  cd gpuocelot
  svn update
else
  svn checkout http://gpuocelot.googlecode.com/svn/trunk/ocelot/ gpuocelot
  cd gpuocelot
fi
python build.py --install
ldconfig

cd ..
cp /vagrant/profile.sh .profile
chown vagrant:vagrant .profile

apt-get -y install gdb valgrind
apt-get -y install python-software-properties
