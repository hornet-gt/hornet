#!/bin/bash
set -x

wget http://releases.llvm.org/4.0.0/llvm-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/cfe-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/libcxx-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/libcxxabi-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/clang-tools-extra-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/polly-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/compiler-rt-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/lld-4.0.0.src.tar.xz
wget http://releases.llvm.org/4.0.0/openmp-4.0.0.src.tar.xz

tar xf llvm-4.0.0.src.tar.xz
tar xf cfe-4.0.0.src.tar.xz
tar xf clang-tools-extra-4.0.0.src.tar.xz
tar xf polly-4.0.0.src.tar.xz
tar xf lld-4.0.0.src.tar.xz
tar xf compiler-rt-4.0.0.src.tar.xz
tar xf libcxx-4.0.0.src.tar.xz
tar xf libcxxabi-4.0.0.src.tar.xz
tar xf openmp-4.0.0.src.tar.xz

rm *.tar.xz
#directory
mkdir llvm-4.0.0.src/build
mkdir llvm-4.0.0.src/tools/clang
mkdir llvm-4.0.0.src/tools/clang/tools/
mkdir llvm-4.0.0.src/tools/clang/tools/extra
mkdir llvm-4.0.0.src/tools/polly
mkdir llvm-4.0.0.src/tools/lld
mkdir llvm-4.0.0.src/projects/compiler-rt
mkdir llvm-4.0.0.src/projects/libcxx
mkdir llvm-4.0.0.src/projects/libcxxabi
mkdir llvm-4.0.0.src/projects/openmp

#copy
mv cfe-4.0.0.src/* llvm-4.0.0.src/tools/clang
mv clang-tools-extra-4.0.0.src/* llvm-4.0.0.src/tools/clang/tools/extra
mv polly-4.0.0.src/* llvm-4.0.0.src/tools/polly
mv lld-4.0.0.src/* llvm-4.0.0.src/tools/lld
mv compiler-rt-4.0.0.src/* llvm-4.0.0.src/projects/compiler-rt
mv libcxx-4.0.0.src/* llvm-4.0.0.src/projects/libcxx
mv libcxxabi-4.0.0.src/* llvm-4.0.0.src/projects/libcxxabi
mv openmp-4.0.0.src/* llvm-4.0.0.src/projects/openmp

rm -r cfe-4.0.0.src clang-tools-extra-4.0.0.src polly-4.0.0.src lld-4.0.0.src compiler-rt-4.0.0.src libcxx-4.0.0.src libcxxabi-4.0.0.src openmp-4.0.0.src
