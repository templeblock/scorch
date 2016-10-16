# scorch

A no-frills deep learning library in C built on top of [TH](https://github.com/torch/torch7/tree/master/lib/TH) and [THNN](https://github.com/torch/nn/tree/master/lib/THNN).

**UNDER CONSTRUCTION**
Pre-alpha, don't use.

## Goals

* Only require a C compiler
* Be easily embeddable
* Be easily bindable from high-level languages through FFI
* Be biased towards speed over convenience
* Allow building all major architectures

## Current

* [ ] Create the main neural network API on a feed forward model, limited to float
* [ ] Contribute ports of modules from Lua to C upstream (THNN)
* [ ] Implement optimizers
* [ ] Extend to other neural network architectures
* [ ] Consider wrapping TH tensor API using a fat pointers approach to achieve polymorphism
* [ ] Implement sample models

## Build

Build TH and THNN:
```
sh get_deps.sh
sh build_deps.sh
```

Build scorch:
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release ../ && make
cd ..
```

Run the sample:
```
# macOS
DYLD_LIBRARY_PATH=install/lib build/scorch
# Linux
LD_LIBRARY_PATH=install/lib build/scorch
```

## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2016, Luca Antiga, Orobix Srl (www.orobix.com).

