# scorch

A minimalistic deep learning library in C.

**UNDER CONSTRUCTION**

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

