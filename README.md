# FDock (CPU)

`fdock_cpu` is a molecular docking code predecessor of [`AutoDock-GPU`](https://github.com/ccsb-scripps/AutoDock-GPU).

## Build (CMake, C++23)

This project now uses CMake. Build out-of-source with Clang (16 or later):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

The resulting binary is `build/fdock_cpu`.

## Run

The former `make run` target is replaced by a simple wrapper script. It assumes the binary exists at `build/fdock_cpu`.

```
./run.sh
```

Defaults inside `run.sh`:

- Grid file: `input_data/1hvr_vegl.maps.fld`
- Ligand file: `input_data/1hvrl.pdbqt`
- NEV: `2500`
- NRUN: `1`
- Graph dumps: `0`

## Clean

To remove artifacts produced by runs:

```
rm -f fdock_cpu initpop.txt seeds.txt docking.dlg docking.xml
```

To clean the CMake build directory:

```
rm -rf build/
```
