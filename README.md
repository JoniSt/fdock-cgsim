# README

`fdock_cpu` is a molecular docking code predecessor of [`AutoDock-GPU`](https://github.com/ccsb-scripps/AutoDock-GPU).

`fdock_cpu` is a C++ program originally implemented in a single-threaded fashion.

# Instructions

- Compilation: `make`
- Execution: `make run`

```zsh
➜  fdock_cpu git:(main) make
g++ -o ./fdock_cpu -Wall ./fdock.c ./getparameters.c ./miscellaneous.c ./performdocking.c ./processgrid.c ./processligand.c ./processresult.c ./searchoptimum.c -lm
./searchoptimum.c: In function ‘void genetic_generational(double (*)[40], const Liganddata*, const Gridinfo*, const double*, Dockpars*, int, int)’:
./searchoptimum.c:674:24: warning: unknown escape sequence: '\B'
  674 |                 printf("Maximal delta movement during mutation: +/-%lfA, maximal delta angle during mutation: +/-%lf\B0\n", abs_max_dmov, abs_max_dang);
      |                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
./searchoptimum.c:675:24: warning: unknown escape sequence: '\B'
  675 |                 printf("Rho lower bound: %lf, maximal delta movement during ls: +/-%lfA, maximal delta angle during ls: +/-%lf\B0\n", rho_lower_bound, base_dmov_mul_sqrt3, base_dang_mul_sqrt3);
      |                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
➜  fdock_cpu git:(main) make run
./fdock_cpu -ffile ./input_data/1hvr_vegl.maps.fld -lfile ./input_data/1hvrl.pdbqt -nev 250000  -nrun 5 
Starting dockings...
Run 1 started...     finished, CPU run time: 14.586s
Run 2 started...     finished, CPU run time: 14.693s
Run 3 started...     finished, CPU run time: 14.569s
Run 4 started...     finished, CPU run time: 14.631s
Run 5 started...     finished, CPU run time: 14.730s


Average run time of one run: 14.642 sec
Program run time: 73.675 sec
```

