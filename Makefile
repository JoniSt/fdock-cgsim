#PWD = $(shell pwd)
PWD = .

HOST_SRCS := $(wildcard ./*.c) $(wildcard ./*.cpp)
HOST_EXE := $(PWD)/fdock_cpu

HOST_SRCS += graphtoy/graphtoy.cpp

HOST_CFLAGS = -Wall -std=gnu++20 -O3
HOST_LFLAGS = -lm

# ===============================
# Extra compilation and linking
# options
# ===============================

DEBUG = 1
#DEBUG = 0

#PROFILE = 1
PROFILE = 0

#REP = 1
REP = 0

ifeq ($(DEBUG), 1)
	HOST_CFLAGS +=-g
endif

ifeq ($(PROFILE), 1)
	HOST_CFLAGS +=-pg
	HOST_LFLAGS +=-pg
endif

ifeq ($(REP), 1)
	HOST_CFLAGS += -DREPRO
endif

HOST_CFLAGS += $(shell $(CXX) -fcoroutines --version > /dev/null 2>&1 && echo "-fcoroutines")

# ===============================
# Program arguments
# ===============================

# Input molecules
GRID_FILE = $(PWD)/input_data/1hvr_vegl.maps.fld
LIGAND_FILE = $(PWD)/input_data/1hvrl.pdbqt

# Molecular docking configuration
#NEV = 250000 # Default: 2500000
#NRUN = 5 # Default: 100
NEV = 2500
NRUN = 1

# Appending program arguments altogether
ARGS = -ffile $(GRID_FILE) -lfile $(LIGAND_FILE) -nev $(NEV) -nrun $(NRUN) -graphdumps 0

# ===============================
# Rules
# ===============================

main:
	$(CXX) -o $(HOST_EXE) $(HOST_CFLAGS) $(HOST_SRCS) $(HOST_LFLAGS)

run:
	$(HOST_EXE) $(ARGS)

clean:
	rm -rf $(HOST_EXE) initpop.txt seeds.txt

