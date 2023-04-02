TARGET_ARCH ?= x86_64
TARGET_OS ?= $(HOST_OS)

INCLUDES += -Icommon/UtilNPP

all: thrust_assignment module3_assignment npp_assignment

thrust_assignment: thrust_assignment.cu
	nvcc $^ -o $@

module3_assignment: module3_assignment.cu
	nvcc $^ -o $@

npp_assignment: npp_assignment.cu
	nvcc $^ -o $@ $(INCLUDES) -lfreeimage -lnppc -lnppial -lnppim -lnppisu -lnppif

clean:
	rm -f thrust_assignment module3_assignment npp_assignment
