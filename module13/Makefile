OBJS ?= common/FreeImage/lib/linux/x86_64/libfreeimage.a

INCLUDES += -I common/FreeImage/include
INCLUDES += -I module13/common/UtilNPP

all: simple_asynch ImageFilter2D test_image assignment

simple_asynch: simple_asynch.cpp
	g++ $^ -o $@ -l OpenCL $(INCLUDES) $(OBJS)

ImageFilter2D: ImageFilter2D.cpp
	g++ $^ -o $@ -l OpenCL $(INCLUDES) $(OBJS)

test_image: test_image.cpp
	g++ $^ -o $@ -l OpenCL $(INCLUDES) $(OBJS)

assignment: assignment.cpp
	g++ $^ -o $@ -l OpenCL $(INCLUDES) $(OBJS)

clean:
	rm -f simple_asynch ImageFilter2D test_image assignment