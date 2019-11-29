
PWD = $(shell pwd)
JAVA_HOME?=/usr/lib/jvm/java-8-openjdk-amd64/
JAVA_ROOT=$(PWD)/../src/main/java/
CPP_ROOT=$(PWD)/../src/main/cpp/
JDK_INCLUDE_ROOT=${JAVA_HOME}/include/
CPPFLAGS := -O1 -fPIC -std=c++14 -g -I$(PWD)/../dependencies/include -I$(JDK_INCLUDE_ROOT) -I$(JDK_INCLUDE_ROOT)/linux -L$(PWD)/../dependencies/lib/
OUTDIR=$(PWD)/target/classes
HEADER=$(CPP_ROOT)/com_intel_analytics_bigdl_ccl_CCLAdapter.h
OBJECTS_DIR = target/cpp


$(OUTDIR)/libccl_java.so: $(OBJECTS_DIR)/ccl.o
		mkdir -p $(OUTDIR)
		$(CXX) -shared -o $@ $(CPPFLAGS) -Wl,--start-group $^ -lccl -Wl,--end-group

$(OBJECTS_DIR)/%.o: $(CPP_ROOT)/%.cpp $(HEADER)
		mkdir -p $(OBJECTS_DIR)
		$(CXX) $(CPPFLAGS) -c $< -o $@

$(HEADER): $(JAVA_ROOT)/com/intel/analytics/bigdl/ccl/CCLAdapter.java
	javac -h $(CPP_ROOT) $^

clean:
	rm -rf $(OBJECTS_DIR) $(OUTDIR)
