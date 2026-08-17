# 1. Provide the absolute or relative path to where you compiled ITensor
LIBRARY_DIR=itensor

# 2. All your TCAPI headers
HEADERS=$(wildcard include/tcapi/*.h)

# --------- Boilerplate below this line -----------
include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk

TENSOR_HEADERS=$(LIBRARY_DIR)/itensor/core.h

# Standalone TCAPI tests (tests/test_*.cc) -------------
TESTSRC=$(wildcard tests/*.cc)
TESTOBJ=$(patsubst tests/%.cc,tests/%.o, $(TESTSRC))
TESTBIN=$(patsubst tests/%.cc,tests/%, $(TESTSRC))
EXAMPLESRC=$(wildcard examples/*.cc)
EXAMPLEOBJ=$(patsubst examples/%.cc,examples/%.o, $(EXAMPLESRC))
EXAMPLEBIN=$(patsubst examples/%.cc,examples/%, $(EXAMPLESRC))

# Rules ------------------
tests/%.o: tests/%.cc tests/tc_test_util.h $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -Iinclude -o $@ $<

tests/%: tests/%.o
	$(CCCOM) $(CCFLAGS) $< -o $@ $(LIBFLAGS)

examples/%.o: examples/%.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -Iinclude -o $@ $<

examples/%: examples/%.o
	$(CCCOM) $(CCFLAGS) $< -o $@ $(LIBFLAGS)

# Targets -----------------
.PHONY: build examples test-all clean

build: $(TESTBIN) $(EXAMPLEBIN)

examples: $(EXAMPLEBIN)

test-all: build
	@echo "== standalone unit test suite =="
	@for t in $(TESTBIN); do echo "--- ./$$t ---"; ./$$t; done

clean:
	rm -fr .debug_objs *.o test_tcapi test_tcapi-g
	rm -fr tests/*.o tests/test_*_itensor
	rm -fr examples/*.o examples/trg
