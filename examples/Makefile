LIBRARY_DIR=../itensor

include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk

trg: trg.cc ../include/tcapi/tcapi.h
	$(CCCOM) $(CCFLAGS) -I../include -o $@ $< $(LIBFLAGS)

.PHONY: clean

clean:
	rm -f trg
