BASE = tc
BISON = bison
CXX = g++
FLEX = flex

all: $(BASE)

%.cc %.hh: %.yy
	$(BISON) $(BISONFLAGS) -o $*.cc $<

%.cc: %.ll
	$(FLEX) $(FLEXFLAGS) -o$@ $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o$@ $<

$(BASE): $(BASE).o driver.o parser.o scanner.o
	$(CXX) -o $@ $^

$(BASE).o: parser.hh
parser.o: parser.hh
scanner.o: parser.hh

run: $(BASE)
	./$<

CLEANFILES =  \
	$(BASE) *.o \
	parser.hh parser.cc scanner.cc

clean:
	rm -f $(CLEANFILES)
