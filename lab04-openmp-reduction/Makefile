BIN_AUTO=binauto
BIN_MANUAL=binmanual
BIN_SEQ=binseq
all:
	g++ -fopenmp -lm auto.cpp -o ${BIN_AUTO}
	g++ -fopenmp -lm manual.cpp -o ${BIN_MANUAL}
	g++ -fopenmp -lm seq.cpp -o ${BIN_SEQ}

auto:
	g++ -fopenmp -lm auto.cpp -o ${BIN_AUTO}
manual:
	g++ -fopenmp -lm manual.cpp -o ${BIN_MANUAL}
seq:
	g++ -fopenmp -lm seq.cpp -o ${BIN_SEQ}
