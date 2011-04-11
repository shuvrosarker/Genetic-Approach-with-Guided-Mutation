network: main.o f2n2.o marchand.o dnc.o
	g++ -o network main.o f2n2.o dnc.o marchand.o
main.o: main.cpp dnc.h marchand.h
	g++ -c dnc.h marchand.h main.cpp 
dnc.o: dnc.cpp dnc.h  f2n2.h
	g++ -c dnc.h dnc.cpp 
marchand.o: marchand.cpp marchand.h f2n2.h
	g++ -c marchand.h marchand.cpp 
f2n2.o: f2n2.cpp f2n2.h
	g++ -c f2n2.h f2n2.cpp
clean:
	rm network f2n2.o main.o dnc.o marchand.o