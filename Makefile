COMPILE = g++
FLAGS = -O3
main: main.cpp instance.o problem.o
	$(COMPILE) $(FLAGS) main.cpp instance.o problem.o

problem.o: problem.cpp problem.h
	$(COMPILE) $(FLAGS) -c problem.cpp
	
instance.o: instance.cpp instance.h
	$(COMPILE) $(FLAGS) -c instance.cpp