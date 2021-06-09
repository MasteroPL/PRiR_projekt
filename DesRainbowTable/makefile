main: des.o indexer.o rainbow_table.o 
	nvcc -c main.cu 

des.o: des.cu des.cuh
	nvcc -o des.o -c des.cu 

indexer.o: indexer.cu indexer.cuh
	nvcc -o indexer.o -c indexer.cu

rainbow_table.o: rainbow_table.cu rainbow_table.h
	nvcc -o rainbow_table.o -c rainbow_table.cu

# jeśli nvcc nie widzi kompilatora c albo cpp to nalezy dodać -ccbin path do kompilatora. np. u mnie -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64"
