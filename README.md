# MicrobeSimulator
General hybrid agent based-continuous field simulator for microbes interacting via secreted compounds. With finite element (using deal.II) and finite difference implementations for continuous fields.

## Installation and usage
The program requires the prior installation of the open source deal.ii library available [here](https://www.dealii.org/). A makefile can then be generated using CMake. Use the command `cmake -DDEAL_II_DIR=path/to/dealii/ .` to generate the makefile. Program is set up with a parameter text file. Information on parameters and configurations is given in the configurations directory.

### To Do
- [ ] remove deal.ii dependencies from finite difference implementation (write own Point<> class) 
- [ ] test out doxygen commenting
- [ ] move cell-iterator pair array to a map of sparse vectors
- [ ] figure out better meshing methods, especially for 3D
- [ ] create interface class for chemicals and combine different implementations through inheritance
- [ ] reorganize file structures
- [ ] vectorize bacteria dynamics
- [ ] parallelize chemicals and bacteria interface

