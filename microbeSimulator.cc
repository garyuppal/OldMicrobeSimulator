#include "common/chemicals/field.h"
// #include "common/simulator/simulator.h"
// #include "common/simulator/simulator_BDF2.h"
#include "common/simulator/FDM_simulator.h"
// #include "common/simulator/microbe_BDF2.h"
// #include "common/simulator/bootstrapBDF2.h"
// #include "common/simulator/dgsimulator.h"
// #include "common/simulator/latest_simulator.h"
// #include "common/simulator/stokes_solver.h"

#include "common/simulator/custom_stokes.h"
#include "common/modular_simulator/modular_simulator.h"
#include "common/utility/argparser.h"
#include "common/discrete_field/numerical_velocity.h"
#include "common/discrete_field/FDM_chemical.h"
#include "common/modular_simulator/super_simulator.h"

#include <list>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <chrono> 
#include <functional>

	
int main(int argc, char** argv)
{
  try
    {
  	using namespace dealii;
  	using namespace MicrobeSimulator;
        using namespace std::chrono; 

        auto start = high_resolution_clock::now(); 

  		ArgParser parameters(argc,argv);
  		parameters.print(std::cout);
  		parameters.outputParameters();

  		const unsigned int seed = time(0) + parameters.getJobID();
  		srand(seed);

  		const unsigned int dim = 2;
      const unsigned int numchem = 2;



      // ModularSimulator::Simulator<dim, numchem> simulator;
      // simulator.run(parameters); 

      MetaSim::SuperSimulator<dim, numchem> ss( parameters.getJobID() );

      ss.run_filters(parameters);


  		auto stop = high_resolution_clock::now(); 
  		auto duration = duration_cast<seconds>(stop - start); 

  		std::cout << "\n\n\nTotal program run time: "
  			<< duration.count() << " seconds\n\n" << std::endl; 

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl << std::endl;
      return 1;
    } // try

  return 0;

} // main()


