/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */
#include "src/chemicals/field.h"

#include "src/simulator/FDM_simulator.h"


#include "src/simulator/custom_stokes.h"

#include "src/modular_simulator/modular_simulator.h"

#include "src/utility/argparser.h"

#include "src/discrete_field/numerical_velocity.h"
#include "src/discrete_field/FDM_chemical.h"

#include "src/modular_simulator/super_simulator.h"

#include <list>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <chrono> 
#include <functional>

/** @todo
* move cell iterator array to a map of ``sparse vectors'' computed for each source location
* create a chemical handler -- generalize implementation and number of chemicals
* figure out doxygen and github upload
* vectorize bacteria updates -- reproduction and movement...
* parallelize chemicals solving... (not sure if this will speed things up since bacteria
* need continuous access to chemicals...)
*
* also look up, static, inline, and smart pointers ...
*/

//@todo
// try perhaps wider mixer -- scale
// implement sphere refinement
// do mass check on mixer
// implement splitter

// run non-muation runs for mixer and splitter,
// look at mutation take over time vs populatoin,
// use these results to design futher experiments


      // TO DO:
      // update advection -- tile and dim indep file read for discrete field
      // implement vortex and cylindrical pipe
      // set chemicals to be variable 
	  // ... make bacteria templated and input a custom fitness function
		// be able to switch between implementations FDM, FE, DG (try polymorphism -- smart pointers)

	// FDM reflecting boundaries do not seem to be working properly...
// check at_bc function !!!
	
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


