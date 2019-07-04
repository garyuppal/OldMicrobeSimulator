#ifndef MICROBE_SIMULATOR_SUPER_SIMULATOR
#define MICROBE_SIMULATOR_SUPER_SIMULATOR

#include "./modular_simulator.h" // should include all necessary headers

namespace MicrobeSimulator{ namespace MetaSim{

/** meta parameters:
* mixer radius, 
* number channels
*/ // keep width fixed ...

// also want initial cheater and bacteria initialized within left_length region ...


/** TO DO
*  write filter and mixer initialization functions for geomerty XX
* write filter and mixer run functions for Simulator 	XX
*  -- set initial bacteria location within left length  XX

** add deterministic mutation to bacteria ...

* write reassignment functions in super simulator
* send test run
*/

// struct filter_parameters
// {	
// 	std::vector<unsigned int> number_channels
// };


template<int dim, int numchem>
class SuperSimulator{
public:
	SuperSimulator(unsigned int job_id);

	void run(const ArgParser& parameters);
	void run_filters(const ArgParser& parameters); //, GeoTypes::Filter filter);
	void run_mixers(const ArgParser& parameters); //, GeoTypes::Mixer mixer);
	// void run_filter(const ArgParser& parameters); // need to modify with runs ...
	// void run_mixer();
private:
	const unsigned int run_number;

	// std::vector<unsigned int> number_walls; // number channels - 1

	// void reassign_output_directory(); // each job_id should have a different directory
	// void reassign_geometry(); 
};


/** IMPLEMENTATION
*/
// -------------------------------------------------------------------------------
template<int dim, int numchem>
SuperSimulator<dim, numchem>::SuperSimulator(unsigned int job_id) // pick parameteres based on job id
	:
	run_number(job_id)
{}

	// for each run, want to change:
	// 1. output directory
	// 2. geometry --- this one is a little more difficult since the initial geo file is separate ...
	// 		instead, initialize geometry from parameters instead of file
	// 3. initial number of bacteria ...

template<int dim, int numchem>
void
SuperSimulator<dim, numchem>::run(const ArgParser& parameters)
{

	ModularSimulator::Simulator<dim, numchem> simulator;

	simulator.run(parameters);
}


template<int dim, int numchem>
void
SuperSimulator<dim, numchem>::run_filters(const ArgParser& parameters)
{

	ModularSimulator::Simulator<dim, numchem> simulator;
	GeoTypes::Filter filter;

	const double height = 3.; // fixed for all mixers
	const unsigned int max_num_walls = 5; // channels minus 1

	const unsigned int number_walls 
		= ( (run_number-1) % max_num_walls) + 1; 

	filter.number_channels = number_walls + 1;
	filter.wall_thickness = 0.1;

	// get channel thickness based on number of channels
	filter.channel_thickness = (height - filter.wall_thickness*((double)number_walls) ) 
		/ filter.number_channels;

	filter.left_length = 5.;
	filter.right_length = 3.;

	filter.printInfo(std::cout); 

	// for(unsigned int i = 1; i <= 15; ++i)
	// {
	// 	unsigned int number_channels = ( (i-1) % 5) + 1;

	// 	std::cout << "job_id: " << i << std::endl
	// 		<< "number_channels: " << number_channels << std::endl;
	// }

	// simulator.run(parameters, filter);
}


template<int dim, int numchem>
void
SuperSimulator<dim, numchem>::run_mixers(const ArgParser& parameters)
{

	ModularSimulator::Simulator<dim, numchem> simulator;

	GeoTypes::Mixer mixer;

	simulator.run(parameters, mixer);
}


}}
#endif