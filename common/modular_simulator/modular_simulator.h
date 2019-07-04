#ifndef MICROBE_SIMULATOR_MODULAR_SIMULATOR_H
#define MICROBE_SIMULATOR_MODULAR_SIMULATOR_H

#include <deal.II/base/utilities.h>

#include <deal.II/grid/tria.h>

#include "../utility/argparser.h"
#include "../geometry/geometry.h"

#include "../simulator/exact_solutions.h"
#include "../simulator/cell_iterator_map.h"

#include "./stokes_handler.h"
#include "./chemical_fe_base.h"
#include "./fe_chemical.h"
#include "./fitness_functions.h"

namespace MicrobeSimulator{ namespace ModularSimulator{
	using namespace dealii;


template<int dim, int numchem>
class Simulator{
public:
	Simulator(const unsigned int number_refinements = 3,
			const unsigned int stokes_degree = 1);

	void run(const ArgParser& parameters);
	
	void run(const ArgParser& parameters, GeoTypes::Filter filter);
	void run(const ArgParser& parameters, GeoTypes::Mixer mixer);

	void run_test();

private:
	Triangulation<dim>			triangulation;

	Geometry<dim> 				geometry;

	StokesHandler<dim>			stokes_handler;
	Chemical_FE_Base<dim>		chemical_fe_base; 
	// want this to ideally be implementation independent...
	// can perhaps create a chemical handler class that has either an
	// fe_base, dg_base, or fdm_base ...

	std::array<FE_Chemical<dim>, numchem> 	chemicals;

	// AdvectionField<dim> 		advection_field; // attach stokes handler -- only use if necessary...
	Bacteria<dim, numchem>		bacteria;
	FitnessFunctions::OR_Fitness<dim, numchem>	fitness_function;

	PointCellMap<dim>			point_cell_map; 
		// change this to store sparse vectors...
		// point cell map is used to secrete to chemicals ...

	// SYSTEM CONSTANTS:
	std::string 				output_directory;
	double 						run_time;
	double						time;
	double						time_step;

	unsigned int 				save_period;
	unsigned int 				save_step_number;
	unsigned int 				time_step_number;

	// METHODS:
	void setup_system_constants(const ArgParser& parameters);
	
	void setup_geometry(const ArgParser& parameters); // *** want to override parameters for geometry initialization ...
	void setup_geometry(GeoTypes::Filter filter);
	void setup_geometry(GeoTypes::Mixer mixer);
	void output_geometry() const;

	void setup_grid(unsigned int initial_refinement); // from geometry 
	void solve_stokes();
	void setup_chemicals(const ArgParser& parameters);	// stokes should be solved before this
	void setup_secretion_map(double resolution); // chemical base needs to be setup before this

	// void setup_bacteria(const ArgParser& parameters);
	void setup_bacteria(const ArgParser& parameters, double left_length = -1.);	

	void setup_fitness(const ArgParser parameters); 
	double get_CFL_time_step() const;

	void setup_system(const ArgParser& parameters); // include solving stokes
	void setup_system(const ArgParser& parameters, GeoTypes::Filter filter);
	void setup_system(const ArgParser& parameters, GeoTypes::Mixer mixer);

	// helpers:
	// std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, 
	// 	unsigned int number_groups) const;

	std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, 
		unsigned int number_groups, 
		double left_length) const;

	void update_chemicals();

	// output:
	void output_grid(const std::string& file_name = "grid") const;
	void output_chemicals() const;
	void output_bacteria() const;

	void run_microbes(); // called by others after setting up
}; 


/** IMPLEMENTATION
*/
//------------------------------------------------------------------------------------------------------

template<int dim, int numchem>
Simulator<dim, numchem>::Simulator(const unsigned int number_refinements,
								const unsigned int stokes_degree)
	:
	triangulation (Triangulation<dim>::maximum_smoothing),
	stokes_handler(triangulation, number_refinements, stokes_degree),
	chemical_fe_base(triangulation, stokes_degree), // using same degree for chemicals
	// chemical(chemical_fe_base),
	output_directory("."),
	time(0),
	save_step_number(0),
	time_step_number(0) 
{}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_system_constants(const ArgParser& parameters)
{
	std::cout << "...Setting up system constants" << std::endl;
	run_time = parameters.getRunTime();
	save_period = parameters.getSavePeriod();

	output_directory = parameters.getOutputDirectory();
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_geometry(const ArgParser& parameters)
{
	std::cout << "...Initializing geometry" << std::endl;
	geometry.initialize(parameters.getGeometryFile(), 
						parameters.getMeshFile());

	output_geometry();
}
	

template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_geometry(GeoTypes::Filter filter)
{
	std::cout << "...Initializing FILTER geometry" << std::endl;

	geometry.initialize(filter);

	output_geometry();
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_geometry(GeoTypes::Mixer mixer)
{
	std::cout << "...Initializing MIXER geometry" << std::endl;

	geometry.initialize(mixer);

	output_geometry();
}


template<int dim, int numchem>
void
Simulator<dim, numchem>::output_geometry() const
{
	std::string geo_out_file = output_directory + "/geometryInfo.dat";
	std::ofstream geo_out(geo_out_file);
	geometry.printInfo(geo_out);
	geometry.printInfo(std::cout);

	// output boundary and obstacles for easier postprocessing:
	geometry.outputGeometry(output_directory); 
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_grid(unsigned int initial_refinement)
{
	GridGenerationTools::build_mesh_from_geometry(geometry, triangulation);
	triangulation.refine_global(initial_refinement);
	output_grid("before_stokes_grid");
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::output_grid(const std::string& file_name) const
{
	std::string grid_out_file = output_directory + "/" + file_name + ".eps";

	std::ofstream out (grid_out_file);
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "...Grid written to " << grid_out_file << std::endl;
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::solve_stokes()
{
	// get boundary conditions:
	std::vector<unsigned int> no_slip_ids = {2, 3}; 
	if(dim == 3)
	{
		no_slip_ids.push_back(4);
		no_slip_ids.push_back(5);
	}

	unsigned int sphere_id = GridGenerationTools::id_sphere_begin;
	for(unsigned int i = 0; i < geometry.getNumberSpheres(); ++i, ++sphere_id)
		no_slip_ids.push_back(sphere_id);

	unsigned int rect_id = GridGenerationTools::id_rectangle_begin;
	for(unsigned int i = 0; i < geometry.getNumberRectangles(); ++i, ++rect_id)
		no_slip_ids.push_back(rect_id);

	// set boundaries (taking 0 to be inlet, and 1 to be outlet):
	stokes_handler.setNoSlipIDs(no_slip_ids);

	// solve and output:
	stokes_handler.solve(output_directory); 
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_chemicals(const ArgParser& parameters)
{
	chemical_fe_base.setup_chemical_base(stokes_handler);

	/** initialize chemicals
	*	later perhaps would like to have argparser give separate values for each chemical...
	* @todo generalize argparser for multiple chemicals...
	*/
	for(unsigned int i = 0; i < numchem; ++i)
	{
		chemicals[i].reinit(chemical_fe_base);
		chemicals[i].setDiffusionConstant( parameters.getGoodDiffusionConstant() );
		chemicals[i].setDecayConstant( parameters.getGoodDecayConstant() );
	}

	// waste chemical:
	chemicals[numchem-1].setDiffusionConstant( parameters.getWasteDiffusionConstant() );
	chemicals[numchem-1].setDecayConstant( parameters.getWasteDecayConstant() );
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_secretion_map(double resolution)
{
	std::cout << "...Setting up point-cell map" << std::endl;
	point_cell_map.initialize(geometry, chemical_fe_base.get_dof_handler(), resolution); // using default resolution
	point_cell_map.printInfo(std::cout); 
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_bacteria(const ArgParser& parameters,
										double left_length)
{
	const unsigned int number_bacteria = parameters.getNumberBacteria();
	const unsigned int number_groups = parameters.getNumberGroups();

	if(number_bacteria < 1)
		return;

	std::cout << "...Initializing bacteria" << std::endl;

	// Need a better, more general way to get region of desired initial locations ...
	std::vector<Point<dim> > locations = 
		getBacteriaLocations(number_bacteria, number_groups, left_length); /* move to a simulator tools namespace ...*/

	std::array<double, numchem> rates;
	for(unsigned int i = 0; i < numchem; ++i)
		rates[i] = parameters.getGoodSecretionRate();
	rates[numchem-1] = parameters.getWasteSecretionRate();

	bacteria.initialize(parameters.getBacteriaDiffusionConstant(),
						number_bacteria,
						rates,
						locations);

	bacteria.printInfo(std::cout);
}


template<int dim, int numchem>
std::vector<Point<dim> > 
Simulator<dim, numchem>::getBacteriaLocations(unsigned int number_bacteria, 
      unsigned int number_groups, double left_length) const
{
  std::cout << "...Finding group positions" << std::endl;
  if(number_groups == 0)
    number_groups = number_bacteria; // no groups same as NB ``groups''

  std::vector<Point<dim> > group_locations;
  group_locations.reserve(number_groups);

  Point<dim> temp_point;
  for(unsigned int i = 0; i < number_groups; i++)
  {
    bool found = false;

    while(!found)
    {
      // temp_point[dim_itr] = (xmax-xmin)*((double)rand() / RAND_MAX) + xmin;
      for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
      {
      	double width = geometry.getWidth(dim_itr);

      	/// possibly initialize in subdomain:
      	if( (dim_itr == 0) && (left_length > 0) )
      		width = left_length; 

        temp_point[dim_itr] = (width)*((double)rand() / RAND_MAX) 
          + geometry.getBottomLeftPoint()[dim_itr];
      }

      if( geometry.isInDomain(temp_point) )
      {
        group_locations.push_back(temp_point);
        found = true;
      }
    } // while not found
  } // for group locations
  std::cout << "...Group positions found." << std::endl;

  return group_locations;
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_fitness(const ArgParser parameters)
{
	std::array<FE_Chemical<dim>*, numchem> chem_ptrs;
	for(unsigned int i = 0; i < numchem; ++i)
		chem_ptrs[i] = &chemicals[i];

	fitness_function.attach_chemicals(chem_ptrs);

	/** would like to generalize this if possible, argparser needs to store things in
	* a better manner
	*/
	fitness_function.setup_fitness_constants(parameters.getAlphaGood(),
											parameters.getAlphaWaste(),
											parameters.getGoodSaturation(),
											parameters.getWasteSaturation(),
											parameters.getSecretionCost());

	fitness_function.printInfo(std::cout);
}


template<int dim, int numchem>
double
Simulator<dim, numchem>::get_CFL_time_step() const
{
	const double min_time_step = 0.01;

	const double maximal_velocity = stokes_handler.get_maximal_velocity();
	double cfl_time_step = 0;

	std::cout << "...maximal_velocity is: " << maximal_velocity << std::endl;

	if(maximal_velocity >= 0.01)
		cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
			chemical_fe_base.get_fe_degree() *
			GridTools::minimal_cell_diameter(triangulation) /
			maximal_velocity;
	else
		cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
			chemical_fe_base.get_fe_degree() *
			GridTools::minimal_cell_diameter(triangulation) /
			0.01;

	cfl_time_step = std::min(min_time_step,cfl_time_step);

	std::cout << "...using time step: " << cfl_time_step << std::endl;

	return cfl_time_step;
}


/// setup full system including solving stokes equation, in preparation for 
/// running microbe simulation
template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_system(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_grid(parameters.getGlobalRefinement()); // from geometry 
	solve_stokes();
	setup_chemicals(parameters);

	const double resolution = 0.2;
	setup_secretion_map(resolution);
	
	setup_bacteria(parameters);
	setup_fitness(parameters); 
	time_step = get_CFL_time_step();
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_system(const ArgParser& parameters,
										GeoTypes::Filter filter)
{
	setup_system_constants(parameters);
	setup_geometry(filter);
	setup_grid(parameters.getGlobalRefinement()); // from geometry 
	solve_stokes();
	setup_chemicals(parameters);

	const double resolution = 0.2;
	setup_secretion_map(resolution);
	
	setup_bacteria(parameters, filter.left_length);
	setup_fitness(parameters); 
	time_step = get_CFL_time_step();
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::setup_system(const ArgParser& parameters,
										GeoTypes::Mixer mixer)
{
	setup_system_constants(parameters);
	setup_geometry(mixer);
	setup_grid(parameters.getGlobalRefinement()); // from geometry 
	solve_stokes();
	setup_chemicals(parameters);

	const double resolution = 0.2;
	setup_secretion_map(resolution);
	
	setup_bacteria(parameters, mixer.left_length);
	setup_fitness(parameters); 
	time_step = get_CFL_time_step();
}

/* OUTPUT
*/

template<int dim, int numchem>
void 
Simulator<dim, numchem>::output_chemicals() const
{
	for(unsigned int i = 0; i < numchem; ++i)
		chemicals[i].output_solution(output_directory, i, save_step_number);
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::output_bacteria() const
{
	std::string outfile = output_directory
							+ "/bacteria_" 
							+ Utilities::int_to_string(save_step_number,4)
							+ ".dat";
	std::ofstream out(outfile);
	bacteria.print(out);
}


template<int dim, int numchem>
void
Simulator<dim, numchem>::update_chemicals()
{
	const unsigned int number_bacteria = bacteria.getSize();
	std::vector<Point<dim> > locations(number_bacteria);

	for(unsigned int b = 0; b < number_bacteria; ++b)
		locations[b] = bacteria.getLocation(b);

	for(unsigned int c = 0; c < numchem; ++c)
	{
		const unsigned int number_bacteria = bacteria.getSize();
		std::vector<double> amounts(number_bacteria);

		for(unsigned int b = 0; b < number_bacteria; ++b)
			amounts[b] = bacteria.getSecretionRate(b, c);

		chemicals[c].update(time_step, point_cell_map, geometry, locations, amounts);
	}
}

/** RUN:
* Main run function:
*/

template<int dim, int numchem>
void
Simulator<dim, numchem>::run_microbes()
{
	// loop through time:
	std::cout << "\n\n--------------------------------------------------------------------\n"
		<< "\t Starting microbe simulation...\n" << std::endl;
	do{
		// update chemicals:
		update_chemicals();

		// update bacteria:
		bacteria.randomWalk(time_step, &geometry, 
			stokes_handler.get_dof_handler(), stokes_handler.get_solution()); /// *** make this better
		bacteria.reproduce(time_step, fitness_function);
		// bacteria.mutate()

		// update time:
		time += time_step;
		++time_step_number;

		// output:
		if(time_step_number % save_period == 0)
		{
			std::cout << "saving at time: " << time << std::endl;
	   		output_chemicals();
	   		output_bacteria();
	   		++save_step_number;
	   	}

	}while( (time < run_time) && bacteria.isAlive() );
}

//--------------------------------------------------------------------------------------------
template<int dim, int numchem>
void 
Simulator<dim, numchem>::run(const ArgParser& parameters)
{
	setup_system(parameters);

	run_microbes();
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::run(const ArgParser& parameters, 
									GeoTypes::Filter filter)
{
	setup_system(parameters, filter);

	run_microbes();
}


template<int dim, int numchem>
void 
Simulator<dim, numchem>::run(const ArgParser& parameters, 
									GeoTypes::Mixer mixer)
{
	setup_system(parameters, mixer);

	run_microbes();
}



// DEBUG:
// -----------------------------------------------------------------------------------------

template<int dim, int numchem>
void
Simulator<dim, numchem>::run_test()
{

	// setup_system_constants(parameters);
	// setup_geometry(parameters);
	// setup_grid();

	// GridGenerationTools::build_mesh_from_geometry(geometry, triangulation);

	GridGenerationTools::build_mixer_mesh(5., 3., 3., 1., triangulation, true);
	triangulation.refine_global(2);

    {
      std::string grid_out_file = "before_stokes_grid.eps";

      std::ofstream out (grid_out_file);
      GridOut grid_out;
      grid_out.write_eps (triangulation, out);
      std::cout << "...Grid written to " << grid_out_file << std::endl;
    }

	std::vector<unsigned int> no_slip_ids = {2, 3, 10, 11}; // for mixer...

	stokes_handler.setNoSlipIDs(no_slip_ids);

	stokes_handler.solve(output_directory); // number of cycles ..., output each cycle...

    {
      std::string grid_out_file = "after_stokes_grid.eps";

      std::ofstream out (grid_out_file);
      GridOut grid_out;
      grid_out.write_eps (triangulation, out);
      std::cout << "...Grid written to " << grid_out_file << std::endl;
    }

    chemical_fe_base.setup_chemical_base(stokes_handler);

    chemicals[0].reinit(chemical_fe_base);
    chemicals[0].setDiffusionConstant(1.0);
    chemicals[0].setDecayConstant(0.0);

    /** TEST CHEMICAL EVOLUTION:
    */
    chemicals[0].project_initial_condition(ExactSolutions::Gaussian<dim>(Point<dim>(2. ,1.5), 0.5));

   	time_step = 0.001; // get cfl from stokes ...
   	run_time = 1.0;
   	time = 0.;
   	save_step_number = 0;
   	time_step_number = 0;
   	save_period = 10; 

   	do{
   		chemicals[0].update(time_step);
   		time += time_step;
		++time_step_number;

		if(time_step_number % save_period == 0)
		{
			std::cout << "saving at time: " << time << std::endl;
	   		chemicals[0].output_solution(".", 101, save_step_number);
	   		++save_step_number;
	   	}


   	}while(time < run_time);


}


}}



#endif



