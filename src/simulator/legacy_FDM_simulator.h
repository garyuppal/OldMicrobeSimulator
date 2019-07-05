#ifndef MICROBE_SIMULATOR_FDM_SIMULATOR
#define MICROBE_SIMULATOR_FDM_SIMULATOR


#include "../utility/argparser.h"
#include "../discrete_field/FDM_chemical.h"
#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "../bacteria/bacteria.h"

#include "../bacteria/fitness.h"


#include "../discrete_field/discrete_function.h"

namespace MicrobeSimulator{

	template<int dim>
	class Gaussian : public DiscreteFunction<dim>{
	public:
		Gaussian()
		{
			for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
				center[dim_itr] = 3.5;
		}
		~Gaussian(){}
		double value(const Point<dim>& p) const override
		{
			return  exp( -(p-center)*(p-center) );
		}

	private:
		Point<dim> center;
	};


	template<int dim>
	class FDMSimulator{
	public:
		FDMSimulator();

		void run(const ArgParser& systemParameters);
		void run_debug(const ArgParser& systemParameters);
		void run_chemDebug(const ArgParser& systemParameters);

	private:
		Geometry<dim> geometry;
		AdvectionField<dim> advectionField;
		Bacteria<dim> bacteria;

		FDMChemical<dim> public_good;
		FDMChemical<dim> waste;

	    double time; // t
	    double time_step; // dt
	    unsigned int timeStepNumber; // n
		const unsigned int reproduction_delay;

		unsigned int save_period;
		unsigned int saveStepNumber;
		double run_time;

		void setup_system(const ArgParser& systemParameters);
  		std::vector<Point<dim> > getBacteriaLocations(const Geometry<dim>& geometry, 
			unsigned int number_bacteria, unsigned int number_groups);
	};

//IMPLEMENTATION
//---------------------------------------------------------------------------

	template<int dim>
	FDMSimulator<dim>::FDMSimulator()
		:
		time(0),
		timeStepNumber(0),
		reproduction_delay(5),
		saveStepNumber(0)
	{}


	template<int dim>
	void FDMSimulator<dim>::setup_system(const ArgParser& systemParameters)
	{
		time_step = systemParameters.getTimeStep();
		run_time = systemParameters.getRunTime();
		save_period = systemParameters.getSavePeriod();	

		std::cout << "...Initializing geometry" << std::endl;
		geometry.initialize(systemParameters.getGeometryFile());

		std::vector<Point<dim> > locations = getBacteriaLocations(geometry, 
			systemParameters.getNumberBacteria(), systemParameters.getNumberGroups());

		std::cout << "...Initializing advection" << std::endl;
		advectionField.initialize(systemParameters.getVelocityType(),
									geometry.getBottomLeftPoint(),
									geometry.getTopRightPoint(),
									systemParameters.getMaximumVelocity()
									);
							//  double vrad = 0, double vrotation = 0);

		std::cout << "...Initializing bacteria" << std::endl;
		bacteria.initialize(systemParameters.getBacteriaDiffusionConstant(), 
			systemParameters.getNumberBacteria(),
			locations,
			systemParameters.getGoodSecretionRate(),
			 systemParameters.getWasteSecretionRate() );

		std::cout << "...Setting fitness constants" << std::endl;
		bacteria.setFitnessConstants(systemParameters.getAlphaGood(),
									systemParameters.getAlphaWaste(),
									systemParameters.getGoodSaturation(),
									systemParameters.getWasteSaturation(),
									systemParameters.getSecretionCost() );
		//double a1, double a2, double k1, double k2, double b);


		std::cout << "using discretization: " << std::endl;
		for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
			std::cout << "\t" << geometry.getDiscretization()[dim_itr] << std::endl;
		std::cout << std::endl;

		std::cout << "...Initializing chemicals" << std::endl;
		public_good.initialize(geometry.getBottomLeftPoint(),geometry.getTopRightPoint(),
			geometry.getDiscretization(),geometry.getBoundaryConditions());

		public_good.setDiffusionConstant(systemParameters.getGoodDiffusionConstant());
		public_good.setDecayConstant(systemParameters.getGoodDecayConstant());

		waste.initialize(geometry.getBottomLeftPoint(),geometry.getTopRightPoint(),
			geometry.getDiscretization(),geometry.getBoundaryConditions());

		waste.setDiffusionConstant(systemParameters.getWasteDiffusionConstant());
		waste.setDecayConstant(systemParameters.getWasteDecayConstant());


	}

	template<int dim>
	std::vector<Point<dim> > FDMSimulator<dim>::getBacteriaLocations(const Geometry<dim>& geometry, 
				unsigned int number_bacteria, unsigned int number_groups)
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
					temp_point[dim_itr] = (geometry.getWidth(dim_itr))*((double)rand() / RAND_MAX) 
						+ geometry.getBottomLeftPoint()[dim_itr];

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



	template<int dim>
	void FDMSimulator<dim>::run(const ArgParser& systemParameters)
	{
		setup_system(systemParameters);
	
		// const double scale = 1.0/625.0; // need to incorporate this better ...
		const double scale = 1.0/25.0; // need to incorporate this better ...


		bacteria.print(std::cout);

		do{
			// update chemicals:
			for(unsigned int i = 0; i < bacteria.getSize(); i++)
			{
				public_good.secreteTo(bacteria.getLocation(i),
					scale*time_step*bacteria.getGoodSecretionRate(i));
				waste.secreteTo(bacteria.getLocation(i),
					scale*time_step*bacteria.getWasteSecretionRate(i));
			}

			public_good.update(time_step, &advectionField); 
			waste.update(time_step, &advectionField);

			// update bacteria:
		   if(timeStepNumber > reproduction_delay)
		   {
				bacteria.reproduce(time_step, public_good, waste);
				// std::cout << "now with " << bacteria.getSize() << " bacteria" << std::endl;
				// bacteria.mutateBacteria(mutation_rate,mutation_diff);
				bacteria.randomWalk(time_step, &geometry, &advectionField);		
			}

			// update time:
			time += time_step;
			++timeStepNumber;

			// output:
			if(timeStepNumber % save_period == 0) // could put into a function output_results()
			{
				// output bacteria:
				std::string outfile = systemParameters.getOutputDirectory() 
					+ "/bacteria" + std::to_string(saveStepNumber) + ".dat";
				std::ofstream out(outfile);
				bacteria.print(out);

				// output public good:
				std::string goodOutfile = systemParameters.getOutputDirectory() 
					+ "/chemical01_" + std::to_string(saveStepNumber) + ".dat";
				std::ofstream goodOut(goodOutfile);
				public_good.print(goodOut);

				// output waste field:
				std::string wasteOutfile = systemParameters.getOutputDirectory() 
					+ "/chemical02_" + std::to_string(saveStepNumber) + ".dat";
				std::ofstream wasteOut(wasteOutfile);
				waste.print(wasteOut);

				std::cout << "now with " << bacteria.getSize() << " bacteria" << std::endl;

				saveStepNumber++;
				std::cout << "time: " << time << std::endl;
				// output_results(saveStep); // output chemicals
			}		


		} while(time <= run_time);
	}	  		














	template<int dim>
	void FDMSimulator<dim>::run_debug(const ArgParser& systemParameters)
	{
		setup_system(systemParameters);

		bacteria.print(std::cout);
		std::cout << "\nPUBLIC GOOD\n" << std::endl;
		public_good.print(std::cout);

		std::cout << "\nWASTE\n" << std::endl;
		waste.print(std::cout);

		const bool testing_chemicals = true;
		const bool testing_bacteria = true;

		std::cout << "\n...Running FDM debug in " 
			<< systemParameters.getDimension() << " dimensions...\n" << std::endl;

		const double scale = 1.0/625.0;

		std::cout << "time step is: " << time_step << std::endl;

		do{

			//update chemicals:
			// secrete
			// update
			if(testing_chemicals)
			{

				// if(timeStepNumber < 10)
				// {
					// update chemicals:
					// std::cout << "... secrete" << std::endl;
					for(unsigned int i = 0; i < bacteria.getSize(); i++)
					{
						public_good.secreteTo(bacteria.getLocation(i),
							scale*time_step*bacteria.getGoodSecretionRate(i));
						// waste.secreteTo(bacteria.getLocation(i),
						// 	scale*time_step*bacteria.getWasteSecretionRate(i));
					}
				// }	 
				// std::cout << "... update" << std::endl;
				public_good.update(time_step);
				// waste.update(time_step);
			}
			if(testing_bacteria)
			{

				// update bacteria:
			    // if(timeStepNumber > reproduction_delay)
					// bacteria.reproduce(time_step, dof_handler, solution1, solution2);
	//			std::cout << "now with " << bacteria.getSize() << " bacteria" << std::endl;
				// bacteria.mutateBacteria(mutation_rate,mutation_diff);
				// std::cout << "TIME STEP: " << timeStepNumber << std::endl;
				// bacteria.print(std::cout);
				// bacteria.randomWalk(time_step, &geometry);		
			}

			// update time:
			time += time_step;
			++timeStepNumber;

			// std::cout << "Time: " << time<< std::endl;
			// std::cout << "timeStepNumber: " << timeStepNumber << std::endl;
			// std::cout << "save_period: " << save_period << std::endl;
			// std::cout << "saveStepNumber: " << saveStepNumber << std::endl;
			// std::cout << "run time: " << run_time << std::endl;
			// output:
			if(timeStepNumber % save_period == 0) // could put into a function output_results()
			{
				// output bacteria:
				std::string outfile = systemParameters.getOutputDirectory() 
					+ "/bacteria" + std::to_string(saveStepNumber) + ".dat";
				std::ofstream out(outfile);
				bacteria.print(out);

				// output public good:
				std::string goodOutfile = systemParameters.getOutputDirectory() 
					+ "/chemical01_" + std::to_string(saveStepNumber) + ".dat";
				std::ofstream goodOut(goodOutfile);
				public_good.print(goodOut);


				// std::cout << "\n\nPublic good field:\n\n" << std::endl;
				// public_good.print(std::cout);

				// std::cout << "\n\nPG Aux field:\n\n" << std::endl;
				// public_good.printAux(std::cout);

				// // output bacteria:
				// std::string wasteOutfile = systemParameters.getOutputDirectory() 
				// 	+ "/chemical02_" + std::to_string(saveStepNumber) + ".dat";
				// std::ofstream wasteOut(wasteOutfile);
				// waste.print(wasteOut);

				saveStepNumber++;
				std::cout << "time: " << time << std::endl;
				// output_results(saveStep); // output chemicals
			}		


		} while(time <= run_time);
	}


	template<int dim>
	void FDMSimulator<dim>::run_chemDebug(const ArgParser& systemParameters)
	{
		Gaussian<dim> my_function;

		setup_system(systemParameters);

		// std::cout << "\n\nBEFORE FUNCTION\n\n";
		// public_good.print(std::cout);

		public_good.projectFunction(my_function);

		// std::cout << "\n\nAFTER FUNCTION\n\n";
		// public_good.print(std::cout);
		// output public good:
		std::string goodOutfile = systemParameters.getOutputDirectory() 
			+ "/chemical01_" + std::to_string(saveStepNumber) + ".dat";
		std::ofstream goodOut(goodOutfile);
		public_good.print(goodOut);

		std::cout << "\n\nTIME LOOP: advect and diffuse\n\n" << std::endl;

		do{
			// update chemical:
			public_good.update(time_step, &advectionField); 

			// update time:
			time += time_step;
			++timeStepNumber;

			// output:
			if(timeStepNumber % save_period == 0) // could put into a function output_results()
			{
				++saveStepNumber;

				// output public good:
				std::string goodOutfile = systemParameters.getOutputDirectory() 
					+ "/chemical01_" + std::to_string(saveStepNumber) + ".dat";
				std::ofstream goodOut(goodOutfile);
				public_good.print(goodOut);

				std::cout << "time: " << time << std::endl;
				// output_results(saveStep); // output chemicals
			}		

		} while(time <= run_time);

	}



}


#endif

