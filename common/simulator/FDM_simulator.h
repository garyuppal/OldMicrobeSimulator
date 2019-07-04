#ifndef MICROBE_SIMULATOR_FDM_SIMULATOR
#define MICROBE_SIMULATOR_FDM_SIMULATOR


#include "../utility/argparser.h"
#include "../discrete_field/FDM_chemical.h"
#include "../discrete_field/discrete_function.h"

#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "../bacteria/bacteria.h"

#include "../bacteria/fitness.h"


namespace MicrobeSimulator{

	template<int dim, int NumberChemicals>
	class AndFitnessFunction : public FitnessBase<dim, NumberChemicals>
	{
	public:
		AndFitnessFunction();
		~AndFitnessFunction();

		virtual double value(const Point<dim>& location, 
			const std::array<double, NumberChemicals>& secretion_rates) const;

		void setup_fitness(std::array<FDMChemical<dim>*, NumberChemicals> cps,
			double pg_bene, double w_harm, double pg_sat, double w_sat, 
			double beta, double gamma);

		void printInfo(std::ostream& out) const;

	private: 
		std::array<FDMChemical<dim>*, NumberChemicals> chemical_pointers; 
		// @ todo -- could do array of smart pointers?

		double public_good_benefit;
		double waste_harm;

		double public_good_saturation;
		double waste_saturation;

		double secretion_cost; // same for 1 and 2 -- zero for 3
		double inefficiency_penalty; 		
	};


	template<int dim, int NumberChemicals>
	AndFitnessFunction<dim, NumberChemicals>::AndFitnessFunction()
	{}


	template<int dim, int NumberChemicals>
	AndFitnessFunction<dim, NumberChemicals>::~AndFitnessFunction()
	{}


	template<int dim, int NumberChemicals>
	void 
	AndFitnessFunction<dim, NumberChemicals>::setup_fitness(
		std::array<FDMChemical<dim>*, NumberChemicals> cps,
		double pg_bene, double w_harm, double pg_sat, double w_sat, 
		double beta, double gamma)
	{
		chemical_pointers = cps;
		
		public_good_benefit = pg_bene;
		waste_harm = w_harm;

		public_good_saturation = pg_sat;
		waste_saturation = w_sat;

		secretion_cost = beta;
		inefficiency_penalty = gamma;
	}


	template<int dim, int NumberChemicals>
	double
	AndFitnessFunction<dim, NumberChemicals>::value(const Point<dim>& location, 
			const std::array<double, NumberChemicals>& secretion_rates) const
	{
		const double c1 = chemical_pointers[0]->value(location);
		const double c2 = chemical_pointers[1]->value(location);
		const double cw = chemical_pointers[2]->value(location);

		// std::cout << "c1 = " << c1 << std::endl
		// 		<< "c2 = " << c2 << std::endl
		// 		<< "cw = " << cw << std::endl;

		const double is_generalist = ( (secretion_rates[0] != 0) && (secretion_rates[1] != 0)  ?
									0. : 1. );

		const double return_value = 
			public_good_benefit*(c1*c2)/( c1*c2 + public_good_saturation )
			- waste_harm*cw/( cw + waste_saturation )
			- secretion_cost*secretion_rates[0] - secretion_cost*secretion_rates[1]
			- inefficiency_penalty*is_generalist;

		// std::cout << "fitness is: " << return_value << std::endl;

		return return_value;
	}


	template<int dim, int NumberChemicals>
	void 
	AndFitnessFunction<dim, NumberChemicals>::printInfo(std::ostream& out) const
	{
		out << "\n\n-----------------------------------------------------" << std::endl
			<< "\t\tFITNESS FUNCTION (for " << NumberChemicals << " chemicals)" << std::endl
			<< "-----------------------------------------------------" << std::endl
			<< "\t public good benefit: " << public_good_benefit << std::endl
			<< "\t waste harm: " << waste_harm << std::endl
			<< "\t public good saturation: " << public_good_saturation << std::endl
			<< "\t waste saturation: " << waste_saturation << std::endl
			<< "\t secretion cost: " << secretion_cost << std::endl
			<< "\t inefficiency penalty: " << inefficiency_penalty << std::endl;
	}


// simulator---------------------------------------------------------------------------------------------
	template<int dim, int NumberChemicals>
	class FDMSimulator{
	public:
		FDMSimulator();

		void run(const ArgParser& parameters);
		void run_chem_debug(const ArgParser& parameters);

	private:
		Geometry<dim> geometry;
		AdvectionField<dim> advection_field;
		Bacteria<dim, NumberChemicals> bacteria;

		std::array<FDMChemical<dim>, NumberChemicals> chemicals;		

		AndFitnessFunction<dim, NumberChemicals> fitness_function;

		std::string output_directory;

		// FDMChemical<dim> public_good;
		// FDMChemical<dim> waste;

	    double time; // t
	    double time_step; // dt
	    unsigned int timeStepNumber; // n
		const unsigned int reproduction_delay;

		unsigned int save_period;
		unsigned int saveStepNumber;
		double run_time;

		void setup_system(const ArgParser& parameters);

		void setup_system_constants(const ArgParser& parameters);
		void setup_geometry(const ArgParser& parameters);
		void setup_advection(const ArgParser& parameters);
		void setup_bacteria(const ArgParser& parameters);
		void setup_chemicals(const ArgParser& parameters);
		void setup_fitness(const ArgParser& parameters);

  		std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, 
  			unsigned int number_groups);

  		void output_chemicals() const;
  		void output_bacteria() const;
	};



	// chemical for debugging:
	//-----------------------------------------------------------------------
	template<int dim> // could add a default template parameter...
	class Gaussian : public DiscreteFunction<dim>{
	public:
		Gaussian()
			:
			DiscreteFunction<dim>()
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


//IMPLEMENTATION
//---------------------------------------------------------------------------

	template<int dim, int NumberChemicals>
	FDMSimulator<dim, NumberChemicals>::FDMSimulator()
		:
		time(0),
		timeStepNumber(0),
		reproduction_delay(5),
		saveStepNumber(0)
	{
        // implementing here for two public goods -> three chemicals
        if(NumberChemicals != 3)
            throw std::runtime_error("ERROR: Class implented here for 3 chemicals");
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_system(const ArgParser& parameters)
	{
		setup_system_constants(parameters);
		setup_geometry(parameters);
		setup_advection(parameters);
		setup_bacteria(parameters);
		setup_chemicals(parameters);
		setup_fitness(parameters);
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_system_constants(const ArgParser& parameters)
	{
		std::cout << "...Setting up system constants" << std::endl;
		
		run_time = parameters.getRunTime();
		save_period = parameters.getSavePeriod();
		time_step = parameters.getTimeStep();

		output_directory = parameters.getOutputDirectory();
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_geometry(const ArgParser& parameters)
	{
		std::cout << "...Initializing geometry" << std::endl;
		geometry.initialize( parameters.getGeometryFile() );

		std::string geo_out_file = output_directory + "/geometryInfo.dat";
		std::ofstream geo_out(geo_out_file);
		geometry.printInfo(geo_out);
		geometry.printInfo(std::cout);

		geometry.outputGeometry(output_directory); // boundary and obstacles for easy viewing
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_advection(const ArgParser& parameters)
	{
		std::cout << "...Initializing advection" << std::endl;

		if( (parameters.getVelocityType() != VelocityType::NUMERICAL_FLOW)
			&& (parameters.getVelocityType() != VelocityType::TILE_FLOW) )
		{
			advection_field.initialize(parameters.getVelocityType(),
		            geometry.getBottomLeftPoint(),
		            geometry.getTopRightPoint(),
		            parameters.getMaximumVelocity()
		            );
		        //  double vrad = 0, double vrotation = 0);
		}
		else
		{
			advection_field.initialize(parameters.getVelocityType(),
		            geometry.getBottomLeftPoint(),
		            geometry.getTopRightPoint(),
		            geometry.getScaleFactor(),
		            parameters.getMaximumVelocity(),
		            parameters.getVelocityFile_X(),
		            parameters.getVelocityFile_Y() );
		}

		advection_field.printInfo(std::cout);
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_bacteria(const ArgParser& parameters)
	{
		const unsigned int number_bacteria = parameters.getNumberBacteria();

		if(number_bacteria < 1)
			return;

		std::cout << "...Initializing bacteria" << std::endl;

		std::cout << "\t... getting group locations" << std::endl;
		std::vector<Point<dim> > locations = getBacteriaLocations(
													parameters.getNumberBacteria(),
													parameters.getNumberGroups() );

		std::array<double, NumberChemicals> rates = {
				parameters.getGoodSecretionRate(),
				parameters.getGoodSecretionRate(),
				parameters.getWasteSecretionRate()
			};

		bacteria.initialize(parameters.getBacteriaDiffusionConstant(), 
			number_bacteria,
			rates,
			locations);

		bacteria.printInfo(std::cout);
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_chemicals(const ArgParser& parameters)
	{
		std::cout << "...setting up chemicals" << std::endl;

		std::cout << "\t ...using discretization: " << std::endl;
		for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
			std::cout << "\t" << geometry.getDiscretization()[dim_itr] << std::endl;
		std::cout << std::endl;

		for(unsigned int i = 0; i < NumberChemicals; i++)
			chemicals[i].initialize(geometry.getBottomLeftPoint(),
									geometry.getTopRightPoint(),
									geometry.getDiscretization(),
									geometry.getBoundaryConditions());

		chemicals[0].setDiffusionConstant(parameters.getGoodDiffusionConstant());
		chemicals[0].setDecayConstant(parameters.getGoodDecayConstant());

		chemicals[1].setDiffusionConstant(parameters.getGoodDiffusionConstant());
		chemicals[1].setDecayConstant(parameters.getGoodDecayConstant());

		chemicals[2].setDiffusionConstant(parameters.getWasteDiffusionConstant());
		chemicals[2].setDecayConstant(parameters.getWasteDecayConstant());

		for(unsigned int i = 0; i < NumberChemicals; i++)
			chemicals[i].printInfo(std::cout);
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::setup_fitness(const ArgParser& parameters)
	{
		std::cout << "...Setting up fitness function" << std::endl;

		fitness_function.setup_fitness(
			std::array<FDMChemical<dim>*, NumberChemicals>{&chemicals[0], &chemicals[1], &chemicals[2]},
			parameters.getAlphaGood(),
			parameters.getAlphaWaste(),
			parameters.getGoodSaturation(),
			parameters.getWasteSaturation(),
			parameters.getSecretionCost(),
			parameters.getInefficiencyPenalty());

		fitness_function.printInfo(std::cout); 
	}


	template<int dim, int NumberChemicals>
	std::vector<Point<dim> > 
	FDMSimulator<dim, NumberChemicals>::getBacteriaLocations(unsigned int number_bacteria, 
		unsigned int number_groups)
	{
		std::cout << "\t...Finding group positions" << std::endl;
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


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::output_chemicals() const
	{
		for(unsigned int i = 0; i < NumberChemicals; i++)
		{
			std::string filename = output_directory 	
								+ "/chemical"
								+ std::to_string(i)
								+ "_"
								+ std::to_string(saveStepNumber)
								+ ".dat";

			std::ofstream out(filename);
			chemicals[i].print(out);
		}
	}


	template<int dim, int NumberChemicals>
	void 
	FDMSimulator<dim, NumberChemicals>::output_bacteria() const
	{
		std::string filename = output_directory
							+ "/bacteria_"
							+ std::to_string(saveStepNumber)
							+ ".dat";

		std::ofstream out(filename);
		bacteria.print(out);
	}


// RUN --------------------------------------------------------------------------------------------
	template<int dim, int NumberChemicals>
	void
	FDMSimulator<dim, NumberChemicals>::run(const ArgParser& parameters)
	{
		setup_system(parameters);

		const double mutation_rate = parameters.getMutationRate();
		const double original_secretion_rate = parameters.getGoodSecretionRate();
		const double secretion_factor = 1./25.;

		do{

			// update chemicals:
			for(unsigned int chem = 0; chem < NumberChemicals; chem++)
			{
				for(unsigned int i = 0; i < bacteria.getSize(); i++)
					chemicals[chem].secreteTo(bacteria.getLocation(i),
								time_step*secretion_factor*bacteria.getSecretionRate(i,chem));

				chemicals[chem].update(time_step); //, &advection_field); 
				//@todo -- chemical should know when there is no flow -- to speed up
			}


			// update bacteria:
			if(timeStepNumber > reproduction_delay)
				bacteria.reproduce(time_step, fitness_function);

			// std::cout << "with " << bacteria.getSize() << " total bacteria" << std::endl;

			bacteria.mutate(time_step, 
							mutation_rate, 
							/* delta secretion */ 0, 
							original_secretion_rate, 
							/* binary mutation */ true); 

			bacteria.randomWalk(time_step, &geometry); 

			// update time:
			time += time_step;
			++timeStepNumber;

			// output:
			if( timeStepNumber % save_period == 0)
			{
				std::cout << "time: " << time << std::endl;
				++saveStepNumber;

				output_bacteria();
				output_chemicals();
			}

		}while( time <= run_time && bacteria.isAlive() );
	}


	template<int dim, int NumberChemicals>
	void
	FDMSimulator<dim, NumberChemicals>::run_chem_debug(const ArgParser& parameters)
	{
		setup_system(parameters);

		Gaussian<dim> test_function;

		for(unsigned int i = 1; i < NumberChemicals; i++)
			chemicals[i].projectFunction(test_function);

		do{
			for(unsigned int i = 0; i < NumberChemicals; i++)
				chemicals[i].update(time_step);

			fitness_function.value(Point<dim>(0,0),
				std::array<double, NumberChemicals>{0,0,0});

			time += time_step;
			++timeStepNumber;

			if( timeStepNumber % save_period == 0)
			{
				std::cout << "time: " << time << std::endl;
				++saveStepNumber;

				output_chemicals();
			}
		}while( time <= run_time );
	}



}


#endif

