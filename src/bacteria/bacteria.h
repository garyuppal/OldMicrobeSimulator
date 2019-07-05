#ifndef BACTERIA_H
#define BACTERIA_H


#include <deal.II/base/point.h>
using dealii::Point;

#include <deal.II/dofs/dof_handler.h>
using dealii::DoFHandler;

#include <deal.II/lac/vector.h>
using dealii::Vector;

#include <deal.II/lac/block_vector.h>
using dealii::BlockVector; 

// @todo: implement a stokes handler -- or actualy, 
// have current advection class handle this -- 
// release any dependence of bacteria on deal.ii library...
// ``installation'' of bacteria simulator should be possible without deal ii if not using FEM

#include <deal.II/numerics/vector_tools.h>
// using dealii::VectorTools;

#include "single_bacterium.h"
#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "fitness.h"

#include "../discrete_field/FDM_chemical.h"

#include <iostream>
#include <string>
#include <sstream>
#include <array>

namespace MicrobeSimulator{

	template<int dim, int NumberChemicals>
	class Bacteria{
	public:
		// constructors:
		Bacteria();
		Bacteria(double db, unsigned int numberBacteria, 
			const std::array<double, NumberChemicals>& secretion_rates);

		// Bacteria(double db, unsigned int numberBacteria,
		// 	const std::vector<Point<dim> >& locations, 
		// 	double gs, double ws);

		Bacteria(const Bacteria& b);
	 
		// accessors:
		unsigned int getSize() const;
		Point<dim> getLocation(unsigned int i) const;

		std::array<double, NumberChemicals>
			 getSecretionRates(unsigned int bact) const;
		double getSecretionRate(unsigned int bact, unsigned int chem) const;

		// legacy:
		// double getGoodSecretionRate(unsigned int i) const;
		// double getWasteSecretionRate(unsigned int i) const;

		// mutators:
		// void initialize(double db, unsigned int numberBacteria, 
		// 	double gs, double ws, unsigned int number_cheaters = 0); // initialize at origin
		void initialize(double db, unsigned int numberBacteria,
			const std::array<double, NumberChemicals>& rates);

		// void initialize(double db, unsigned int numberBacteria,
		// 	const std::vector<Point<dim> >& locations, 
		// 	double gs, double ws, unsigned int number_cheaters = 0); 

		void initialize(double db, unsigned int numberBacteria,
			const std::array<double, NumberChemicals>& rates, 
			const std::vector<Point<dim> >& locations);

		// void setFitnessConstants(double a1, double a2, double k1, double k2, double b);

		// functions:
		void randomWalk(double timeStep, 
			const Geometry<dim>* const geometry, 
			const AdvectionField<dim>* const advection = NULL);

		void randomWalk(double timeStep,
			const Geometry<dim>* const geometry,
			const DoFHandler<dim>& stokes_dof,
			const BlockVector<double>& stokes_solution);

		// legacy:
		// void reproduce(double dt, const DoFHandler<dim>& dofh, 
		// 	const Vector<double>& sol1, const Vector<double>& sol2);

		// void reproduce(double dt, const FDMChemical<dim>& goods,
		// 	const FDMChemical<dim>& waste);

		void reproduce(double dt, const FitnessBase<dim, NumberChemicals>& fitness_function);

		void remove_fallen_bacteria(double right_edge);
		// void reproduce(Chemicals<dim,NumberGoods> chemicals, fitness function...); 
			// cost will have to be updated internally ... fitness -= costs...
		void mutate(double time_step, 
				double mutation_rate, 
				double deltaSecretion, 
				double original_secretion_rate = 0,
				bool binary_mutation = false);
		 // @todo: write another function --- what about for multiple public goods ? 


		bool isAlive() const;

		void print(std::ostream &out) const; 
		void printInfo(std::ostream &out) const; 

	private:
		double diffusionConstant;

		std::vector<SingleBacterium<dim, NumberChemicals> > bacteria;

		// legacy:
		// double getFitness( const  DoFHandler<dim>& dofh, 
		// 	const Vector<double>& sol1, const Vector<double>& sol2,
		// 	const Point<dim>& location, double sec_rate);

		// double getFitness(const FDMChemical<dim>& goods, const FDMChemical<dim>& waste,
		// 	const Point<dim>& location, double sec_rate);

		// switch to indpendent version:
		// double getFitness(const FitnessBase<dim,NumberChemicals>& fitness_function);
	}; // class Bacterium



// IMPLEMENTATION
//-----------------------------------------------------------------------------
	template<int dim, int NumberChemicals>
	Bacteria<dim, NumberChemicals>::Bacteria() 
	{}


	template<int dim, int NumberChemicals>
	Bacteria<dim, NumberChemicals>::Bacteria(double db, 
		unsigned int numberBacteria, 
		const std::array<double, NumberChemicals>& secretion_rates)
		:
		diffusionConstant(db),
		bacteria(numberBacteria)
	{
		for(unsigned int i = 0; i < numberBacteria; i++)
			bacteria[i].setSecretionRates(secretion_rates);
	}
	
	// template<int dim, int NumberChemicals>
	// Bacteria<dim, NumberChemicals>::Bacteria(double db, const std::vector<Point<dim> >& locations, 
	// 		double gs, double ws)
	// 	:
	// 	diffusionConstant(db),
	// 	bacteria(locations.size())
	// {
	// 	for(unsigned int i = 0; i < locations.size(); i++)
	// 	{
	// 		bacteria[i].setLocation(locations[i]);
	// 		bacteria[i].setGoodSecretionRate(gs);
	// 		bacteria[i].setWasteSecretionRate(ws);
	// 	}
	// }
		
	template<int dim, int NumberChemicals>
	Bacteria<dim, NumberChemicals>::Bacteria(const Bacteria& b)
	{
		diffusionConstant = b.diffusionConstant;
		bacteria = b.bacteria; // equal operation for vectors???
	}
	 

	// accessors: 
	template<int dim, int NumberChemicals>
	unsigned int 
	Bacteria<dim, NumberChemicals>::getSize() const
	{ 
		return bacteria.size();
	}


	template<int dim, int NumberChemicals>
	Point<dim> 
	Bacteria<dim, NumberChemicals>::getLocation(unsigned int i) const
	{ 
		return bacteria[i].getLocation(); 
	}


	template<int dim, int NumberChemicals>
	std::array<double, NumberChemicals>
	Bacteria<dim, NumberChemicals>::getSecretionRates(unsigned int bact) const
	{ 
		return bacteria[bact].getSecretionRates(); 
	}


	template<int dim, int NumberChemicals>
	double
	Bacteria<dim, NumberChemicals>::getSecretionRate(unsigned int bact,
			unsigned int chem) const
	{ 
		return bacteria[bact].getSecretionRate(chem); 
	}


	// mutators:

	// legacy:
	// template<int dim, int NumberChemicals>
	// void Bacteria<dim, NumberChemicals>::initialize(double db, unsigned int numberBacteria, 
	// 	double gs, double ws, unsigned int number_cheaters)
	// {
	// 	bacteria.clear();
	// 	bacteria.reserve(numberBacteria);

	// 	diffusionConstant = db;

	// 	unsigned int i = 0;
	// 	for( ; i < number_cheaters; i++)
	// 		bacteria.push_back( SingleBacterium<dim>(Point<dim>(), 0., ws) );

	// 	for( ; i < numberBacteria; i++)
	// 		bacteria.push_back( SingleBacterium<dim>(Point<dim>() , gs, ws) );
	// }


	// template<int dim, int NumberChemicals>
	// void Bacteria<dim, NumberChemicals>::initialize(double db, 
	// 	unsigned int numberBacteria,
	// 	const std::vector<Point<dim> >& locations, 
	// 	double gs, double ws, unsigned int number_cheaters)
	// {
	// 	bacteria.clear();
	// 	bacteria.reserve(numberBacteria);

	// 	diffusionConstant = db;

	// 	const unsigned int number_groups = locations.size();

	// 	unsigned int i = 0;
	// 	for( ; i < number_cheaters; i++)
	// 	{
	// 		unsigned int group_index = i % number_groups;	

	// 		bacteria.push_back( SingleBacterium<dim>(locations[group_index], 0., ws));
	// 	} 

	// 	for( ; i < numberBacteria; i++)
	// 	{
	// 		unsigned int group_index = i % number_groups;	

	// 		bacteria.push_back( SingleBacterium<dim>(locations[group_index], gs, ws));
	// 	} // for number of bacteria

	// }


	template<int dim, int NumberChemicals>
	void 
	Bacteria<dim, NumberChemicals>::initialize(double db, 
		unsigned int numberBacteria,
		const std::array<double, NumberChemicals>& rates)
	{
		bacteria.clear();
		bacteria.reserve(numberBacteria);

		diffusionConstant = db;

		for(unsigned int i = 0; i < numberBacteria; i++)
			bacteria.push_back( SingleBacterium<dim, NumberChemicals>(Point<dim>() , rates) );
	}


	template<int dim, int NumberChemicals>
	void 
	Bacteria<dim, NumberChemicals>::initialize(double db,
		unsigned int numberBacteria,
		const std::array<double, NumberChemicals>& rates, 
		const std::vector<Point<dim> >& locations)
	{
		bacteria.clear();
		bacteria.reserve(numberBacteria);

		diffusionConstant = db;

		const unsigned int number_groups = locations.size();

		for(unsigned int i=0; i < numberBacteria; i++)
		{
			unsigned int group_index = i % number_groups;	
			bacteria.push_back( SingleBacterium<dim, NumberChemicals>(locations[group_index], rates) );
		} // for number of bacteria

	}

	// template<int dim, int NumberChemicals>
	// void Bacteria<dim, NumberChemicals>::setFitnessConstants(double a1, double a2, 
	// 	double k1, double k2, double b)
	// {
	// 	alpha1 = a1;
	// 	alpha2 = a2;
	// 	saturation_const1 = k1;
	// 	saturation_const2 = k2;
	// 	beta1 = b;
	// }

	// functions:
	template<int dim, int NumberChemicals>
	void Bacteria<dim, NumberChemicals>::randomWalk(double timeStep, 
			const Geometry<dim>* const geometry, 
			const AdvectionField<dim>* const advection)
	{
		for(unsigned int i = 0; i < bacteria.size(); i++)
			bacteria[i].randomStep(timeStep, diffusionConstant, geometry, advection);

		// for open BC's in x-coordinate:
		if( geometry->getBoundaryConditions()[0] == BoundaryCondition::OPEN )
			remove_fallen_bacteria(geometry->getTopRightPoint()[0]);
	}


	template<int dim, int NumberChemicals>
	void 
	Bacteria<dim, NumberChemicals>::randomWalk(double timeStep,
			const Geometry<dim>* const geometry,
			const DoFHandler<dim>& stokes_dof,
			const BlockVector<double>& stokes_solution)
	{
		for(unsigned int i = 0; i < bacteria.size(); i++)
			bacteria[i].randomStep(timeStep, diffusionConstant, geometry,
				stokes_dof, stokes_solution);

		// for open BC's in x-coordinate:
		if( geometry->getBoundaryConditions()[0] == BoundaryCondition::OPEN )
			remove_fallen_bacteria(geometry->getTopRightPoint()[0]);
	}


	template<int dim, int NumberChemicals>
	void Bacteria<dim, NumberChemicals>::reproduce(double dt, 
		const FitnessBase<dim, NumberChemicals>& fitness_function)
		// const  DoFHandler<dim>& dofh, 
		// const Vector<double>& sol1, const Vector<double>& sol2)
	{
		unsigned int size = bacteria.size();
		std::vector<unsigned int> to_kill, to_rep;

		for(unsigned int i = 0; i < size; i++)
		{
			const double fit = dt*bacteria[i].getFitness(fitness_function); 
			// getFitness(dofh,sol1,sol2,
			// 	bacteria[i].getLocation(),bacteria[i].getGoodSecretionRate());
			if(fit < 0)
				to_kill.push_back(i);
			else
			{
				double prob =  ((double) rand() / (RAND_MAX));
				if(prob<fit)
					to_rep.push_back(i);
			}
		} // find those to kill or reproduce

		// go through cases:
		unsigned int num_to_kill = to_kill.size();
		unsigned int num_to_rep = to_rep.size();

		int new_size = size + num_to_rep - num_to_kill;

		if(num_to_kill <= num_to_rep)
		{
			// new size:
			bacteria.reserve(size + num_to_rep - num_to_kill);

			unsigned int i = 0;
			for( ; i < num_to_kill; i++)
		    	bacteria[ to_kill[i] ] = bacteria[ to_rep[i] ]; 
		    // replace dead with offspring

		    // push back remaining:
		    for(; i < num_to_rep; i++)
		      bacteria.push_back( bacteria[ to_rep[i] ] );
		}
		else if( new_size > 0 )
		{
		    unsigned int i = 0;
		    for( ; i < num_to_rep; i++)
		      bacteria[ to_kill[i] ] = bacteria[ to_rep[i] ];

		    // delete remaining: // *** could probably speed this part up
		    for(; i < num_to_kill; i++)
		      bacteria.erase( bacteria.begin() + to_kill[i] ); // could get reverse iterator to end...
		}
		else
		{
			bacteria.clear();
		} // empty vector

	} // later pass a fitness field object...


	// legacy:
	// template<int dim, int NumberChemicals>
	// void Bacteria<dim, NumberChemicals>::reproduce(double dt, const FDMChemical<dim>& goods,
	// 	const FDMChemical<dim>& waste)
	// {
	// 	unsigned int size = bacteria.size();
	// 	std::vector<unsigned int> to_kill, to_rep;

	// 	for(unsigned int i = 0; i < size; i++)
	// 	{
	// 		const double fit = dt*getFitness(goods, waste,
	// 			bacteria[i].getLocation(),bacteria[i].getGoodSecretionRate());
	// 		if(fit < 0)
	// 			to_kill.push_back(i);
	// 		else
	// 		{
	// 			double prob =  ((double) rand() / (RAND_MAX));
	// 			if(prob<fit)
	// 				to_rep.push_back(i);
	// 		}
	// 	} // find those to kill or reproduce

	// 	// go through cases:
	// 	unsigned int num_to_kill = to_kill.size();
	// 	unsigned int num_to_rep = to_rep.size();

	// 	int new_size = size + num_to_rep - num_to_kill;

	// 	if(num_to_kill <= num_to_rep)
	// 	{
	// 		// new size:
	// 		bacteria.reserve(size + num_to_rep - num_to_kill);

	// 		unsigned int i = 0;
	// 		for( ; i < num_to_kill; i++)
	// 	    	bacteria[ to_kill[i] ] = bacteria[ to_rep[i] ]; 
	// 	    // replace dead with offspring

	// 	    // push back remaining:
	// 	    for(; i < num_to_rep; i++)
	// 	      bacteria.push_back( bacteria[ to_rep[i] ] );
	// 	}
	// 	else if( new_size > 0 )
	// 	{
	// 	    unsigned int i = 0;
	// 	    for( ; i < num_to_rep; i++)
	// 	      bacteria[ to_kill[i] ] = bacteria[ to_rep[i] ];

	// 	    // delete remaining: // *** could probably speed this part up
	// 	    for(; i < num_to_kill; i++)
	// 	      bacteria.erase( bacteria.begin() + to_kill[i] ); // could get reverse iterator to end...
	// 	}
	// 	else
	// 	{
	// 		bacteria.clear();
	// 	} // empty vector

	// } // later pass a fitness field object...


	template<int dim, int NumberChemicals>
	void 
	Bacteria<dim, NumberChemicals>::remove_fallen_bacteria(double right_edge)
	{
		const double tolerance = 0.2;

		for(unsigned int i = 0; i < bacteria.size(); i++)
			if( std::fabs(bacteria[i].getLocation()[0] - right_edge) < tolerance )
				bacteria.erase( bacteria.begin() + i );
	}


	// legacy:
	// template<int dim, int NumberChemicals>
	// double Bacteria<dim, NumberChemicals>::getFitness(const  DoFHandler<dim>& dofh, 
	// 	const Vector<double>& sol1, const Vector<double>& sol2, 
	// 	const Point<dim>& location, double sec_rate)
	// {
	// 	double result = 0;
	// 	double c1, c2;

	// 	c1 = dealii::VectorTools::point_value(dofh, sol1, location);
	// 	c2 = dealii::VectorTools::point_value(dofh, sol2, location);

	// 	result = alpha1*(c1/(c1+saturation_const1)) 
	// 	      - alpha2*(c2/(c2+saturation_const2))
	// 		  - beta1*sec_rate;

	// 	// std::cout << "c1 = " << c1 << std::endl
	// 	// 	<< "c2 = " << c2 << std::endl << std::endl;

	// 	// std::cout << "fitness is: " << result << std::endl << std::endl;
	// 	return result;
	// }


	// template<int dim, int NumberChemicals>
	// double Bacteria<dim, NumberChemicals>::getFitness(const FDMChemical<dim>& goods, 
	// 	const FDMChemical<dim>& waste,
	// 	const Point<dim>& location, double sec_rate)
	// {
	// 	const double c1 = goods.value(location);
	// 	const double c2 = waste.value(location);

	// 	double result = alpha1*(c1/(c1+saturation_const1)) 
	// 		- alpha2*(c2/(c2+saturation_const2))
	// 		- beta1*sec_rate;

	// 	// std::cout << "Fitness constants are: "
	// 	// 	<< "\t alpha_1: " << alpha1 << std::endl
	// 	// 	<< "\t alpha 2: " << alpha2 << std::endl
	// 	// 	<< "\t beta: " << beta1 << std::endl 
	// 	// 	<< "\t s_good: " << sec_rate << std::endl
	// 	// 	<< "\t k1: " << saturation_const1 << std::endl
	// 	// 	<< "\t k2: " << saturation_const2 << std::endl;

	// 	// std::cout << "c1 = " << c1 << std::endl
	// 	// 	<< "c2 = " << c2 << std::endl << std::endl;

	// 	// std::cout << "fitness is: " << result << std::endl << std::endl;

	// 	return result;
	// }


	template<int dim, int NumberChemicals>
	void 
	Bacteria<dim, NumberChemicals>::mutate(double time_step, double mutation_rate, 
			double deltaSecretion, double original_secretion_rate, bool binary_mutation)
	{
		for(unsigned int i = 0; i < bacteria.size(); i++)
		{
			// random number between 0 and 1:
			double attempt = ((double) rand() / (RAND_MAX));

			if(attempt < time_step*mutation_rate)
				bacteria[i].mutate(deltaSecretion, original_secretion_rate, binary_mutation);
		}
	}

	template<int dim, int NumberChemicals>
	bool 
	Bacteria<dim, NumberChemicals>::isAlive() const
	{
		return !bacteria.empty();
	}



	template<int dim, int NumberChemicals>
	void Bacteria<dim, NumberChemicals>::print(std::ostream &out) const
	{
		for(unsigned int i = 0; i < bacteria.size(); i++)
		{
			bacteria[i].printBacterium(out);
			// out << bacteria[i].getLocation() << " "
			// 	<< bacteria[i].getGoodSecretionRate()
			// 	<< std::endl;
		}
		out << std::endl;
	} 


	template<int dim, int NumberChemicals>
	void
	Bacteria<dim, NumberChemicals>::printInfo(std::ostream& out) const
	{

	    out << "\n\n-----------------------------------------------------" << std::endl;
	    out << "\t\tBACTERIA INFO:";
	    out << "\n-----------------------------------------------------" << std::endl;

	    out << "\nNumber Bacteria: " << bacteria.size() << std::endl
	    	<< "\nDiffusion constant: " << diffusionConstant << std::endl
	    	<< "\nNumber chemicals: " << NumberChemicals << std::endl
	    	<< "\nIntial secretion rates: " << std::endl;
	    for(unsigned int j = 0; j < NumberChemicals; j++)
	    	out << "\t " << j << "th rate: " << bacteria[0].getSecretionRate(j) << std::endl;

	    	// << "\nGood secretion rate: " << bacteria[0].getGoodSecretionRate() 
	    	// << "\nWaste secretion rate: " << bacteria[0].getWasteSecretionRate() << std::endl
	    	// << "\nFITNESS: " << std::endl
	    	// << "\t public good benefit: " << alpha1 << std::endl
	    	// << "\t waste harm: " << alpha2 << std::endl
	    	// << "\t goods saturation constant: " << saturation_const1 << std::endl
	    	// << "\t waste saturation constant: " << saturation_const2 << std::endl
	    	// << "\t secretion cost: " << beta1 << std::endl;


        out << "\n-----------------------------------------------------\n\n" << std::endl;

	}
}

#endif  // bacterium.h
