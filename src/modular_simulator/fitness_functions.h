#ifndef MICROBE_SIMULATOR_FITNESS_FUNCTIONS_H
#define MICROBE_SIMULATOR_FITNESS_FUNCTIONS_H

#include "../bacteria/fitness.h"
#include "./fe_chemical.h"

#include <array>

namespace MicrobeSimulator{ namespace FitnessFunctions{
	using namespace ModularSimulator;


/** A general OR type fitness function
* benefit is proportional to sum of (numchem-1) chemical values
* waste chemical is taken as last chemical in array
* secretion cost is given as same for first (numchem-1) rates
*/

template<int dim, int numchem>
class OR_Fitness : public FitnessBase<dim, numchem>
{
public:
	OR_Fitness() {}
	~OR_Fitness() {}

	virtual double value(const Point<dim>& location,
						const std::array<double, numchem>& secretion_rates) const;

	void attach_chemicals(std::array<FE_Chemical<dim>*, numchem> chem_ptrs);

	void setup_fitness_constants(
		double bene, double harm, double bsat, double hsat, double cost);

	void printInfo(std::ostream& out) const;

private:
	std::array<FE_Chemical<dim>*, numchem>		chemicals;

	double 		benefit_constant;
	double 		harm_constant;

	double		benefit_saturation;
	double 		harm_saturation;

	double 		secretion_cost;
};

// IMPLEMENTATION:
//---------------------------------------------------------------------------

template<int dim, int numchem>
void 
OR_Fitness<dim,numchem>::attach_chemicals(std::array<FE_Chemical<dim>*, numchem> chem_ptrs)
{
	chemicals = chem_ptrs;
}


template<int dim, int numchem>
void 
OR_Fitness<dim, numchem>::setup_fitness_constants(
	double bene, double harm, double bsat, double hsat, double cost)
{
	benefit_constant = bene;
	harm_constant = harm;

	benefit_saturation = bsat;
	harm_saturation = hsat;

	secretion_cost = cost;
}


template<int dim, int numchem>
void 
OR_Fitness<dim, numchem>::printInfo(std::ostream& out) const
{
  out << "\n\n-----------------------------------------------------" << std::endl
    << "\t\tOR FITNESS FUNCTION (for " << numchem << " chemicals)" << std::endl
    << "-----------------------------------------------------" << std::endl
    << "\t public good benefit: " << benefit_constant << std::endl
    << "\t waste harm: " << harm_constant << std::endl
    << "\t public good saturation: " << benefit_saturation << std::endl
    << "\t waste saturation: " << harm_saturation << std::endl
    << "\t secretion cost: " << secretion_cost << std::endl;
}


template<int dim, int numchem>
double 
OR_Fitness<dim, numchem>::value(const Point<dim>& location, 
        const std::array<double, numchem>& secretion_rates) const
{
	double total_goods = 0.;
	double total_secretion_rate = 0.;

	for(unsigned int i = 0; i < numchem-1; ++i)
	{
		total_goods += chemicals[i]->value(location);
		total_secretion_rate += secretion_rates[i];
	} // for all public goods
	const double waste = chemicals[numchem-1]->value(location);

	const double return_value =
		benefit_constant * total_goods / ( total_goods + benefit_saturation )
		- harm_constant * waste / ( waste + harm_saturation )
		- secretion_cost * total_secretion_rate; 

  return return_value;
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------







/** A general AND type fitness function
* benefit is proportional to product of (numchem-1) chemical values
* waste chemical is taken as last chemical in array
* secretion cost is given as same for first (numchem-1) rates
*/

template<int dim, int numchem>
class AND_Fitness : public FitnessBase<dim, numchem>
{
public:
	AND_Fitness() {}
	~AND_Fitness() {}

	virtual double value(const Point<dim>& location,
						const std::array<double, numchem>& secretion_rates) const;

	void attach_chemicals(std::array<FE_Chemical<dim>*, numchem> chem_ptrs);

	void setup_fitness_constants(
		double bene, double harm, double bsat, double hsat, double cost);

	void printInfo(std::ostream& out) const;

private:
	std::array<FE_Chemical<dim>*, numchem>		chemicals;

	double 		benefit_constant;
	double 		harm_constant;

	double		benefit_saturation;
	double 		harm_saturation;

	double 		secretion_cost;
};

// IMPLEMENTATION:
//---------------------------------------------------------------------------

template<int dim, int numchem>
void 
AND_Fitness<dim,numchem>::attach_chemicals(std::array<FE_Chemical<dim>*, numchem> chem_ptrs)
{
	chemicals = chem_ptrs;
}


template<int dim, int numchem>
void 
AND_Fitness<dim, numchem>::setup_fitness_constants(
	double bene, double harm, double bsat, double hsat, double cost)
{
	benefit_constant = bene;
	harm_constant = harm;

	benefit_saturation = bsat;
	harm_saturation = hsat;

	secretion_cost = cost;
}


template<int dim, int numchem>
void 
AND_Fitness<dim, numchem>::printInfo(std::ostream& out) const
{
  out << "\n\n-----------------------------------------------------" << std::endl
    << "\t\tOR FITNESS FUNCTION (for " << numchem << " chemicals)" << std::endl
    << "-----------------------------------------------------" << std::endl
    << "\t public good benefit: " << benefit_constant << std::endl
    << "\t waste harm: " << harm_constant << std::endl
    << "\t public good saturation: " << benefit_saturation << std::endl
    << "\t waste saturation: " << harm_saturation << std::endl
    << "\t secretion cost: " << secretion_cost << std::endl;
}


template<int dim, int numchem>
double 
AND_Fitness<dim, numchem>::value(const Point<dim>& location, 
        const std::array<double, numchem>& secretion_rates) const
{
	double total_goods = 1.;
	double total_secretion_rate = 0.;

	for(unsigned int i = 0; i < numchem-1; ++i)
	{
		total_goods *= chemicals[i]->value(location);
		total_secretion_rate += secretion_rates[i];
	} // for all public goods
	const double waste = chemicals[numchem-1]->value(location);

	const double return_value =
		benefit_constant * total_goods / ( total_goods + benefit_saturation )
		- harm_constant * waste / ( waste + harm_saturation )
		- secretion_cost * total_secretion_rate; 

  return return_value;
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

/** SUM Fitness ...
* @ todo add a sum fitness function
*/



}} // close namespaces

#endif

