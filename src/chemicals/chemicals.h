#ifndef MICROBE_SIMULATOR_CHEMICALS_H
#define MICROBE_SIMULATOR_CHEMICALS_H

#include <vector>
#include <exception>

#include "chemical_fdm.h"
#include "../utility/argparser.h"
#include "../sources/sources.h"

namespace MicrobeSimulator{

	template<int dim, int NumberChemicals>
	class Chemicals{
	public:
		Chemicals(ArgParser<dim> parameters); 
		~Chemicals(); 

		void update();

		double value(unsigned int chemical_number, 
			const Point<dim>& location) const;

		void outputResults() const;
		void printInfo() const;
	private:
		ChemicalInterface<dim>** chemicals; 
		Sources<dim, NumberChemicals> sources;
		// FEM_Base femBase; // multiple inheritance??? DG_Base? // common to all chemicals // Base_interface???
	};




	template<int dim, int NumberChemicals>
	Chemicals<dim,NumberChemicals>::Chemicals(ArgParser<dim> parameters)
	{
	    chemicals = new ChemicalInterface<dim>*[number_chemicals];

	    // parameters.getImplementationType() : FE, DG, FD
	    for(unsigned int i=0; i < number_chemicals; i++)
	      chemicals[i] = new ChemicalFDM<dim>(parameters.getDiffusionConstant(i), 
	      	parameters.getDecayConstant(i),parameters.getTimeStep()); 
	}


	template<int dim, int NumberChemicals>
	Chemicals<dim,NumberChemicals>::~Chemicals()
	{
	    for(unsigned int i=0; i< number_chemicals; i++)
	      delete chemicals[i];
	    delete[] chemicals;
	}


	template<int dim, int NumberChemicals>
	void Chemicals<dim,NumberChemicals>::update()
	{
		for(unsigned int i=0; i < NumberChemicals; i++)
		{
			for(unsigned int bact=0; bact < sources.numberBacteria(); bact++)
			{
				chemicals[i]->secreteOntoField(sources.location(bact),
					sources.secretionRate(bact));
			}
			chemicals[i]->update();
		}
	}


	template<int dim, int NumberChemicals>
	double Chemicals<dim,NumberChemicals>::value(unsigned int chemical_number, 
		const Point<dim>& location) const
	{
		return chemicals[chemical_number]->value(location);
	}


	template<int dim, int NumberChemicals>
	void Chemicals<dim,NumberChemicals>::outputResults() const
	{
		// NEED TO IMPLEMENT -- maybe also make virtual
	}


	template<int dim, int NumberChemicals>
	void Chemicals<dim,NumberChemicals>::printInfo() const
	{
		std::cout << "\n THERE ARE " << number_chemicals 
			<< " CHEMICALS***\n" << std::endl;

		for(unsigned int i=0; i < number_chemicals; i++)
			chemicals[i]->printInfo();
	}

}


#endif // chemical.h
