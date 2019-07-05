#ifndef MICROBE_SIMULATOR_CHEMICAL_FINITE_DIFFERENCE_H
#define MICROBE_SIMULATOR_CHEMICAL_FINITE_DIFFERENCE_H

#include <vector>
#include "chemical_interface.h"

namespace MicrobeSimulator{

	template<int dim>
	class Chemical_FDM : public ChemicalInterface<dim>
	{
	public:
		// ChemicalInterface();
		Chemical_FDM(double diff, double decay, double dt);

		// interface functions: (override)
		virtual double value(const Point<dim>& p) const;  
		virtual void secreteOntoField(const Point<dim>& p, double secretionRate);
		virtual void update();

		virtual void printInfo() const; 

	private:
		unsigned int number_elements[dim];
		double inverseInterval[dim];

		std::vector<double> field;
		std::vector<double> diffusion_field;
		std::vector<double> advection_field;

		void updateDiffusion();
		void updateAdvection();

		unsigned int indexFromPoint(const Point<dim>& p) const; 
		// can set to work with ghost nodes! -- or create your own accessor
			// unsigned int oneDimINdex(i,j) ...

	}; // class Chemical_Core


	// template<int dim>
	// ChemicalInterface<dim>::ChemicalInterface()
	// 	:
	// 	diffusion_constant(0),
	// 	decay_constant(0)
	// 	{}

	template<int dim>
	Chemical_FDM<dim>::Chemical_FDM(double diff, double decay, double dt)
		:
		ChemicalInterface(diff,decay,dt)
		{}

	template<int dim>
	void Chemical_FDM<dim>::value(const Point<dim>& p) const
	{
		return field[ indexFromPoint(p) ];
	}

	template<int dim>
	void Chemical_FDM<dim>::secreteOntoField(const Point<dim>& p, 
		double secretionRate)
	{
		unsigned int i = indexFromPoint(p);
		field[i] += secretionRate*ChemicalInterface<dim>::time_step;
	}


	template<int dim>
	void Chemical_FDM<dim>::update()
	{
		updateDiffusion();
		updateAdvection();

		for(unsigned int i = 0; i < size; i++)
			field[i] += diffusion_field[i] + advection_field[i]; 
	}

	template<int dim>
	void Chemical_FDM<dim>::updateDiffusion()
	{

	}

	template<int dim>
	void Chemical_FDM<dim>::updateAdvection()
	{
		
	}

	template<int dim>
	Chemical_FDM<dim>::printInfo() const
	{
		std::cout << "Chemical: \n\t Diffusion Constant: " 
			<< ChemicalInterface<dim>::diffusion_constant
			<< "\n\t Decay Constant: "
			<< ChemicalInterface<dim>::diffusion_constant 
			<< "\n\t Time Step: "
			<< ChemicalInterface<dim>::time_step << std::endl;
	}



}

#endif // chemical_core.h
