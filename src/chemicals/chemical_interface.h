#ifndef MICROBE_SIMULATOR_CHEMICAL_INTERFACE_H
#define MICROBE_SIMULATOR_CHEMICAL_INTERFACE_H

#include <deal.II/base/point.h>
using dealii::Point;



namespace MicrobeSimulator{

	template<int dim>
	class ChemicalInterface
	{
	public:
		// ChemicalInterface();
		ChemicalInterface(double diff, double decay, double dt);
		virtual ~ChemicalInterface() {} 

		// interface functions:
		virtual double value(const Point<dim>& p) const = 0;  
		virtual void secreteOntoField(const Point<dim>& p, double secretionRate) = 0;
		virtual void update() = 0;

		// accessors:
		double getDiffusionConstant() const;
		double getDecayConstant() const;
		double getTimeStep() const;

		// mutators:
		void setDiffusionConstant(double diff);
		void setDecayConstant(double decay);
		void setTimeStep(double dt);

		virtual void printInfo() const =0; // for output, testing, and debugging

	protected:
		double diffusion_constant;
		double decay_constant;
		double time_step;

	}; // class Chemical_Core


	// template<int dim>
	// ChemicalInterface<dim>::ChemicalInterface()
	// 	:
	// 	diffusion_constant(0),
	// 	decay_constant(0)
	// 	{}

	template<int dim>
	ChemicalInterface<dim>::ChemicalInterface(double diff, double decay, double dt)
		:
		diffusion_constant(diff),
		decay_constant(decay),
		time_step(dt)
		{}

	template<int dim>
	double ChemicalInterface<dim>::getDiffusionConstant() const
	{
		return diffusion_constant;
	}

	template<int dim>
	double ChemicalInterface<dim>::getDecayConstant() const
	{
		return decay_constant;
	}

	template<int dim>
	double ChemicalInterface<dim>::getTimeStep() const
	{
		return time_step;
	}

	template<int dim>
	void ChemicalInterface<dim>::setDiffusionConstant(double diff)
	{
		diffusion_constant = diff;
	}

	template<int dim>
	void ChemicalInterface<dim>::setDecayConstant(double decay)
	{
		decay_constant = decay;
	}

	template<int dim>
	void ChemicalInterface<dim>::setTimeStep(double dt)
	{
		time_step = dt;
	}
}

#endif // chemical_core.h
