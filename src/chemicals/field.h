#ifndef MICROBE_SIMULATOR_FIELD_H
#define MICROBE_SIMULATOR_FIELD_H

#include <deal.II/base/point.h>
using dealii::Point;

#include <cmath>
#include <vector>
#include <array>
#include <iostream>

#include "../advection/advection.h"

namespace MicrobeSimulator{
	
	template<int dim>
	class Field{
	private:
		double diffusion_constant;
		double decay_constant;
		std::vector<double> field;
		std::vector<double> temp; // for storing diffusion and advection terms

		Point<dim> bottom_left;
		Point<dim> top_right;
		std::array<unsigned int, dim> discretization;

		// part of evolution:
		void intializeFieldVectors();
		void updateDiffusion();
		void updateAdvection(const AdvectionField<dim>* const advection); // put these together?
		unsigned int indexFromPoint(const Point<dim>& p);

		unsigned int index(int i, int j) const;
		unsigned int index(int i, int j, int k) const; // include boundary conditions

	public:
		Field();
		Field(double diff, double decay, 
			const Point<dim>& lower, const Point<dim>& upper,
			const std::array<unsigned int, dim>& disc);
		Field(const Field& f);

		// accessors:
		double value(const Point<dim>& p) const;
		double getDiffusionConstant() const;
		double getDecayConstant() const;

		// mutators:
		void setDiffusionConstant(double diff);
		void setDecayConstant(double decay);
		void setBounds(const Point<dim>& lower, const Point<dim>& upper);
		void setDiscretization(const std::array<unsigned int, dim>& disc);		

		// evolve:
		void secreteTo(const Point<dim>& p, double amount);
		void update(double time_step, const AdvectionField<dim>* const advection = NULL);

		void print(std::ostream& out) const;
	};


//IMPLEMENTATION
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
	template<int dim>
	Field<dim>::Field()
		:
		diffusion_constant(0),
		decay_constant(0)
	{}

	template<int dim>
	Field<dim>::Field(double diff, double decay, 
			const Point<dim>& lower, const Point<dim>& upper,
			const std::array<unsigned int, dim>& disc)
		:
		diffusion_constant(diff),
		decay_constant(decay),
		bottom_left(lower),
		top_right(upper),
		discretization(disc)
	{
		intializeFieldVectors();
	}

	template<int dim>
	Field<dim>::Field(const Field& f)
	{
		diffusion_constant = f.diffusion_constant;
		decay_constant = f.decay_constant;
		field = f.field;
		temp = f.temp;
		bottom_left = f.bottom_left;
		top_right = f.top_right;
		discretization = f.discretization;
	}

	template<int dim>
	void Field<dim>::intializeFieldVectors()
	{
		unsigned int size = 1;
		for(unsigned int i = 0; i < dim; i++)
			size *= discretization[i];

		field.assign(size,0);
		temp.assign(size,0);
	}


	template<int dim>
	unsigned int Field<dim>::index(int i, int j) const
	{
		// if(inBounds(i,j)) // or assign new i j k 
			return i + j*discretization[0]; 
		// else
		// 	return 0;
	}

	template<int dim>	
	unsigned int Field<dim>::index(int i, int j, int k) const
	{
		return i + j*discretization[0] + k*discretization[0]*discretization[1]; 
	}

	template<int dim>
	void Field<dim>::updateDiffusion()
	{
		if(dim == 2)
		{
			const double invDXsqr = std::pow((discretization[0] - 1)/(top_right(0) - bottom_left(0)),2);
			const double invDYsqr = std::pow((discretization[1] - 1)/(top_right(1) - bottom_left(1)),2);

			for(int i = 0; i < discretization[0]; i++)
				for(int j = 0; j < discretization[1]; j++)
					temp[index(i,j)] = diffusion_constant*(
						invDXsqr*(field[index(i+1,j)] - 2*field[index(i,j)] + field[index(i-1,j)])
						+ invDYsqr*(field[index(i,j+1)] - 2*field[index(i,j)] + field[index(i-1,j)])
						);
		}

		else if(dim == 3)
		{
			const double invDXsqr = std::pow((discretization[0] - 1)/(top_right(0) - bottom_left(0)),2);
			const double invDYsqr = std::pow((discretization[1] - 1)/(top_right(1) - bottom_left(1)),2);			
			const double invDZsqr = std::pow((discretization[2] - 1)/(top_right(2) - bottom_left(2)),2);
			
			
			for(int i = 0; i < discretization[0]; i++)
				for(int j = 0; j < discretization[1]; j++)
					for(int k = 0; k < discretization[2]; k++)
						temp[index(i,j,k)] = diffusion_constant*(
							invDXsqr*(field[index(i+1,j,k)] 
								- 2*field[index(i,j,k)] + field[index(i-1,j,k)])
							+ invDYsqr*(field[index(i,j+1,k)] 
								- 2*field[index(i,j,k)] + field[index(i-1,j,k)])
							+ invDZsqr*(field[index(i,j,k+1)] 
								- 2*field[index(i,j,k)] + field[index(i,j,k-1)])
							);
		}
		else
		{
			throw std::runtime_error("Field diffusion not implemented for desired dimension");
		}
	}

	template<int dim>
	void Field<dim>::updateAdvection(const AdvectionField<dim>* const advection)
	{
		// 1st order upwind:  // *** FINISH IMPLEMENTING
		if(dim == 2)
		{
			// Point<dim> velocity;
			// velocity = advection->value(location(i,j)); // index to point ...
			std::cout << "advection for dim == 2 not yet implemented" << std::endl;
		}
		else if(dim == 3)
		{
			std::cout << "advection for dim == 3 not yet implemented" << std::endl;
		}
		else
		{
			throw std::runtime_error("Field advection not implemented for desired dimension");
		}


	} // needs a velocity function ...

	template<int dim>
	unsigned int Field<dim>::indexFromPoint(const Point<dim>& p)
	{
		return 0; // *** FINISH IMPLEMENTING
	}

	template<int dim>
	double Field<dim>::value(const Point<dim>& p) const
	{
		return field[indexFromPoint(p)];
	}

	template<int dim>
	double Field<dim>::getDiffusionConstant() const
	{return diffusion_constant;}

	template<int dim>
	double Field<dim>::getDecayConstant() const
	{return decay_constant;}

	template<int dim>
	void Field<dim>::setDiffusionConstant(double diff)
	{diffusion_constant = diff;}

	template<int dim>
	void Field<dim>::setDecayConstant(double decay)
	{decay_constant = decay;}

	template<int dim>
	void Field<dim>::setBounds(const Point<dim>& lower, const Point<dim>& upper)
	{
		bottom_left = lower;
		top_right = upper;
	}

	template<int dim>
	void Field<dim>::setDiscretization(const std::array<unsigned int, dim>& disc)
	{
    	discretization = disc; 
    	intializeFieldVectors(); // reinitialize field
	}

	// evolve:
	template<int dim>
	void Field<dim>::secreteTo(const Point<dim>& p, double amount)
	{
		field[indexFromPoint(p)] = field[indexFromPoint(p)] + amount;
	}

	template<int dim>
	void Field<dim>::update(double time_step, const AdvectionField<dim>* const advection)
	{
		updateDiffusion(); // updates temp

		if(advection != NULL)
			updateAdvection(advection); // add to diffusion -> temp = diff + adv

		// field = field + dt*(temp - decay_constant*field);
		for(unsigned int i = 0; i < field.size(); i++)
			field[i] = field[i] + time_step*(
				temp[i] - decay_constant*field[i]);
	}

	template<int dim>
	void Field<dim>::print(std::ostream& out) const
	{
		if(dim == 2)
		{
			for(int j = 0; j < discretization[1]; j++)
			{
				for(int i = 0; i < discretization[0]; i++)
				{
					out << field.at(index(i,j)) << " ";
				}
				out << std::endl;
			}
		}
		else if(dim == 3)
		{
			for(int k = 0; k < discretization[2]; k++)
			{
				for(int j = 0; j < discretization[1]; j++)
				{
					for(int i = 0; i < discretization[0]; i++)
					{
						out << field.at(index(i,j,k)) << " ";
					}
					out << std::endl;
				}
				out << std::endl;
			} // for
		} // if dim == 3
	} // print

} // namespace MicrobeSimulator


#endif
