#ifndef MICROBE_SIMULATOR_CHEMICAL_FINITE_ELEMENT_H
#define MICROBE_SIMULATOR_CHEMICAL_FINITE_ELEMENT_H

#include <deal.II/lac/vector.h>
using dealii::Vector;

#include <deal.II/lac/sparse_matrix.h>
using dealii::SparseMatrix;


#include "chemical_interface.h"
// #include "fem_base.h"


namespace MicrobeSimulator{

	template<int dim>
	class ChemicalFE : public ChemicalInterface<dim>
	{
	public:
		ChemicalFE(double diff, double decay); 
		~ChemicalFE();
		// ChemicalFE(FEM_Base* base);

		double value(const Point<dim>& p) const override; 
		// void secreteTo(const Bacteria& bact, double dt); // override;
		void update(double dt, const Function<dim>& sourceFunction) override;


	    void printTest() const override;

	private:
		// FEM_Base* base_ptr;

		SparseMatrix<double> system_matrix;

	    Vector<double>       solution;
	    Vector<double>       old_solution;
	    Vector<double>       system_rhs;


	    void setup_system(); // maybe call FEM_Base setup ***if needed, have a flag that tells us its initialized...

	    // Vector<double>       tmp; // to store rhs

	 //    double               time; // t
	 //    double               time_step; // dt
	 //    unsigned int         timestep_number; // n

	 //    // chemical parameters:
	 //    double diffusion_constant;
		// double decay_constant;

	    // const double         theta; // can use crank-nicolson...
	}; // class Chemical_FE

}

#include "chemical_fe.cpp"

#endif // chemical_fe.h
