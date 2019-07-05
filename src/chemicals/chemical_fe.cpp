#include "chemical_fe.h"

namespace MicrobeSimulator{
	template<int dim>
	ChemicalFE<dim>::ChemicalFE(double diff, double decay) 
		:
		ChemicalInterface<dim>(diff,decay)
	    {}


	template<int dim>
	ChemicalFE<dim>::~ChemicalFE() 
	    {}


	template<int dim>
	double ChemicalFE<dim>::value(const Point<dim> &p) const // override virtual
	{
		return 0;
	}



	template<int dim>
	void ChemicalFE<dim>::update(double dt, const Function<dim>& sourceFunction) // override virtual 
	{

	}


	template<int dim>
	void ChemicalFE<dim>::setup_system() 
	{
		// // maybe call FEM_Base setup ***if needed, have a flag that tells us its initialized...
	 //  if( !(base_ptr->isInitialized()) ) 
	 //  	base_ptr->setup_system();

	 //  system_matrix.reinit(base_ptr->getSparsityPattern() );
	 //  solution.reinit( base_ptr->getDofHandler().n_dofs() );
	 //  old_solution.reinit( base_ptr->getDofHandler().n_dofs() );
	 //  system_rhs.reinit( base_ptr->getDofHandler().n_dofs() );
	 //  tmp.reinit( base_ptr->getDofHandler().n_dofs() );
	}


	template<int dim>
	void ChemicalFE<dim>::printTest() const 
	{
		std::cout << "FINITE ELEMENT CHEMICAL CLASS! in " << dim << " dimensions!" << std::endl;
		std::cout << "diffusion const: " << ChemicalInterface<dim>::diffusion_constant 
			<< " decay: " << ChemicalInterface<dim>::decay_constant << std::endl;
	}
}

// chemical_fe.cc
