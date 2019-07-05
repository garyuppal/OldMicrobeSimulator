#ifndef MICROBE_SIMULATOR_CHEMICAL_FE_BASE_H
#define MICROBE_SIMULATOR_CHEMICAL_FE_BASE_H


/** necessary header files...
* @todo clean up, remove those not used...
*/

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
// can probably remove cg...
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>
#include <sstream>
#include <limits>

#include "./stokes_handler.h"


namespace MicrobeSimulator{ namespace ModularSimulator{
	using namespace dealii;


	/// split into a base class handling  common aspects and individual chemcals?? ...
	/// other option is to template class with num_chem and do everything here...
	/// split option might be useful if seeking implementation independent access...

template<int dim>
class Chemical_FE_Base{  
public:
	Chemical_FE_Base(const Triangulation<dim>& tria,
					const unsigned int degree = 1);

	void setup_chemical_base();
	void setup_chemical_base(const StokesHandler<dim>& stokes_solution);

	/// ACCESSORS:
	const SparsityPattern& getSparsityPattern() const;
	unsigned int get_n_dofs() const;
	const SparseMatrix<double>& get_mass_matrix() const;
	const SparseMatrix<double>& get_stiffness_matrix() const;
	unsigned int get_fe_degree() const;
	const FE_Q<dim>& get_fe() const;
	const DoFHandler<dim>& get_dof_handler() const;
	const ConstraintMatrix& get_constraints() const;

	// velocity values:
	std::vector<Tensor<1, dim> >	
	get_cell_velocity_values(unsigned int cell_id) const;

	Tensor<1, dim>		
	get_cell_quad_velocity_value(unsigned int cell_id, unsigned int q_point) const;

private:
	// common to all chemicals:

	const Triangulation<dim>*		triangulation;
	const unsigned int      		chemical_fe_degree;
	FE_Q<dim>               		chemical_fe;
	DoFHandler<dim>         		chemical_dof_handler;

	ConstraintMatrix        		chemical_constraints;
	SparsityPattern         		chemical_sparsity_pattern;

	SparseMatrix<double>    		chemical_mass_matrix;
	SparseMatrix<double>    		chemical_diffusion_matrix;

	// velocity:
	std::vector<std::vector<Tensor<1, dim> > >	velocity_values; //(n_q_points);
	bool using_stokes_velocity;

	// initialization functions
	void setup_dofs();
	void assemble_chemical_matrices();
	void setup_velocity_values(const StokesHandler<dim>& stokes_solution);
};


/** IMPLEMENTATION:
*/
//-------------------------------------------------------------------------
template<int dim>
Chemical_FE_Base<dim>::Chemical_FE_Base(const Triangulation<dim>& tria,
										const unsigned int degree)
	:
	triangulation(&tria),
	chemical_fe_degree(degree),
	chemical_fe( chemical_fe_degree ),
	chemical_dof_handler(*triangulation),
	using_stokes_velocity(false)
{}


template<int dim>
void 
Chemical_FE_Base<dim>::setup_dofs()
{
	{
		chemical_dof_handler.distribute_dofs(chemical_fe);
		chemical_constraints.clear();
		DoFTools::make_hanging_node_constraints(chemical_dof_handler,
		                                	    chemical_constraints);
		chemical_constraints.close();
	}
	{
		DynamicSparsityPattern dsp(chemical_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(chemical_dof_handler,
			                            dsp,
			                            chemical_constraints,
			/*keep constrained_dofs = */ false); 
		chemical_sparsity_pattern.copy_from(dsp);
	}

	chemical_mass_matrix.reinit(chemical_sparsity_pattern);
	chemical_diffusion_matrix.reinit(chemical_sparsity_pattern);
}


template<int dim>
void 
Chemical_FE_Base<dim>::assemble_chemical_matrices()
{
	std::cout << "...assembling chemical matrices" << std::endl;
	chemical_mass_matrix = 0;
	chemical_diffusion_matrix = 0;

	QGauss<dim> quadrature_formula(chemical_fe_degree + 2);
	FEValues<dim> fe_values(chemical_fe, quadrature_formula,
	      update_values | update_gradients | update_JxW_values);

	const unsigned int dofs_per_cell = chemical_fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
	FullMatrix<double> local_diffusion_matrix(dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	std::vector<double>           phi_T (dofs_per_cell);
	std::vector<Tensor<1, dim> >  grad_phi_T(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
		cell = chemical_dof_handler.begin_active(),
		endc = chemical_dof_handler.end();

	for(; cell != endc; ++cell)
	{
		local_mass_matrix = 0;
		local_diffusion_matrix = 0;

		fe_values.reinit(cell);

		for(unsigned int q = 0; q < n_q_points; ++q)
		{
			for(unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_T[k] = fe_values.shape_grad(k,q);
				phi_T[k] = fe_values.shape_value(k,q);
			}

			for(unsigned int i = 0; i < dofs_per_cell; ++i)
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
				{
				local_mass_matrix(i,j) += (phi_T[i] * phi_T[j]
				*fe_values.JxW(q));

				local_diffusion_matrix(i,j) += (grad_phi_T[i] * grad_phi_T[j]
				*fe_values.JxW(q));
				}
		} // for quadrature points

		cell->get_dof_indices(local_dof_indices);
		chemical_constraints.distribute_local_to_global(local_mass_matrix,
						                                local_dof_indices,
						                                chemical_mass_matrix);
		chemical_constraints.distribute_local_to_global(local_diffusion_matrix,
						                                local_dof_indices,
						                                chemical_diffusion_matrix);
	} // for cells
}


/** setup vector velocity_values for adding advection to chemicals
*/

template<int dim>
void 
Chemical_FE_Base<dim>::setup_velocity_values(const StokesHandler<dim>& stokes_solution)
{
	const QGauss<dim> quadrature_formula(chemical_fe_degree + 2);

	FEValues<dim> stokes_fe_values(stokes_solution.get_fe(),
									quadrature_formula,
									update_values);

	// const unsigned int dofs_per_cell = chemical_fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	std::vector<Tensor<1, dim> >		cell_velocity_values(n_q_points);
	velocity_values.reserve(chemical_dof_handler.n_dofs()); // or is it number of active cells?

	const FEValuesExtractors::Vector velocities(0);

	//typename DoFHandler<dim>::active_cell_iterator 
	auto cell         = chemical_dof_handler.begin_active();
	const auto endc   = chemical_dof_handler.end();
	auto stokes_cell  = stokes_solution.get_dof_handler().begin_active();
	unsigned int cell_id = 0; 

	for(; cell!=endc; ++cell, ++stokes_cell, ++cell_id)
	{
	    stokes_fe_values.reinit(stokes_cell);

		stokes_fe_values[velocities].get_function_values(
											stokes_solution.get_solution(), 
											cell_velocity_values);

		velocity_values.push_back(cell_velocity_values);
	} // for cells

	using_stokes_velocity = true;
}


template<int dim>
void
Chemical_FE_Base<dim>::setup_chemical_base()
{
	setup_dofs();
	assemble_chemical_matrices();
}


template<int dim>
void
Chemical_FE_Base<dim>::setup_chemical_base(const StokesHandler<dim>& stokes_solution)
{
	setup_chemical_base();
	setup_velocity_values(stokes_solution);
}


/* ACCESSORS:
* these will be needed to let chemical classes make use of the base class
*/

template<int dim>
const SparsityPattern& 
Chemical_FE_Base<dim>::getSparsityPattern() const
{
	return chemical_sparsity_pattern;
}


template<int dim>
unsigned int 
Chemical_FE_Base<dim>::get_n_dofs() const
{
	return chemical_dof_handler.n_dofs();
}


template<int dim>
const SparseMatrix<double>& 
Chemical_FE_Base<dim>::get_mass_matrix() const
{
	return chemical_mass_matrix;
}


template<int dim>
const SparseMatrix<double>& 
Chemical_FE_Base<dim>::get_stiffness_matrix() const
{
	return chemical_diffusion_matrix;
}


template<int dim>
unsigned int 
Chemical_FE_Base<dim>::get_fe_degree() const
{
	return chemical_fe_degree;
}


template<int dim>
const FE_Q<dim>& 
Chemical_FE_Base<dim>::get_fe() const
{
	return chemical_fe;
}


template<int dim>
const DoFHandler<dim>& 
Chemical_FE_Base<dim>::get_dof_handler() const
{
	return chemical_dof_handler;
}


template<int dim>
const ConstraintMatrix& 
Chemical_FE_Base<dim>::get_constraints() const
{
	return chemical_constraints;
}

// velocity values:
template<int dim>
std::vector<Tensor<1, dim> >	
Chemical_FE_Base<dim>::get_cell_velocity_values(unsigned int cell_id) const
{
	if(using_stokes_velocity == false)
		return std::vector<Tensor<1,dim> >{Tensor<1, dim>()};

	return velocity_values[cell_id]; 
}


template<int dim>
Tensor<1, dim>		
Chemical_FE_Base<dim>::get_cell_quad_velocity_value(unsigned int cell_id, 
													unsigned int q_point) const
{
	if(using_stokes_velocity == false)
		return Tensor<1,dim>(); // zero by default

	return velocity_values[cell_id][q_point]; 
}




}} // close namespace

#endif
