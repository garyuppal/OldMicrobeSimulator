#ifndef MICROBE_SIMULATOR_FE_CHEMICAL_H
#define MICROBE_SIMULATOR_FE_CHEMICAL_H


#include "../simulator/exact_solutions.h"

#include "./chemical_fe_base.h"

#include "../geometry/geometry.h"
#include "../simulator/cell_iterator_map.h"

namespace MicrobeSimulator{ namespace ModularSimulator{
	using namespace dealii;

template<int dim>
class FE_Chemical{
public:	
	FE_Chemical();
	FE_Chemical(const Chemical_FE_Base<dim>& chem_base);
	FE_Chemical(const Chemical_FE_Base<dim>& chem_base,
				const double diffusion,
				const double decay);

	// accessors:
	double getDiffusionConstant() const;
	double getDecayConstant() const;
	double value(const Point<dim>& location) const;

	// mutators:
	void setDiffusionConstant(double diffusion);
	void setDecayConstant(double decay);

	// methods:
	void reinit();
	void reinit(const Chemical_FE_Base<dim>& chem_base);

	void project_initial_condition(const Function<dim> &initial_condition);

	void update(const double time_step);  // also need to include sources...
	void update(const double time_step, 
				const PointCellMap<dim>& mapping,
				const Geometry<dim>& geometry,
				const std::vector<Point<dim> >& locations, 
				const std::vector<double>& amounts);

	void output_solution(const std::string& output_directory,
						 const unsigned int chem_id,
						 const unsigned int save_step) const;

private:
	const Chemical_FE_Base<dim>*	chemical_base;

	double diffusion_constant;
	double decay_constant;

	// chemical objects:
	SparseMatrix<double>			system_matrix;
	Vector<double>					solution;
	Vector<double>					old_solution;
	Vector<double>					old_old_solution; // for second order time schemes

	Vector<double> 					right_hand_side; // see if we can get away with just this...
	Vector<double>					source; // can probably do away with this , but use for now

	bool use_bdf2_scheme;

    double compute_viscosity(const std::vector<Tensor<1,dim> >& velocity_values,
                const double cell_diameter);
	void update_system_matrix(const double time_step, const bool using_sources); // need to include sources ...
	void update_source_from_map(const PointCellMap<dim>& mapping,
								const Geometry<dim>& geometry,
								const std::vector<Point<dim> >& locations, 
								const std::vector<double>& amounts);
	void solve();
};


/** IMPLEMENTATION
*/
//-----------------------------------------------------------------------------
template<int dim>
FE_Chemical<dim>::FE_Chemical()
	:
	chemical_base(NULL),
	use_bdf2_scheme(false)
{}


template<int dim>
FE_Chemical<dim>::FE_Chemical(const Chemical_FE_Base<dim>& chem_base)
	:
	chemical_base(&chem_base),
	use_bdf2_scheme(false)
{}


template<int dim>
FE_Chemical<dim>::FE_Chemical(const Chemical_FE_Base<dim>& chem_base,
							const double diffusion,
							const double decay)
	:
	chemical_base(&chem_base),
	diffusion_constant(diffusion),
	decay_constant(decay),
	use_bdf2_scheme(false)
{}


template<int dim>
double 
FE_Chemical<dim>::getDiffusionConstant() const
{
	return diffusion_constant;
}


template<int dim>
double 
FE_Chemical<dim>::getDecayConstant() const
{
	return decay_constant;
}


template<int dim>
double 
FE_Chemical<dim>::value(const Point<dim>& location) const
{
	return	VectorTools::point_value(
				chemical_base->get_dof_handler(),
				solution,
				location);
}


// mutators:
template<int dim>
void 
FE_Chemical<dim>::setDiffusionConstant(double diffusion)
{
	diffusion_constant = diffusion;
}


template<int dim>
void 
FE_Chemical<dim>::setDecayConstant(double decay)
{
	decay_constant = decay;
}


// methods:
template<int dim>
void 
FE_Chemical<dim>::reinit()
{
	system_matrix.reinit(chemical_base->getSparsityPattern());

	solution.reinit(chemical_base->get_n_dofs());
	old_solution.reinit(chemical_base->get_n_dofs());
	old_old_solution.reinit(chemical_base->get_n_dofs());

	right_hand_side.reinit(chemical_base->get_n_dofs());

	// perhaps remove this:
	source.reinit(chemical_base->get_n_dofs());
}


template<int dim>
void 
FE_Chemical<dim>::reinit(const Chemical_FE_Base<dim>& chem_base)
{
	chemical_base = &chem_base;
	reinit();
}

template<int dim>
void 
FE_Chemical<dim>::project_initial_condition(const Function<dim> &initial_condition)
{
	VectorTools::project(chemical_base->get_dof_handler(),
						chemical_base->get_constraints(),
						QGauss<dim>(chemical_base->get_fe_degree() + 2 ),
						initial_condition,
						old_solution);
	solution = old_solution;
}


template<int dim>
void 
FE_Chemical<dim>::update(const double time_step)
{
	update_system_matrix(time_step, false);
	solve();

	old_old_solution = old_solution;
	old_solution = solution;
}

template<int dim>
void 
FE_Chemical<dim>::update(const double time_step, 
						const PointCellMap<dim>& mapping,
						const Geometry<dim>& geometry,
						const std::vector<Point<dim> >& locations, 
						// // maybe use vector of location-amount pairs instead
						// @todo perhaps restucture bacteria to be vector of location-rates pairs
						const std::vector<double>& amounts)
{
	update_source_from_map(mapping, geometry, locations, amounts); // amounts chosen according to chem
	
	update_system_matrix(time_step, true);
	solve();

	old_old_solution = old_solution;
	old_solution = solution;
}


template<int dim>
void
FE_Chemical<dim>::update_source_from_map(const PointCellMap<dim>& point_cell_map,
										const Geometry<dim>& geometry,
										const std::vector<Point<dim> >& locations, 
										const std::vector<double>& amounts)
{
	source = 0;

	const double scale_factor = 0.0016;
	const unsigned int number_bacteria = locations.size();

	const unsigned int dofs_per_cell = chemical_base->get_fe().dofs_per_cell;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	// loop over bacteria:
	for(unsigned int i = 0; i < number_bacteria; ++i)
	{
		std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
		cell_point = 
			point_cell_map.get_cell_point_pair(locations[i], &geometry);

		Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

		FEValues<dim> fe_values(StaticMappingQ1<dim>::mapping, 
				                chemical_base->get_dof_handler().get_fe(),
				                q, 
				                UpdateFlags(update_values));

		fe_values.reinit(cell_point.first);

		cell_point.first->get_dof_indices (local_dof_indices);

		/* shoudlnt this be += ?? */

		for (unsigned int j=0; j<dofs_per_cell; ++j)
			source(local_dof_indices[j]) += //check ***
				scale_factor*amounts[i]*fe_values.shape_value(j,0);
	}
}



/** ASSEMBLE CHEMICALS SYSTEM
* THIS IS THE MAIN STEP OF SOLVING
*	want to have option of not including a flow, in a clean way
*   also -- flow contribution need only be calculated once, since flow is
* 	stationary ... maybe compute this in a seperate method for just the first time step
* can use bdf2 bool as an indicator -- or store a ``is_first_time_step'' bool
* and set bdf2 bool equal to that, or !that ...
*/

template<int dim>
void 
FE_Chemical<dim>::update_system_matrix(const double time_step, const bool using_sources) // sources, advection ...
{
	// const bool use_bdf2_scheme = (time_step_number !=0); // ...
	if(use_bdf2_scheme == true)
	{
		system_matrix.copy_from(chemical_base->get_mass_matrix());
		system_matrix *= (1.5 + time_step*decay_constant);
		system_matrix.add( time_step*diffusion_constant,
							chemical_base->get_stiffness_matrix());
	}
	else
	{
		system_matrix.copy_from(chemical_base->get_mass_matrix());
		system_matrix *= (1.0 + time_step*decay_constant);
		system_matrix.add(time_step*diffusion_constant,
							chemical_base->get_stiffness_matrix());
	}

	// RIGHT HAND SIDE:
	right_hand_side = 0;

	const QGauss<dim> quadrature_formula(chemical_base->get_fe_degree() + 2);

	FEValues<dim>	chemical_fe_values(chemical_base->get_fe(),
			                            quadrature_formula,                                           
			                            update_values    |
			                            update_gradients |
			                            update_hessians  |
			                            update_quadrature_points  |
			                            update_JxW_values);

	const unsigned int dofs_per_cell = chemical_base->get_fe().dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	Vector<double> local_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// extrapolated values, gradients, laplacians
	std::vector<double>				old_values(n_q_points);
	std::vector<double>				old_old_values(n_q_points);
	std::vector<Tensor<1, dim> >	old_grads(n_q_points);
	std::vector<Tensor<1, dim> >	old_old_grads(n_q_points);
	std::vector<double>				old_laplacians(n_q_points);
	std::vector<double> 			old_old_laplacians(n_q_points);

	// shape functions:
	std::vector<double> 			phi_T(dofs_per_cell);
	std::vector<Tensor<1, dim> >	grad_phi_T(dofs_per_cell);


	//typename DoFHandler<dim>::active_cell_iterator 
	auto cell         = chemical_base->get_dof_handler().begin_active();
	const auto endc   = chemical_base->get_dof_handler().end();
	unsigned int cell_id = 0; // for velocity access
	// auto stokes_cell  = stokes_dof_handler.begin_active();

	for(; cell!=endc; ++cell , ++cell_id)
	{
		local_rhs = 0;

		chemical_fe_values.reinit(cell);

		chemical_fe_values.get_function_values(old_solution, old_values);
		chemical_fe_values.get_function_values(old_old_solution, old_old_values);

		chemical_fe_values.get_function_gradients(old_solution, old_grads);
		chemical_fe_values.get_function_gradients(old_old_solution, old_old_grads);

		chemical_fe_values.get_function_laplacians(old_solution, old_laplacians);
		chemical_fe_values.get_function_laplacians(old_old_solution, old_old_laplacians);

		const double nu = compute_viscosity(
										chemical_base->get_cell_velocity_values(cell_id),
										cell->diameter());


		// local to global:
		for(unsigned int q = 0; q < n_q_points; ++q)
		{
			for(unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_T[k] = chemical_fe_values.shape_grad(k,q);
				phi_T[k] = chemical_fe_values.shape_value(k,q);
			}

			const double T_term_for_rhs
				= (use_bdf2_scheme ?
                	(2.*old_values[q] - 0.5*old_old_values[q])
					:
					old_values[q]);

			const Tensor<1, dim> ext_grad_T
				= (use_bdf2_scheme ?
					(2.*old_grads[q] - old_old_grads[q])
					:
					old_grads[q]);

			const Tensor<1, dim> extrapolated_u = 
					chemical_base->get_cell_quad_velocity_value(cell_id, q);
			//velocity_values[q];

			for(unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				local_rhs(i) +=
					(
						T_term_for_rhs * phi_T[i]
						-
						time_step *
						extrapolated_u * ext_grad_T * phi_T[i]
						-
						time_step *
						nu * ext_grad_T * grad_phi_T[i]
					) 
					* 
					chemical_fe_values.JxW(q); 
			} // for rhs dofs
		} // for quadrature points

		cell->get_dof_indices(local_dof_indices);
		chemical_base->get_constraints().distribute_local_to_global(
															local_rhs,
															local_dof_indices,
															right_hand_side);

	} // for cells

	/** INCLUDE SOURCE TERMS ...
	*/
	// if map != NULL ...
	if(using_sources == true)
		right_hand_side.add(time_step, source);

	use_bdf2_scheme = true; // set to true after first run

}


template<int dim>
double 
FE_Chemical<dim>::compute_viscosity(const std::vector<Tensor<1,dim> >& velocity_values,
            						const double cell_diameter)
{
	const double beta = 0.017 * dim; // *** heuristic -- run experiments to get ``best'' value

	double max_velocity = 0;

	const unsigned int n_q_points = velocity_values.size();

	for(unsigned int q = 0; q < n_q_points; ++q)
	{
		const Tensor<1,dim> u = velocity_values[q];

		max_velocity = std::max(std::sqrt(u*u), max_velocity);
	}

	// nu = beta ||w(x)|| h(x)
	return beta*max_velocity*cell_diameter;
}


template<int dim>
void 
FE_Chemical<dim>::solve()
{
    SolverControl solver_control(1000, 1e-8 * right_hand_side.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> chemical_preconditioner;
    chemical_preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, right_hand_side,
           chemical_preconditioner);
    chemical_base->get_constraints().distribute(solution);
}


template<int dim>
void 
FE_Chemical<dim>::output_solution(const std::string& output_directory,
					 const unsigned int chem_id,
					 const unsigned int save_step) const
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(chemical_base->get_dof_handler());

	std::string chemical_name = "Chemical" + Utilities::int_to_string(chem_id,3);

	data_out.add_data_vector(solution, chemical_name);
	data_out.build_patches();
	const std::string filename = output_directory
				  + "/"
	              + chemical_name
	              + "_"
	              + Utilities::int_to_string(save_step,4)
	              + ".vtk";
	std::ofstream output(filename.c_str());
	data_out.write_vtk(output);
}


}}

#endif