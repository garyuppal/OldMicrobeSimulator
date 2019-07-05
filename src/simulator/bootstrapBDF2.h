#ifndef MICROBE_SIMULATOR_BOOTSTRAP_BDF2_H
#define MICROBE_SIMULATOR_BOOTSTRAP_BDF2_H


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

#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <sstream>
#include <limits>
#include <random>
// #include "../bacteria/fitness.h"
// #include "../chemicals/chemicals.h"
// #include "../sources/sources.h"

#include "../utility/argparser.h"
#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "../bacteria/bacteria.h"
#include "../utility/my_grid_generator.h"

#include "./exact_solutions.h"

namespace MicrobeSimulator{ namespace Bootstrap{
	using namespace dealii;

template<int dim>
class DebugBootstrap{
public:
	DebugBootstrap();

	void run_test_grid(const ArgParser& parameters);

	void run_bacteria(const ArgParser& parameters);

	void run_viscosity(const ArgParser& parameters);

	void run(const ArgParser& parameters);
	void run_pointSource(const ArgParser& parameters);
	void run_debug(const ArgParser& parameters);

	void run_advection(const ArgParser& parameters);
	void run_full(const ArgParser& parameters);
	
    void run_bdf2_decay(const ArgParser& parameters);

    void run_assemble_bdf_point(const ArgParser& parameters);
    void run_assemble_point(const ArgParser& parameters);

private:
		Geometry<dim> geometry;
		AdvectionField<dim> advection_field;
		Bacteria<dim> bacteria;

		// FINITE ELEMENT CHEMICAL:
		const unsigned int 		fe_degree;

		Triangulation<dim> 		triangulation;
		FE_Q<dim>				fe;
		DoFHandler<dim>			dof_handler;

		// ConstraintMatrix 	 	constraint_matrix;
	    ConstraintMatrix     	constraints;
	    SparsityPattern      	sparsity_pattern;

	    SparseMatrix<double> 	mass_matrix; 
	    SparseMatrix<double> 	stiffness_matrix;

	    SparseMatrix<double> 	goods_system_matrix;
	    Vector<double>			public_goods;
	    Vector<double> 			old_public_goods;
	    Vector<double> 			old_old_public_goods;
	    Vector<double>      	goods_rhs;
	    Vector<double>			goods_source;

	    SparseMatrix<double> 	waste_system_matrix;
	    Vector<double>			waste;
	    Vector<double> 			old_waste;
	    Vector<double> 			old_old_waste;
	    Vector<double>      	waste_rhs;
	    Vector<double>			waste_source;

	    Vector<double>			temporary;
	    Vector<double> 			exact_solution;

	    Vector<double>			mass_check_vector;

	    double good_diffusion_constant;
	    double good_decay_constant;

	    double waste_diffusion_constant;
	    double waste_decay_constant;

	    // time stepping and saving:
	    double time;
	    double time_step;
	    unsigned int time_step_number;

	    unsigned int save_period;
	    unsigned int save_step_number;

	    double run_time;
	    std::string output_directory;

	    const unsigned int reproduction_delay;

	    ConvergenceTable 		good_convergence_table;
	    ConvergenceTable 		waste_convergence_table;


	    //--------------------------------------------------------------------------------
	    void setup_system(const ArgParser& parameters);
	    void setup_system_constants(const ArgParser& parameters);
	    void setup_geometry(const ArgParser& parameters);
	    void setup_advection(const ArgParser& parameters);
	    void setup_bacteria(const ArgParser& parameters);

		std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, 
		  			unsigned int number_groups);
  		std::vector<Point<dim> > getMixerLocations(unsigned int number_groups);

	    void setup_grid(const ArgParser& parameters, unsigned int cycle = 0);
    	void refine_grid(unsigned int global_refinement, 
    		unsigned int sphere_refinement);
	    void refineMeshSpheres(unsigned int sphere_refinement);
	    void output_grid();
	    void setup_dofs();
	    void initialize_vectors_matrices();
	    void assemble_matrices();

	    void assemble_simple_system();
	    void solve();
	    void output_chemicals() const;

	    // artificial viscosity:
	    void assemble_advection_system();
	    	double compute_viscosity(const std::vector<Tensor<1,dim> >& velocity_values,
	    							const double cell_diameter);
		double get_CFL_time_step();
 		double get_maximal_velocity();

	    void assemble_full_system(const bool use_sources);

		void updateSources();
        void output_bacteria() const; 


	    // debugging:
	    void create_mass_check_vector();
	    void process_solution(unsigned int cycle, ConvergenceTable& convergence_table,
	    	const Function<dim>& sol, const Vector<double>& numerical_solution);
	    void project_initial_condition(const Function<dim>& intial_condition,
	    								Vector<double>& numerical_field);
	    void output_convergence_table(ConvergenceTable& convergence_table, 
	    	std::string outfile);

	    void projectExactSolution(const Function<dim>& sol);

	    // void update_exact_solution_vector(const Function<dim>& sol);
	    void process_time_solution(const Function<dim>& sol,
	    		std::vector<double>&	L_infty,
	    		std::vector<double>&	L2,
	    		std::vector<double>&	H1,
	    		const Vector<double>&	num_sol);
	    void output_vector(const std::vector<double>& data, std::string file);

	    double getFieldMass(const Vector<double>& num_sol);

	    std::vector<Point<dim> > getGaussGroup(unsigned int number_bacteria);
	    void run_simple(unsigned int run_number);
	    void run_bdf2(unsigned int run_number);
};


// IMPLEMENTATION:
//--------------------------------------------------------------------------------------------
template<int dim>
DebugBootstrap<dim>::DebugBootstrap()
	:
	fe_degree(1),
	fe(fe_degree),
	dof_handler(triangulation),
	time(0),
	time_step_number(0),
	save_step_number(0),
	output_directory("./"),
	reproduction_delay(5)
{}

template<int dim>
void 
DebugBootstrap<dim>::setup_system(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_advection(parameters);
	setup_bacteria(parameters);
	setup_grid(parameters);
	setup_dofs();
	initialize_vectors_matrices();
	assemble_matrices();

	std::cout << "matrices assembled, getting time step" << std::endl;

	time_step = get_CFL_time_step();
}


template<int dim>
void 
DebugBootstrap<dim>::setup_system_constants(const ArgParser& parameters)
{
	std::cout << "...Setting up system constants" << std::endl;
	run_time = parameters.getRunTime();
	save_period = parameters.getSavePeriod();
	
	good_diffusion_constant = parameters.getGoodDiffusionConstant();
	good_decay_constant = parameters.getGoodDecayConstant();

	waste_diffusion_constant = parameters.getWasteDiffusionConstant();
	waste_decay_constant = parameters.getWasteDecayConstant();

	output_directory = parameters.getOutputDirectory();
}


template<int dim>
void 
DebugBootstrap<dim>::setup_geometry(const ArgParser& parameters)
{
	std::cout << "...Initializing geometry" << std::endl;
	geometry.initialize(parameters.getGeometryFile(), 
		parameters.getMeshFile());
	
	std::string geo_out_file = output_directory + "/geometryInfo.dat";
	std::ofstream geo_out(geo_out_file);
	geometry.printInfo(geo_out);
	geometry.printInfo(std::cout);
}


template<int dim>
void 
DebugBootstrap<dim>::setup_advection(const ArgParser& parameters)
{
	std::cout << "...Initializing advection" << std::endl;
	if( (parameters.getVelocityType() != VelocityType::NUMERICAL_FLOW)
		&& (parameters.getVelocityType() != VelocityType::TILE_FLOW) )
	{
		advection_field.initialize(parameters.getVelocityType(),
								geometry.getBottomLeftPoint(),
								geometry.getTopRightPoint(),
								parameters.getMaximumVelocity()
								);
						//  double vrad = 0, double vrotation = 0);
	}
	else
	{
		advection_field.initialize(parameters.getVelocityType(),
								geometry.getBottomLeftPoint(),
								geometry.getTopRightPoint(),
								geometry.getScaleFactor(),
								parameters.getMaximumVelocity(),
								parameters.getVelocityFile_X(),
								parameters.getVelocityFile_Y() );
	}

	advection_field.printInfo(std::cout);
}


template<int dim>
void 
DebugBootstrap<dim>::setup_bacteria(const ArgParser& parameters)
{
	const unsigned int number_bacteria = parameters.getNumberBacteria();

	if(number_bacteria < 1)
		return;

	std::cout << "...Initializing bacteria" << std::endl;
	// std::vector<Point<dim> > locations ={Point<2>(0,0)}; 

	std::cout << "\t... using mixer locations" << std::endl;
	std::vector<Point<dim> > locations = {Point<2>(3. ,2.), Point<2>(3. ,4.)};

	bacteria.initialize(parameters.getBacteriaDiffusionConstant(), 
						number_bacteria,
						locations,
						parameters.getGoodSecretionRate(),
						parameters.getWasteSecretionRate() );

	std::cout << "...Setting fitness constants" << std::endl;
	bacteria.setFitnessConstants(parameters.getAlphaGood(),
								parameters.getAlphaWaste(),
								parameters.getGoodSaturation(),
								parameters.getWasteSaturation(),
								parameters.getSecretionCost() );
    bacteria.printInfo(std::cout);
}


template<int dim>
std::vector<Point<dim> > 
DebugBootstrap<dim>::getBacteriaLocations(unsigned int number_bacteria, 
	unsigned int number_groups)
{
	std::cout << "...Finding group positions" << std::endl;
	if(number_groups == 0)
		number_groups = number_bacteria; // no groups same as NB ``groups''

	std::vector<Point<dim> > group_locations;
	group_locations.reserve(number_groups);

	Point<dim> temp_point;
	for(unsigned int i = 0; i < number_groups; i++)
	{
		bool found = false;

		while(!found)
		{
			// temp_point[dim_itr] = (xmax-xmin)*((double)rand() / RAND_MAX) + xmin;
			for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
				temp_point[dim_itr] = (geometry.getWidth(dim_itr))*((double)rand() / RAND_MAX) 
					+ geometry.getBottomLeftPoint()[dim_itr];

			if( geometry.isInDomain(temp_point) )
			{
				group_locations.push_back(temp_point);
				found = true;
			}
		} // while not found
	} // for group locations
	std::cout << "...Group positions found." << std::endl;

	return group_locations;
}


template<int dim>  		
std::vector<Point<dim> > 
DebugBootstrap<dim>::getGaussGroup(unsigned int number_bacteria)
{
	std::vector<Point<dim> > group(number_bacteria);

	if(number_bacteria == 1)
		return group; 

	const double center = 0.0;
	const double width = 1.0;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(center, width);

	for(unsigned int i = 0; i < number_bacteria; i++)
	{
		Point<dim> temp_point;
		for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
		{
			double value = distribution(generator);
			if( (value > geometry.getBottomLeftPoint()[dim_itr])
					&& (value < geometry.getTopRightPoint()[dim_itr]) )
			{
				temp_point[dim_itr] = value;
			}
			else
			{
				temp_point[dim_itr] = 0.0;
			}
				
		}
		group[i] = temp_point;
	}		

	return group;
}

template<int dim>
std::vector<Point<dim> > 
DebugBootstrap<dim>::getMixerLocations(unsigned int number_groups)
{
	if(dim != 2)
		throw std::runtime_error("Mixer currently not implemented for dim != 2");
	if(number_groups != 2)
		throw std::runtime_error("Number of groups should be set to 2 for mixer");

	std::cout << "...Finding MIXER group positions" << std::endl;

	std::vector<Point<dim> > group_locations;
	group_locations.reserve(number_groups);

	Point<dim> temp_point;
	for(unsigned int i = 0; i < number_groups; i++)
	{
		temp_point[0] = geometry.getSphereAt(0).getCenter()[0] -
			geometry.getSphereAt(0).getRadius() - 2; // two away from base of circle.

		if(i == 0)
			temp_point[1] = geometry.getBottomLeftPoint()[1] 
				+ (1/3)*geometry.getWidth(1); 
		else 
			temp_point[1] = geometry.getTopRightPoint()[1] 
				- (1/3)*geometry.getWidth(1);

		if(geometry.isInDomain(temp_point))
			group_locations.push_back(temp_point);
		else
			throw std::runtime_error("Mixer group point not in domain"); 
	}
	std::cout << "...Group positions found." << std::endl;

	return group_locations;
}


template<int dim>
void 
DebugBootstrap<dim>::setup_grid(const ArgParser& parameters, unsigned int cycle)
{
	std::cout << "...Setting up grid" << std::endl;
	MyGridGenerator<dim> gridGen;
	gridGen.generateGrid(geometry,triangulation); 

	refine_grid(parameters.getGlobalRefinement() + cycle, parameters.getSphereRefinement());
	output_grid();
}


template<int dim>
void 
DebugBootstrap<dim>::refine_grid(unsigned int global_refinement, 
	unsigned int sphere_refinement)
{
	if(dim == 2)
    	refineMeshSpheres(sphere_refinement);

    triangulation.refine_global(global_refinement); 
    std::cout << "...Mesh refined globally: " << global_refinement << " times" << std::endl;
}


template<int dim>
void 
DebugBootstrap<dim>::refineMeshSpheres(unsigned int sphere_refinement)
{
  // for each circle:
  unsigned int number_spheres = geometry.getNumberSpheres();
  for(unsigned int i = 0; i < number_spheres; i++)
  {
    const Point<2> center = geometry.getSphereAt(i).getCenter();
    const double radius = geometry.getSphereAt(i).getRadius();

    for(unsigned int step = 0; step < sphere_refinement; ++step)
    {
      for(auto cell : triangulation.active_cell_iterators())
      {
        for(unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
          const double distance_from_center = center.distance(cell->vertex(v));
                if (std::fabs(distance_from_center - radius) < 1e-10)
                  {
                    cell->set_refine_flag();
                    break;
                  } // if vertex on circle boundary
        } // for each vertex
      } // for each cell in mesh

      triangulation.execute_coarsening_and_refinement();
    } // for each refinement step
  } // for each circle

  std::cout << "...Refined circle boundaries: " << sphere_refinement << " times" << std::endl;
}


template<int dim>
void 
DebugBootstrap<dim>::output_grid()
{
	std::string grid_out_file = output_directory + "/grid.eps";

	std::ofstream out (grid_out_file);
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "...Grid written to " << grid_out_file << std::endl;
}


template<int dim>
void 
DebugBootstrap<dim>::setup_dofs()
{
	std::cout << "...setting up degrees of freedom" << std::endl;
	dof_handler.distribute_dofs(fe);

	std::cout << std::endl << std::endl
	      << "================================================"
	      << std::endl
	      << "Number of active cells: " << triangulation.n_active_cells()
	      << std::endl
	      << "Number of degrees of freedom: " << dof_handler.n_dofs()
	      << std::endl
	      << std::endl;

	constraints.clear();
	DoFTools::make_hanging_node_constraints (dof_handler, constraints);

	// *** can we add no normal flux constraints as well???!!!

	for(unsigned int i = 0; i < dim; i++)
	{
		if(geometry.getBoundaryConditions()[i] == BoundaryCondition::WRAP)
		{
			std::cout << "\n...USING PERIODIC " << i <<"th BOUNDARY" << std::endl;
			// ADD PERIODICITY:
			std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> >
			periodicity_vector;

			const unsigned int direction = i;
			unsigned int bid_one = 0 + 2*i;
			unsigned int bid_two = 1 + 2*i;

			GridTools::collect_periodic_faces(dof_handler, bid_one, bid_two, direction,
			                            periodicity_vector); //, offset, rotation_matrix);

			DoFTools::make_periodicity_constraints<DoFHandler<dim> > 
			(periodicity_vector, constraints); //, fe.component_mask(velocities), first_vector_components);
		} // if periodic 
	} // for each dimension

	constraints.close();
}


template<int dim>
void 
DebugBootstrap<dim>::initialize_vectors_matrices()
{
	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
									dsp,
									constraints,
									/*keep constrained_dofs = */ false); // *** not sure if false or true...

	sparsity_pattern.copy_from(dsp);

	mass_matrix.reinit(sparsity_pattern);
	stiffness_matrix.reinit(sparsity_pattern);
	goods_system_matrix.reinit(sparsity_pattern);
	waste_system_matrix.reinit(sparsity_pattern);

	public_goods.reinit(dof_handler.n_dofs());
	old_public_goods.reinit(dof_handler.n_dofs());
	old_old_public_goods.reinit(dof_handler.n_dofs());
	goods_rhs.reinit(dof_handler.n_dofs());
	goods_source.reinit(dof_handler.n_dofs());

	waste.reinit(dof_handler.n_dofs());
	old_waste.reinit(dof_handler.n_dofs());
	old_old_waste.reinit(dof_handler.n_dofs());
	waste_rhs.reinit(dof_handler.n_dofs());
	waste_source.reinit(dof_handler.n_dofs());

	temporary.reinit(dof_handler.n_dofs());
	exact_solution.reinit(dof_handler.n_dofs());
}


template<int dim>
void 
DebugBootstrap<dim>::assemble_matrices()
{
	std::cout << "...assembling system matrices" << std::endl;
	mass_matrix = 0;
	stiffness_matrix = 0;

	QGauss<dim> quadrature_formula(fe_degree + 2);
	FEValues<dim> fe_values(fe, quadrature_formula,
									update_values | update_gradients | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
	FullMatrix<double> local_stiffness_matrix(dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	std::vector<double> 			phi_T (dofs_per_cell);
	std::vector<Tensor<1, dim> > 	grad_phi_T(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

	for(; cell != endc; ++cell)
	{
		local_mass_matrix = 0;
		local_stiffness_matrix = 0;

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

					local_stiffness_matrix(i,j) += (grad_phi_T[i] * grad_phi_T[j]
						*fe_values.JxW(q));
				}
		}

		cell->get_dof_indices(local_dof_indices);
		constraints.distribute_local_to_global(local_mass_matrix,
												local_dof_indices,
												mass_matrix);
		constraints.distribute_local_to_global(local_stiffness_matrix,
												local_dof_indices,
												stiffness_matrix);
	}
}


template<int dim>
void 
DebugBootstrap<dim>::assemble_simple_system()
{
	updateSources(); 

	// goods:
	goods_system_matrix.copy_from(mass_matrix); 
	goods_system_matrix *= (1.0 + time_step*good_decay_constant);
	goods_system_matrix.add(time_step*good_diffusion_constant, 
									stiffness_matrix);

	mass_matrix.vmult(goods_rhs, old_public_goods);
	goods_rhs.add(time_step, goods_source);

	// waste:
	waste_system_matrix.copy_from(mass_matrix);
	waste_system_matrix *= (1.0 + time_step*waste_decay_constant); 
	waste_system_matrix.add(time_step*waste_diffusion_constant, 
									stiffness_matrix);

	mass_matrix.vmult(waste_rhs, old_waste);
	waste_rhs.add(time_step, waste_source);

//   constraints.condense (goods_system_matrix, goods_rhs); // not sure if this is necessary... should only be for bc's and hanging nodes
//   constraints.condense (waste_system_matrix, waste_rhs);
}


template<int dim>
void 
DebugBootstrap<dim>::solve()
{
	{
		SolverControl solver_control(1000, 1e-8 * goods_rhs.l2_norm());
		SolverCG<> cg(solver_control);

		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(goods_system_matrix, 1.0);
		cg.solve(goods_system_matrix, public_goods, goods_rhs,
		       preconditioner);
		constraints.distribute(public_goods);
	}
	{
		SolverControl solver_control(1000, 1e-8 * waste_rhs.l2_norm());
		SolverCG<> cg(solver_control);

		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(waste_system_matrix, 1.0);
		cg.solve(waste_system_matrix, waste, waste_rhs,
		       preconditioner);
		constraints.distribute(waste);
	}

	 // update solutions:
	old_old_public_goods = old_public_goods;
	old_public_goods = public_goods;
	old_old_waste = old_waste;
	old_waste = waste;
}


template<int dim>
void 
DebugBootstrap<dim>::output_chemicals() const
{
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(public_goods,"public_goods");
		data_out.build_patches();
		const std::string filename = output_directory
									+ "/public_goods-"
									+ Utilities::int_to_string(save_step_number,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(waste,"waste");
		data_out.build_patches();
		const std::string filename = output_directory
									+ "/waste-"
									+ Utilities::int_to_string(save_step_number,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}
}


template<int dim>
void
DebugBootstrap<dim>::output_bacteria() const
{
	std::string outfile = output_directory
		+ "/bacteria-" + std::to_string(save_step_number) + ".dat";
	std::ofstream out(outfile);
	bacteria.print(out);
}




// debugging:
template<int dim>
void 
DebugBootstrap<dim>::process_solution(unsigned int cycle, 
										ConvergenceTable& convergence_table,
										const Function<dim>& sol,
										const Vector<double>& numerical_solution)
{
	Vector<float>	difference_per_cell(triangulation.n_active_cells());

	// compute L2-error
	VectorTools::integrate_difference(dof_handler,
										numerical_solution,
										sol,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::L2_norm);

	// compute H1-error
	VectorTools::integrate_difference(dof_handler,
										numerical_solution,
										sol,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::H1_seminorm);
	const double H1_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::H1_seminorm);;

	const QTrapez<1>		q_trapez;
	const QIterated<dim>	q_iterated(q_trapez, 5);
	VectorTools::integrate_difference(dof_handler,
										numerical_solution,
										sol,
										difference_per_cell,
										q_iterated,
										VectorTools::Linfty_norm);
	const double Linfty_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::Linfty_norm);

	const unsigned int n_active_cells=triangulation.n_active_cells();
	const unsigned int n_dofs=dof_handler.n_dofs();
	std::cout << "Cycle " << cycle << ':'
	        << std::endl
	        << "   Number of active cells:       "
	        << n_active_cells
	        << std::endl
	        << "   Number of degrees of freedom: "
	        << n_dofs
	        << std::endl;

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);
}


template<int dim>
double 
DebugBootstrap<dim>::getFieldMass(const Vector<double>& numerical_solution)
{
	Vector<float>	difference_per_cell(triangulation.n_active_cells());

	VectorTools::integrate_difference(dof_handler,
									numerical_solution,
									ZeroFunction<dim>(),
									difference_per_cell,
									QGauss<dim>(3),
									VectorTools::L2_norm);
	const double mass = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::L2_norm);

	return mass;
}


template<int dim>
void 
DebugBootstrap<dim>::project_initial_condition(const Function<dim>& intial_condition,
								Vector<double>& numerical_field)
{
     VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(fe_degree+2),
	                      intial_condition,
	                      numerical_field);
}


template<int dim>
void 
DebugBootstrap<dim>::projectExactSolution(const Function<dim>& sol)
{
     VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(fe_degree+2),
	                      sol,
	                      exact_solution);
}


template<int dim>
void 
DebugBootstrap<dim>::output_convergence_table(ConvergenceTable& convergence_table,
	std::string outfile)
{

	convergence_table.set_precision("L2", 3);
	convergence_table.set_precision("H1", 3);
	convergence_table.set_precision("Linfty", 3);
	convergence_table.set_scientific("L2", true);
	convergence_table.set_scientific("H1", true);
	convergence_table.set_scientific("Linfty", true);

	convergence_table.set_tex_caption("cells", "\\# cells");
	convergence_table.set_tex_caption("dofs", "\\# dofs");
	convergence_table.set_tex_caption("L2", "L^2-error");
	convergence_table.set_tex_caption("H1", "H^1-error");
	convergence_table.set_tex_caption("Linfty", "L^\\infty-error");

	convergence_table.set_tex_format("cells", "r");
	convergence_table.set_tex_format("dofs", "r");

	convergence_table.add_column_to_supercolumn("cycle", "n cells");
	convergence_table.add_column_to_supercolumn("cells", "n cells");

	std::cout << std::endl;
	convergence_table.write_text(std::cout); 
	

	std::vector<std::string> new_order;
	new_order.emplace_back("n cells");
	new_order.emplace_back("H1");
	new_order.emplace_back("L2");
	convergence_table.set_column_order (new_order);


	convergence_table
		.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
	convergence_table
		.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
	convergence_table
		.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
	convergence_table
		.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

  	std::cout << std::endl;
	convergence_table.write_text(std::cout); 

	std::string filename = output_directory 
							+ "/"
							+ outfile
							+ ".tex";
	std::ofstream table_file(filename);
	convergence_table.write_tex(table_file);
}

template<int dim>
void 
DebugBootstrap<dim>::process_time_solution(const Function<dim>& sol,
		std::vector<double>&	L_infty,
		std::vector<double>&	L2,
		std::vector<double>&	H1,
		const Vector<double>& numerical_solution)
{
	Vector<float>	difference_per_cell(triangulation.n_active_cells());

	// compute L2-error
	VectorTools::integrate_difference(dof_handler,
										numerical_solution,
										sol,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::L2_norm);
	L2.push_back(L2_error);

	// compute H1-error
	VectorTools::integrate_difference(dof_handler,
										numerical_solution,
										sol,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::H1_seminorm);
	const double H1_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::H1_seminorm);;
	H1.push_back(H1_error);

	// compute L_inty_error:
	const QTrapez<1>		q_trapez;
	const QIterated<dim>	q_iterated(q_trapez, 5);
	VectorTools::integrate_difference(dof_handler,
										numerical_solution,
										sol,
										difference_per_cell,
										q_iterated,
										VectorTools::Linfty_norm);
	const double Linfty_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::Linfty_norm);
	L_infty.push_back(Linfty_error);
}


template<int dim>
void 
DebugBootstrap<dim>::output_vector(const std::vector<double>& data, std::string file)
{
	const std::string filename = output_directory + "/" + file + ".dat";
	std::ofstream out(filename);

	for(unsigned int i = 0; i<data.size(); i++)
		out << data[i] << std::endl;
}


// ADVECTION:
// -------------------------------------------------------------------------------------------------
template<int dim>
void 
DebugBootstrap<dim>::assemble_advection_system()
{
	const bool use_bdf2_scheme = (time_step_number != 0);
	// const double beta = ;

	if(use_bdf2_scheme == true)
	{
		goods_system_matrix.copy_from(mass_matrix);
		goods_system_matrix *= 1.5; //3.0/2.0; // ***need to add decay term...
		goods_system_matrix.add(time_step*good_diffusion_constant, stiffness_matrix);
	}
	else
	{
		goods_system_matrix.copy_from(mass_matrix); 
		goods_system_matrix.add(time_step*good_diffusion_constant, 
										stiffness_matrix); // diffusion treated explicitly
	}


	// RHS:
	goods_rhs = 0;

	const QGauss<dim> quadrature_formula(fe_degree + 2);
	FEValues<dim> fe_values(fe,quadrature_formula,                                           
										   update_values    |
                                           update_gradients |
                                           update_hessians  |
                                           update_quadrature_points  |
                                           update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	Vector<double> 	goods_local_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// goods:
	std::vector<Tensor<1, dim> > 	velocity_values(n_q_points); 
	std::vector<double> 			old_goods_values(n_q_points);
	std::vector<double> 			old_old_goods_values(n_q_points);
	std::vector<Tensor<1,dim> >		old_goods_grads(n_q_points);
	std::vector<Tensor<1,dim> >		old_old_goods_grads(n_q_points);
	std::vector<double>				old_goods_laplacians(n_q_points);
	std::vector<double>				old_old_goods_laplacians(n_q_points);

	// shape functions:
	std::vector<double>				phi_T(dofs_per_cell);
	std::vector<Tensor<1,dim> >		grad_phi_T(dofs_per_cell);


	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
													endc = dof_handler.end();

	for(; cell!=endc; ++cell)
	{
		goods_local_rhs = 0;

		fe_values.reinit(cell);
		
		// update goods:
		fe_values.get_function_values(old_public_goods, old_goods_values);
		fe_values.get_function_values(old_old_public_goods, old_old_goods_values);

		fe_values.get_function_gradients(old_public_goods, old_goods_grads);
		fe_values.get_function_gradients(old_old_public_goods, old_old_goods_grads);

		fe_values.get_function_laplacians(old_public_goods, old_goods_laplacians);
		fe_values.get_function_laplacians(old_old_public_goods, old_old_goods_laplacians);
			
		advection_field.value_list(fe_values.get_quadrature_points(), velocity_values); 

		// compute viscosity:
		const double nu_goods = compute_viscosity(velocity_values, cell->diameter());

		// local to global:

		for(unsigned int q = 0; q < n_q_points; ++q)
		{
			for(unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_T[k] = fe_values.shape_grad(k,q);
				phi_T[k] = fe_values.shape_value(k,q);
			}

			// goods:
			const double good_T_term_for_rhs
		      = (use_bdf2_scheme ?
	               (2.*old_goods_values[q] - 0.5*old_old_goods_values[q])
	               :
	               old_goods_values[q]);

		    const Tensor<1,dim> good_ext_grad_T
	            = (use_bdf2_scheme ?
	               (2.*old_goods_grads[q] - old_old_goods_grads[q])
	               :
	               old_goods_grads[q]);

            const Tensor<1,dim> extrapolated_u = velocity_values[q];


	        for(unsigned int i = 0; i < dofs_per_cell; ++i)
	        {
	        	goods_local_rhs(i) +=  
        				(
	        				 good_T_term_for_rhs * phi_T[i]
                             -
                             time_step *
                             extrapolated_u * good_ext_grad_T * phi_T[i]
                             -
                             time_step *
                             nu_goods * good_ext_grad_T * grad_phi_T[i]
                      	)
                        *
                        fe_values.JxW(q); 
	        }
		}

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (goods_local_rhs,
                                              local_dof_indices,
                                              goods_rhs);
    } // loop over cells
}


template<int dim>
void 
DebugBootstrap<dim>::assemble_full_system(const bool use_sources)
{
	const bool use_bdf2_scheme = (time_step_number != 0);
	// const double beta = ;

	if(use_bdf2_scheme == true)
	{
		// goods:
		goods_system_matrix.copy_from(mass_matrix);
		goods_system_matrix *= (1.5 + time_step*good_decay_constant); 
		// not too sure about this..., else treat  decay explictly
		goods_system_matrix.add(time_step*good_diffusion_constant, stiffness_matrix);

		// waste:
		waste_system_matrix.copy_from(mass_matrix);
		waste_system_matrix *= (1.5 + time_step*waste_decay_constant); 
		waste_system_matrix.add(time_step*waste_diffusion_constant, stiffness_matrix);
	}
	else
	{
		// goods:
		goods_system_matrix.copy_from(mass_matrix); 
		goods_system_matrix *= (1.0 + time_step*good_decay_constant);
		goods_system_matrix.add(time_step*good_diffusion_constant, 
										stiffness_matrix); // diffusion treated explicitly

		// waste:
		waste_system_matrix.copy_from(mass_matrix); 
		waste_system_matrix *= (1.0 + time_step*waste_decay_constant);
		waste_system_matrix.add(time_step*waste_diffusion_constant, 
										stiffness_matrix); // diffusion treated explicitly
	}


	// RHS:
	goods_rhs = 0;
	waste_rhs = 0;

	const QGauss<dim> quadrature_formula(fe_degree + 2);
	FEValues<dim> fe_values(fe,quadrature_formula,                                           
										   update_values    |
                                           update_gradients |
                                           update_hessians  |
                                           update_quadrature_points  |
                                           update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	Vector<double> 	goods_local_rhs(dofs_per_cell);
	Vector<double> 	waste_local_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// velocity:
	std::vector<Tensor<1, dim> > 	velocity_values(n_q_points); 

	// goods:
	std::vector<double> 			old_goods_values(n_q_points);
	std::vector<double> 			old_old_goods_values(n_q_points);
	std::vector<Tensor<1,dim> >		old_goods_grads(n_q_points);
	std::vector<Tensor<1,dim> >		old_old_goods_grads(n_q_points);
	std::vector<double>				old_goods_laplacians(n_q_points);
	std::vector<double>				old_old_goods_laplacians(n_q_points);

	// waste:
	std::vector<double> 			old_waste_values(n_q_points);
	std::vector<double> 			old_old_waste_values(n_q_points);
	std::vector<Tensor<1,dim> >		old_waste_grads(n_q_points);
	std::vector<Tensor<1,dim> >		old_old_waste_grads(n_q_points);
	std::vector<double>				old_waste_laplacians(n_q_points);
	std::vector<double>				old_old_waste_laplacians(n_q_points);

	// sources:
	// std::vector<double> 	good_source_values(n_q_points);
	// std::vector<double> 	waste_source_values(n_q_points);
	// if(use_sources == true)
	// 	updateSources();


	// shape functions:
	std::vector<double>				phi_T(dofs_per_cell);
	std::vector<Tensor<1,dim> >		grad_phi_T(dofs_per_cell);


	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
													endc = dof_handler.end();

	for(; cell!=endc; ++cell)
	{
		goods_local_rhs = 0;
		waste_local_rhs = 0;

		fe_values.reinit(cell);
		
		// update goods:
		fe_values.get_function_values(old_public_goods, old_goods_values);
		fe_values.get_function_values(old_old_public_goods, old_old_goods_values);

		fe_values.get_function_gradients(old_public_goods, old_goods_grads);
		fe_values.get_function_gradients(old_old_public_goods, old_old_goods_grads);

		fe_values.get_function_laplacians(old_public_goods, old_goods_laplacians);
		fe_values.get_function_laplacians(old_old_public_goods, old_old_goods_laplacians);
			
		// update waste:
		fe_values.get_function_values(old_waste, old_waste_values);
		fe_values.get_function_values(old_old_waste, old_old_waste_values);

		fe_values.get_function_gradients(old_waste, old_waste_grads);
		fe_values.get_function_gradients(old_old_waste, old_old_waste_grads);

		fe_values.get_function_laplacians(old_waste, old_waste_laplacians);
		fe_values.get_function_laplacians(old_old_waste, old_old_waste_laplacians);

		// update sources:	
		// if(use_sources == true)
		// {
		// 	fe_values.get_function_values(goods_source, good_source_values);
		// 	fe_values.get_function_values(waste_source, waste_source_values);
		// }

		// update velocity:
		advection_field.value_list(fe_values.get_quadrature_points(), velocity_values); 

		// compute viscosity:
		const double nu_goods = compute_viscosity(velocity_values, cell->diameter());
		const double nu_waste = nu_goods; // using same for now -- doesn't depend on residual,
		// only depends on velocity and cell size

		// local to global:

		for(unsigned int q = 0; q < n_q_points; ++q)
		{
			for(unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_T[k] = fe_values.shape_grad(k,q);
				phi_T[k] = fe_values.shape_value(k,q);
			}

			// goods:
			const double good_T_term_for_rhs
		      = (use_bdf2_scheme ?
	               (2.*old_goods_values[q] - 0.5*old_old_goods_values[q])
	               :
	               old_goods_values[q]);

		    const Tensor<1,dim> good_ext_grad_T
	            = (use_bdf2_scheme ?
	               (2.*old_goods_grads[q] - old_old_goods_grads[q])
	               :
	               old_goods_grads[q]);

	        // waste:
			const double waste_T_term_for_rhs
		      = (use_bdf2_scheme ?
	               (2.*old_waste_values[q] - 0.5*old_old_waste_values[q])
	               :
	               old_waste_values[q]);

		    const Tensor<1,dim> waste_ext_grad_T
	            = (use_bdf2_scheme ?
	               (2.*old_waste_grads[q] - old_old_waste_grads[q])
	               :
	               old_waste_grads[q]);	            

	        // velocity:
            const Tensor<1,dim> extrapolated_u = velocity_values[q];

	        for(unsigned int i = 0; i < dofs_per_cell; ++i)
	        {
	        	goods_local_rhs(i) +=  
        				(
	        				 good_T_term_for_rhs * phi_T[i]
                             -
                             time_step *
                             extrapolated_u * good_ext_grad_T * phi_T[i]
                             -
                             time_step *
                             nu_goods * good_ext_grad_T * grad_phi_T[i]
	                         // +
                          //    time_step *
                          //    good_source_values[q] * phi_T[i] 
                             // *** double check: check if need old_old_source ...
                      	)
                        *
                        fe_values.JxW(q); 

	        	waste_local_rhs(i) +=  
        				(
	        				 waste_T_term_for_rhs * phi_T[i]
                             -
                             time_step *
                             extrapolated_u * waste_ext_grad_T * phi_T[i]
                             -
                             time_step *
                             nu_waste * waste_ext_grad_T * grad_phi_T[i]
                             // +
                             // time_step *
                             // waste_source_values[q] * phi_T[i]
                      	)
                        *
                        fe_values.JxW(q); 
	        }
		}

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (goods_local_rhs,
                                              local_dof_indices,
                                              goods_rhs);
      constraints.distribute_local_to_global (waste_local_rhs,
                                              local_dof_indices,
                                              waste_rhs);
    } // loop over cells

    if(use_sources == true)
    {
	    updateSources();
	    goods_rhs.add(time_step, goods_source);
	    waste_rhs.add(time_step, waste_source);
	}
}


template<int dim>
void
DebugBootstrap<dim>::updateSources()
{
	goods_source = 0;
	waste_source = 0;

	const unsigned int number_bacteria = bacteria.getSize();
	const double scale_factor = 0.0016; //*** check this -- still seems too small!! ??
 
	for(unsigned int i = 0; i < number_bacteria; i++)
	{
		const double good_secretion = bacteria.getGoodSecretionRate(i);
		const double waste_secretion = bacteria.getWasteSecretionRate(i);
// std::cout << "gs: " << good_secretion << " ws: " << waste_secretion << std::endl;
		VectorTools::create_point_source_vector(dof_handler,
												bacteria.getLocation(i),
												temporary);

		goods_source.add(scale_factor*good_secretion, temporary);
		waste_source.add(scale_factor*waste_secretion, temporary);
	}
}


template<int dim>
void
DebugBootstrap<dim>::create_mass_check_vector()
{
	mass_check_vector.reinit(dof_handler.n_dofs());

	const QGauss<dim>  quadrature_formula((fe.degree+1));
	FEValues<dim> fe_values (fe, quadrature_formula,
	                     update_values | update_gradients | update_JxW_values);
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();

	Vector<double>       cell_rhs(dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell: dof_handler.active_cell_iterators())
	{
		fe_values.reinit (cell);

		cell_rhs = 0;
		for (unsigned int q_index=0; q_index < n_q_points; ++q_index)
		{
			for (unsigned int i=0; i < dofs_per_cell; ++i)
				cell_rhs(i) += (fe_values.shape_value (i, q_index) *
			                1 *
			                fe_values.JxW (q_index));
		}
		cell->get_dof_indices (local_dof_indices);

		// for (unsigned int i=0; i < dofs_per_cell; ++i)
		// 	mass_check_vector(local_dof_indices[i]) += cell_rhs(i);
      constraints.distribute_local_to_global (cell_rhs,
                                      local_dof_indices,
                                      mass_check_vector);
	}

}


template<int dim>
double 
DebugBootstrap<dim>::compute_viscosity(const std::vector<Tensor<1,dim> >& velocity_values,
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
double 
DebugBootstrap<dim>::get_CFL_time_step()
{
	const double min_time_step = 0.01;

	const double maximal_velocity = get_maximal_velocity();
	double cfl_time_step = 0;


	std::cout << "maximal_velocity is: " << maximal_velocity << std::endl;

	if(maximal_velocity >= 0.01)
		cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
			fe_degree *
			GridTools::minimal_cell_diameter(triangulation) /
			maximal_velocity;
	else
		cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
			fe_degree *
			GridTools::minimal_cell_diameter(triangulation) /
			0.01;

	cfl_time_step = std::min(min_time_step,cfl_time_step);

	std::cout << "...using time step: " << cfl_time_step << std::endl;

	return cfl_time_step;
}


template<int dim>
double 
DebugBootstrap<dim>::get_maximal_velocity()
{
	const QIterated<dim> quadrature_formula(QTrapez<1>(), fe_degree+1);
	const unsigned int n_q_points = quadrature_formula.size();

	FEValues<dim> fe_values(fe,quadrature_formula,update_values | update_quadrature_points);
	std::vector<Tensor<1,dim> > velocity_values(n_q_points);
	double max_velocity = 0;

	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

	for(;cell != endc; ++cell)
	{
		fe_values.reinit(cell);
		advection_field.value_list(fe_values.get_quadrature_points(), velocity_values);

		for(unsigned int q = 0; q < n_q_points; ++q)
			max_velocity = std::max(max_velocity, velocity_values[q].norm());
	}

	return max_velocity;
}

// ---------------------------------------------------------------------------------------------------
template<int dim>
void 
DebugBootstrap<dim>::run(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_advection(parameters);

	const unsigned int number_runs = 5;
	std::array<unsigned int, number_runs> number_bacteria = {1, 10, 100, 500, 1000}; //{1, 10, 100}; //, 500, 1000};

	// std::array<double, number_runs> good_decay = {50, 50, 50, 50, 50};
	// std::array<double, number_runs> waste_decay = {15, 15, 15, 15, 15};

	for(unsigned int i = 0; i < number_runs; i++)
	{

		std::cout << "...Initializing bacteria" << std::endl;
		std::vector<Point<dim> > locations = getGaussGroup(number_bacteria[i]); 

		bacteria.initialize(parameters.getBacteriaDiffusionConstant(), 
							number_bacteria[i],
							locations,
							parameters.getGoodSecretionRate(),
							parameters.getWasteSecretionRate() );

		std::cout << "...Setting fitness constants" << std::endl;
		bacteria.setFitnessConstants(parameters.getAlphaGood(),
									parameters.getAlphaWaste(),
									parameters.getGoodSaturation(),
									parameters.getWasteSaturation(),
									parameters.getSecretionCost() );
	    bacteria.printInfo(std::cout);

	    // set up decay constant:
	    good_decay_constant = 50; //good_decay[i];
	    waste_decay_constant = 15; //waste_decay[i];

		setup_grid(parameters);
		setup_dofs();

		time_step = 0.001;
		save_period = 100;
		run_time = 1.0;

		run_simple(i);
		run_bdf2(i);
	}
}


template<int dim>
void
DebugBootstrap<dim>::run_simple(unsigned int run_number)
{

	std::cout << "SIMPLE:\n\tgood decay: " << good_decay_constant
		<< "\n\twaste decay: " << waste_decay_constant << std::endl;

	std::vector<double> c1_values, c2_values, total_mass_goods, total_mass_waste;

	// reinitialize:
	// reset time:
		time_step_number = 0;
		save_step_number = 0;
		time = 0;
		
	initialize_vectors_matrices();
	assemble_matrices();
	create_mass_check_vector();

	do{
		assemble_simple_system(); // include point sources
		solve();
	   	time += time_step;
    	++time_step_number;


		total_mass_goods.push_back( public_goods*mass_check_vector );
		total_mass_waste.push_back( waste*mass_check_vector );

    	if( time_step_number % save_period == 0 )
    	{
    		++save_step_number;
    		std::cout << "time: " << time << std::endl;
    		// output_chemicals();

    		const double c1 = dealii::VectorTools::point_value(dof_handler,
    															 public_goods,
    															 Point<2>(0,0));

			const double c2 = dealii::VectorTools::point_value(dof_handler,
    															 waste,
    															 Point<2>(0,0));

    		
    		c1_values.push_back(c1);
    		c2_values.push_back(c2);
    	}

	} while(time < run_time);

	output_vector(c1_values,"c1_values_simple_" + Utilities::int_to_string(run_number,2));
	output_vector(c2_values,"c2_values_simple_" + Utilities::int_to_string(run_number,2));

	output_vector(total_mass_goods,"mass_c1_simple_" + Utilities::int_to_string(run_number,2));
	output_vector(total_mass_waste,"mass_c2_simple_" + Utilities::int_to_string(run_number,2));
}


template<int dim>
void
DebugBootstrap<dim>::run_bdf2(unsigned int run_number)
{
	std::cout << "BDF2:\n\tgood decay: " << good_decay_constant
		<< "\n\twaste decay: " << waste_decay_constant << std::endl;

	std::vector<double> c1_values, c2_values, total_mass_goods, total_mass_waste;

	// reinitialize:
	// reset time:
		time_step_number = 0;
		save_step_number = 0;
		time = 0;
		
	initialize_vectors_matrices();
	assemble_matrices();
	create_mass_check_vector();

	do{
		assemble_full_system(true);// include point sources
		solve();
	   	time += time_step;
    	++time_step_number;

		total_mass_goods.push_back( public_goods*mass_check_vector );
		total_mass_waste.push_back( waste*mass_check_vector );

    	if( time_step_number % save_period == 0 )
    	{
    		++save_step_number;
    		std::cout << "time: " << time << std::endl;
    		output_chemicals();

    		const double c1 = dealii::VectorTools::point_value(dof_handler,
    															 public_goods,
    															 Point<2>(0,0));

			const double c2 = dealii::VectorTools::point_value(dof_handler,
    															 waste,
    															 Point<2>(0,0));

			// const double mass_goods = getFieldMass(public_goods);
			// const double mass_waste = getFieldMass(waste);
    		
    		c1_values.push_back(c1);
    		c2_values.push_back(c2);
    	}

	} while(time < run_time);

	output_vector(c1_values,"c1_values_bdf2_" + Utilities::int_to_string(run_number,2));
	output_vector(c2_values,"c2_values_bdf2_" + Utilities::int_to_string(run_number,2));

	output_vector(total_mass_goods,"mass_c1_bdf2_" + Utilities::int_to_string(run_number,2));
	output_vector(total_mass_waste,"mass_c2_bdf2_" + Utilities::int_to_string(run_number,2));
}



template<int dim>
void
DebugBootstrap<dim>::run_bdf2_decay(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_advection(parameters);


    // set up decay constant:
    good_decay_constant = 0; //50; //good_decay[i];
    waste_decay_constant = 0; //15; //waste_decay[i];

	setup_grid(parameters);
	setup_dofs();

	time_step = 0.001;
	save_period = 100;
	run_time = 1.0;




	std::cout << "BDF2:\n\tgood decay: " << good_decay_constant
		<< "\n\twaste decay: " << waste_decay_constant << std::endl;

	std::vector<double> c1_values, c2_values, total_mass_goods, total_mass_waste;

	// reinitialize:
	// reset time:
		time_step_number = 0;
		save_step_number = 0;
		time = 0;
		
	initialize_vectors_matrices();
	assemble_matrices();
	create_mass_check_vector();

	// initial gaussian:
	project_initial_condition(ExactSolutions::Gaussian<dim>(),
								public_goods);

	project_initial_condition(ExactSolutions::Gaussian<dim>(),
								waste);

	old_public_goods = public_goods;
	old_waste = waste;

	do{
		assemble_full_system(false);// include point sources
		solve();
	   	time += time_step;
    	++time_step_number;

		total_mass_goods.push_back( public_goods*mass_check_vector );
		total_mass_waste.push_back( waste*mass_check_vector );

    	if( time_step_number % save_period == 0 )
    	{
    		++save_step_number;
    		std::cout << "time: " << time << std::endl;
    		output_chemicals();

    		const double c1 = dealii::VectorTools::point_value(dof_handler,
    															 public_goods,
    															 Point<2>(0,0));

			const double c2 = dealii::VectorTools::point_value(dof_handler,
    															 waste,
    															 Point<2>(0,0));

			// const double mass_goods = getFieldMass(public_goods);
			// const double mass_waste = getFieldMass(waste);
    		
    		c1_values.push_back(c1);
    		c2_values.push_back(c2);
    	}

	} while(time < run_time);

	output_vector(c1_values,"c1_values_bdf2");
	output_vector(c2_values,"c2_values_bdf2");

	output_vector(total_mass_goods,"mass_c1_bdf2" );
	output_vector(total_mass_waste,"mass_c2_bdf2" );
}




template<int dim>
void 
DebugBootstrap<dim>::run_pointSource(const ArgParser& parameters)
{
	setup_system(parameters);

	std::vector<double> c1_values, c2_values, total_mass_goods, total_mass_waste;
	time_step = 0.001;
	save_period = 10;

	do{
		assemble_full_system(true); // include point sources
		solve();

    	time += time_step;
    	++time_step_number;

    	if( time_step_number % save_period == 0 )
    	{
    		++save_step_number;
    		std::cout << "time: " << time << std::endl;
    		output_chemicals();

    		const double c1 = dealii::VectorTools::point_value(dof_handler,
    															 public_goods,
    															 Point<2>(0,0));

			const double c2 = dealii::VectorTools::point_value(dof_handler,
    															 waste,
    															 Point<2>(0,0));

			const double mass_goods = getFieldMass(public_goods);
			const double mass_waste = getFieldMass(waste);
    		
    		c1_values.push_back(c1);
    		c2_values.push_back(c2);

    		total_mass_goods.push_back(mass_goods);
    		total_mass_waste.push_back(mass_waste);
    	}

	} while(time < run_time);

	output_vector(c1_values,"c1_values");
	output_vector(c2_values,"c2_values");

	output_vector(total_mass_goods,"mass_c1");
	output_vector(total_mass_waste,"mass_c2");
}



template<int dim>
void
DebugBootstrap<dim>::run_debug(const ArgParser& parameters)
{
	const unsigned int n_cycles = 6;
    
    setup_system_constants(parameters);
    setup_geometry(parameters);
    setup_advection(parameters);
    setup_bacteria(parameters);

    // error over time:
    std::vector<double> good_L_infty_error_VTIME, good_L2_error_VTIME, good_H1_error_VTIME;
    std::vector<double> waste_L_infty_error_VTIME, waste_L2_error_VTIME, waste_H1_error_VTIME;

	for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
	{
		time = 0;
		time_step_number = 0;
		save_step_number = 0;

		triangulation.clear();
 		setup_grid(parameters, cycle+1);

 		const double distort_factor = 0.2;
		GridTools::distort_random(distort_factor, triangulation);

	    setup_dofs();
	    initialize_vectors_matrices();
	    assemble_matrices();

	    project_initial_condition(		    				
    		ExactSolutions::GaussFull<dim>(good_diffusion_constant,
		    							good_decay_constant,
		    							parameters.getMaximumVelocity(), 
		    							time),
    		public_goods);
         old_public_goods = public_goods;

	    project_initial_condition(		    				
    		ExactSolutions::GaussFull<dim>(waste_diffusion_constant,
		    							waste_decay_constant,
		    							parameters.getMaximumVelocity(), 
		    							time),
    		waste);
	     old_waste = waste;

// output_chemicals();

	    time_step = get_CFL_time_step(); //0.01; // get from CFL condition...
	    do{
	    	assemble_full_system(false);
	    	solve();
// std::cout << "solved time step: " << time_step_number << std::endl;
	    	// old_old_public_goods = old_public_goods;
	    	// old_public_goods = public_goods;
	    	// old_old_waste = old_waste;
	    	// old_waste = waste; // part of solve...

	    	time += time_step;
	    	++time_step_number;

	    	if(cycle == n_cycles-1)
	    	{
		    	process_time_solution(
		    		ExactSolutions::GaussFull<dim>(good_diffusion_constant,
		    			good_decay_constant,
		    			parameters.getMaximumVelocity(), time),
		    		good_L_infty_error_VTIME,
		    		good_L2_error_VTIME,
		    		good_H1_error_VTIME,
		    		public_goods
		    		);

		    	process_time_solution(
		    		ExactSolutions::GaussFull<dim>(waste_diffusion_constant,
		    			waste_decay_constant,
		    			parameters.getMaximumVelocity(), time),
					waste_L_infty_error_VTIME,		
		    		waste_L2_error_VTIME,
		    		waste_H1_error_VTIME,
		    		waste
		    		);
		    	if( time_step_number % save_period == 0 )
		    	{
		    		++save_step_number;
		    		std::cout << "time: " << time << std::endl;
		    		output_chemicals();
		    	}
		    }
	    } while(time < run_time);
	    // *** ALSO DOUBLE CHECK EXACT SOLUTION...
	    if(cycle == n_cycles -1)
	    {
			output_vector(good_L_infty_error_VTIME,"good_L_infty_error");
			output_vector(good_L2_error_VTIME,"good_L2_error");
			output_vector(good_H1_error_VTIME,"good_H1_error");

			output_vector(waste_L_infty_error_VTIME,"waste_L_infty_error");
			output_vector(waste_L2_error_VTIME,"waste_L2_error");
			output_vector(waste_H1_error_VTIME,"waste_H1_error");
		}
	    process_solution(cycle, 
	    	good_convergence_table,
	    	ExactSolutions::GaussFull<dim>(good_diffusion_constant,
	    									parameters.getMaximumVelocity(),
	    									good_decay_constant,
	    									time),
	    	public_goods);
		process_solution(cycle,
			waste_convergence_table,
			ExactSolutions::GaussFull<dim>(waste_diffusion_constant,
	    									parameters.getMaximumVelocity(),
	    									waste_decay_constant,
	    									time),
			waste);
	} // for cycles

	output_convergence_table(good_convergence_table, "good_convergence_table");
	output_convergence_table(waste_convergence_table, "waste_convergence_table");

}


template<int dim>
void
DebugBootstrap<dim>::run_advection(const ArgParser& parameters)
{
	const unsigned int n_cycles = 6;
    
    setup_system_constants(parameters);
    setup_geometry(parameters);
    setup_advection(parameters);



	for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
	{
		time = 0;
		time_step_number = 0;
		save_step_number = 0;

	    // const double resolution = 1.0;
	    // const std::vector<Point<dim> > querry_points = geometry.getQuerryPoints(resolution);

	    ExactSolutions::GaussXVelocityPeriodic<dim> exactSolutionFunction(good_diffusion_constant,
	    		parameters.getMaximumVelocity(), time);

		triangulation.clear();
 		setup_grid(parameters, cycle +1);

 		const double distort_factor = 0.2;
		GridTools::distort_random(distort_factor, triangulation);

	    setup_dofs();
	    initialize_vectors_matrices();
	    assemble_matrices(); // mass and laplace matrices, solve advection explicitly

	    project_initial_condition(exactSolutionFunction);

	    time_step = get_CFL_time_step(); //0.01; // get from CFL condition...

	    // error over time:
	    // std::vector<double> L_infty_error_VTIME, L2_error_VTIME, H1_error_VTIME;

	    do{
	    	assemble_advection_system();
	    	solve();

	    	old_old_public_goods = old_public_goods;
	    	old_public_goods = public_goods;

	    	old_old_waste = old_waste;
	    	old_waste = waste;

	    	time += time_step;
	    	++time_step_number;

	    	// process_time_solution(
	    	// 	ExactSolutions::GaussXVelocityPeriodic<dim>(good_diffusion_constant,
	    	// 		parameters.getMaximumVelocity(), time),
	    	// 	L_infty_error_VTIME,
	    	// 	L2_error_VTIME,
	    	// 	H1_error_VTIME);

	    	if( (cycle == n_cycles-1) && (time_step_number % save_period == 0) )
	    	{
	    		++save_step_number;
	    		std::cout << "time: " << time << std::endl;
	    		output_chemicals();

	   //  		exactSolutionFunction.update_solution_time(time);
			 //    std::vector<double> exact_values(querry_points.size());
			 //    std::vector<double> interpolated_values(querry_points.size());

			 //    exactSolutionFunction.value_list(querry_points, exact_values); 
			    

			 //    for(unsigned int i = 0; i < querry_points.size(); i++)
			 //    {
			 //    	projectExactSolution(exactSolutionFunction);

			 //    	double value = dealii::VectorTools::point_value(dof_handler,
			 //    													 exact_solution,
			 //    													 querry_points[i]);
			 //    	interpolated_values[i] = value;
			 //    }

				// output_vector(exact_values, "exact_values-" 
				// 	+ Utilities::int_to_string(save_step_number,3) );
				// output_vector(interpolated_values, "interpolated_values-" 
				// 	+ Utilities::int_to_string(save_step_number,3));
	    	}
	    } while(time < run_time);
	    // *** ALSO DOUBLE CHECK EXACT SOLUTION...

		// output_vector(L_infty_error_VTIME,"L_infty_error");
		// output_vector(L2_error_VTIME,"L2_error");
		// output_vector(H1_error_VTIME,"H1_error");


	    process_solution(cycle, 
	    	ExactSolutions::GaussXVelocityPeriodic<dim>(good_diffusion_constant,
	    		parameters.getMaximumVelocity(),time));
	} // for cycles

	output_convergence_table();
}



template<int dim>
void
DebugBootstrap<dim>::run_full(const ArgParser& parameters)
{
	const unsigned int n_cycles = 6;
    
    setup_system_constants(parameters);
    setup_geometry(parameters);
    setup_advection(parameters);



	for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
	{
		time = 0;
		time_step_number = 0;
		save_step_number = 0;

	    // const double resolution = 1.0;
	    // const std::vector<Point<dim> > querry_points = geometry.getQuerryPoints(resolution);

	    ExactSolutions::GaussFull<dim> exactSolutionFunction(good_diffusion_constant,
	    		parameters.getMaximumVelocity(), good_decay_constant,time);

		triangulation.clear();
 		setup_grid(parameters, cycle +1);

 		const double distort_factor = 0.2;
		GridTools::distort_random(distort_factor, triangulation);

	    setup_dofs();
	    initialize_vectors_matrices();
	    assemble_matrices(); // mass and laplace matrices, solve advection explicitly

	    project_initial_condition(exactSolutionFunction);

	    time_step = get_CFL_time_step(); //0.01; // get from CFL condition...

	    // error over time:
	    // std::vector<double> L_infty_error_VTIME, L2_error_VTIME, H1_error_VTIME;

	    do{
	    	// assemble_advection_system();
	    	assemble_full_system();
	    	solve();

	    	old_old_public_goods = old_public_goods;
	    	old_public_goods = public_goods;

	    	time += time_step;
	    	++time_step_number;

	    	// process_time_solution(
	    	// 	ExactSolutions::GaussXVelocityPeriodic<dim>(good_diffusion_constant,
	    	// 		parameters.getMaximumVelocity(), time),
	    	// 	L_infty_error_VTIME,
	    	// 	L2_error_VTIME,
	    	// 	H1_error_VTIME);

	    	if( (cycle == n_cycles-1) && (time_step_number % save_period == 0) )
	    	{
	    		++save_step_number;
	    		std::cout << "time: " << time << std::endl;
	    		output_chemicals();

	   //  		exactSolutionFunction.update_solution_time(time);
			 //    std::vector<double> exact_values(querry_points.size());
			 //    std::vector<double> interpolated_values(querry_points.size());

			 //    exactSolutionFunction.value_list(querry_points, exact_values); 
			    

			 //    for(unsigned int i = 0; i < querry_points.size(); i++)
			 //    {
			 //    	projectExactSolution(exactSolutionFunction);

			 //    	double value = dealii::VectorTools::point_value(dof_handler,
			 //    													 exact_solution,
			 //    													 querry_points[i]);
			 //    	interpolated_values[i] = value;
			 //    }

				// output_vector(exact_values, "exact_values-" 
				// 	+ Utilities::int_to_string(save_step_number,3) );
				// output_vector(interpolated_values, "interpolated_values-" 
				// 	+ Utilities::int_to_string(save_step_number,3));
	    	}
	    } while(time < run_time);
	    // *** ALSO DOUBLE CHECK EXACT SOLUTION...

		// output_vector(L_infty_error_VTIME,"L_infty_error");
		// output_vector(L2_error_VTIME,"L2_error");
		// output_vector(H1_error_VTIME,"H1_error");


	    process_solution(cycle, 
	    	ExactSolutions::GaussFull<dim>(good_diffusion_constant,
	    		parameters.getMaximumVelocity(),good_decay_constant,time));
	} // for cycles

	output_convergence_table();
}


template<int dim>
void 
DebugBootstrap<dim>::run_assemble_bdf_point(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_advection(parameters);

	const unsigned int number_bacteria = 1;

	std::cout << "...Initializing bacteria" << std::endl;
	std::vector<Point<dim> > locations = getGaussGroup(number_bacteria); 

	bacteria.initialize(parameters.getBacteriaDiffusionConstant(), 
						number_bacteria,
						locations,
						parameters.getGoodSecretionRate(),
						parameters.getWasteSecretionRate() );

	std::cout << "...Setting fitness constants" << std::endl;
	bacteria.setFitnessConstants(parameters.getAlphaGood(),
								parameters.getAlphaWaste(),
								parameters.getGoodSaturation(),
								parameters.getWasteSaturation(),
								parameters.getSecretionCost() );
    bacteria.printInfo(std::cout);

    // set up decay constant:
    good_diffusion_constant = 0;
    good_decay_constant = 0;
    good_decay_constant = 0; //good_decay[i];
    waste_decay_constant = 0; //waste_decay[i];

	setup_grid(parameters);
	setup_dofs();

	time_step = 0.001;
	save_period = 100;
	run_time = 1.0;
	time_step_number = 0;
	save_step_number = 0;
	time = 0;
	
	initialize_vectors_matrices();
	assemble_matrices();

	assemble_full_system(true); // include point sources
	solve();
	output_chemicals();

}


template<int dim>
void 
DebugBootstrap<dim>::run_assemble_point(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_advection(parameters);

	const unsigned int number_bacteria = 1;

	std::cout << "...Initializing bacteria" << std::endl;
	std::vector<Point<dim> > locations = getGaussGroup(number_bacteria); 

	bacteria.initialize(parameters.getBacteriaDiffusionConstant(), 
						number_bacteria,
						locations,
						parameters.getGoodSecretionRate(),
						parameters.getWasteSecretionRate() );

	std::cout << "...Setting fitness constants" << std::endl;
	bacteria.setFitnessConstants(parameters.getAlphaGood(),
								parameters.getAlphaWaste(),
								parameters.getGoodSaturation(),
								parameters.getWasteSaturation(),
								parameters.getSecretionCost() );
    bacteria.printInfo(std::cout);

    // set up decay constant:
    good_diffusion_constant = 0;
    good_decay_constant = 0;
    good_decay_constant = 0; //good_decay[i];
    waste_decay_constant = 0; //waste_decay[i];

	setup_grid(parameters);
	setup_dofs();

	time_step = 0.001;
	save_period = 100;
	run_time = 1.0;
	time_step_number = 0;
	save_step_number = 0;
	time = 0;
	
	initialize_vectors_matrices();
	assemble_matrices();

	assemble_simple_system(); // include point sources
	solve();

	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(public_goods,"public_goods");
		data_out.build_patches();
		const std::string filename = output_directory
									+ "/simple_public_goods-"
									+ Utilities::int_to_string(save_step_number,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(waste,"waste");
		data_out.build_patches();
		const std::string filename = output_directory
									+ "/simple_waste-"
									+ Utilities::int_to_string(save_step_number,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}

}


template<int dim>
void
DebugBootstrap<dim>::run_viscosity(const ArgParser& parameters)
{
	setup_system(parameters);

	project_initial_condition(ExactSolutions::GaussNeumann<dim>(0.1, 0., 0.),
								public_goods);
	old_public_goods = public_goods;

	project_initial_condition(ExactSolutions::GaussNeumann<dim>(0.1, 0., 0.),
								waste);
	old_waste = waste;

	do{
		assemble_full_system(false);
		solve();

    	time += time_step;
    	++time_step_number;
    	
    	if( time_step_number % save_period == 0 )
    	{
    		++save_step_number;
    		std::cout << "time: " << time << std::endl;
    		output_chemicals();
    	}

	} while(time < run_time);
}


template<int dim>
void
DebugBootstrap<dim>::run_bacteria(const ArgParser& parameters)
{

	std::cout << "running with bacteria " << std::endl;
	setup_system(parameters);

	do{
		assemble_full_system(true);
		solve();
		time += time_step;
    	++time_step_number;

   //  	// update bacteria:
	    if(time_step_number > reproduction_delay)
			bacteria.reproduce(time_step, dof_handler, public_goods, waste);
//			std::cout << "now with " << bacteria.getSize() << " bacteria" << std::endl;
		// bacteria.mutateBacteria(mutation_rate,mutation_diff);
		bacteria.randomWalk(time_step, &geometry, &advection_field);

    	if( time_step_number % save_period == 0 )
    	{
    		++save_step_number;
    		std::cout << "time: " << time << std::endl;
			std::cout << "\tnow with " << bacteria.getSize() << " bacteria\n" << std::endl;
    		output_chemicals();
    		output_bacteria();
    	}

	} while(time < run_time);
}


template<int dim>
void 
DebugBootstrap<dim>::run_test_grid(const ArgParser& parameters)
{

	setup_system_constants(parameters);
	setup_geometry(parameters);


	// setup_grid(parameters);
	std::cout << "...Setting up grid" << std::endl;
	MyGridGenerator<dim> gridGen;
	gridGen.generateGrid(geometry,triangulation); // attach manifold

	refine_grid(2,0);
	output_grid(); // non-refined

	// out put successive refinements:
	for(unsigned int i = 0; i < 5; i++)
	{
		refine_grid(0, 1);

		std::string grid_out_file = output_directory 
									+ "/grid-" 
									+ Utilities::int_to_string(i,2)
									+ ".eps";

		std::ofstream out (grid_out_file);
		GridOut grid_out;
		grid_out.write_eps (triangulation, out);
		std::cout << "...Grid written to " << grid_out_file << std::endl;
	}
}





// close namespaces:
}}

#endif // MicrobeBDF2.h




