#ifndef MICROBE_SIMULATOR_SIMULATOR_BDF2_H
#define MICROBE_SIMULATOR_SIMULATOR_BDF2_H


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
// #include "../bacteria/fitness.h"
// #include "../chemicals/chemicals.h"
// #include "../sources/sources.h"

#include "../utility/argparser.h"
#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "../bacteria/bacteria.h"
#include "../utility/my_grid_generator.h"


namespace MicrobeSimulator{
	using namespace dealii;

	template<int dim>
	class ChemicalInitialCondition : public Function<dim>{
	public:
		ChemicalInitialCondition() : Function<dim>(1), center() {}
		ChemicalInitialCondition(const Point<dim> & c) : Function<dim>(1), center(c) {}

		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

		virtual void vector_value(const Point<dim> &p, Vector<double>   &values) const;

	private:
		Point<dim> center;
	};

	template<int dim>
	double ChemicalInitialCondition<dim>::value(const Point<dim> &p,
										const unsigned int component) const
	{
		(void) component;
		Assert(component == 0,
			ExcMessage("Invalid operation for a scalar function."));
	    Assert ((dim==2) || (dim==3), ExcNotImplemented());
	
		return exp( -(p-center)*(p-center) ); 
	} 

  template <int dim>
  void
  ChemicalInitialCondition<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = ChemicalInitialCondition<dim>::value (p, c);
  }







	template<int dim>
	class ChemicalRightHandSide : public Function<dim>{
	public:
		ChemicalRightHandSide() : Function<dim>(1), center() {}
		ChemicalRightHandSide(const Point<dim> & c) : Function<dim>(1), center(c) {}

		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

		virtual void vector_value(const Point<dim> &p, Vector<double>   &values) const;

	private:
		Point<dim> center;
	};

	template<int dim>
	double ChemicalRightHandSide<dim>::value(const Point<dim> &p,
										const unsigned int component) const
	{
		(void) component;
		Assert(component == 0,
			ExcMessage("Invalid operation for a scalar function."));
	    Assert ((dim==2) || (dim==3), ExcNotImplemented());
	
		return 0; //exp( -(p-center)*(p-center) ); 
	} 

  template <int dim>
  void
  ChemicalRightHandSide<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = ChemicalRightHandSide<dim>::value (p, c);
  }


	template<int dim>
	class Simulator_BDF2{
	public:
		// enum -- chemical type ? ... or polymorphism
		Simulator_BDF2();
		// void run();

		void run(const ArgParser& systemParameters);
		void run_debug(const ArgParser& systemParameters);

		void run_bdf2_test(const ArgParser& systemParameters);

	private:
		Geometry<dim> geometry;
		AdvectionField<dim> advectionField;
		Bacteria<dim> bacteria;

		// FINITE ELEMENT CHEMICALS:
  		double             global_Omega_diameter;
		const unsigned int chemical_fe_degree;

	    Triangulation<dim>   triangulation;
	    FE_Q<dim>            fe;
	    DoFHandler<dim>      dof_handler;

	    ConstraintMatrix     constraints;

	    SparsityPattern      sparsity_pattern;

	    SparseMatrix<double> mass_matrix; 
	    SparseMatrix<double> stiffness_matrix;

	    SparseMatrix<double> advection_matrix; 

	    SparseMatrix<double> system_matrix1;

	    Vector<double>       solution1;
	    Vector<double>       old_solution1;
	    Vector<double> 		 old_old_solution1;

	    Vector<double>       system_rhs1;
	    Vector<double> 		 tmp1; 
	    Vector<double> 		 temporary_one;



	    SparseMatrix<double> system_matrix2;
	    Vector<double>       solution2;
	    Vector<double>       old_solution2;
	    // Vector<double>       old_old_solution2;
	    Vector<double>       system_rhs2;

	    Vector<double>       tmp2; // to store rhs
	    Vector<double>       temporary_two;


        double diffusion_constant1;
	    double diffusion_constant2;
	    double decay_constant1;
	    double decay_constant2;

	    double time; // t
	    double time_step; // dt
	    double old_time_step; // usually should equal dt ...
	    unsigned int timeStepNumber; // n
		const unsigned int reproduction_delay;

		unsigned int save_period;
		unsigned int saveStepNumber;
		double run_time;

		std::string outputDirectory;


		//-----------------------------------------------------------------------------
		// stabilized BDF-2:
		void setup_base_objects(const ArgParser& systemParameters);
		void setup_dofs();
		void assemble_system_matrices();

    	double get_maximal_velocity () const;


		void assemble_chemical_system(const double maximal_velocity);

		std::pair<double,double> get_extrapolated_chemical_range () const;
		double compute_viscosity(
								const std::vector<double>&			old_chemical,
								const std::vector<double>& 			old_old_chemical,
								const std::vector<Tensor<1,dim> >&	old_chemical_grads,
								const std::vector<Tensor<1,dim> >&	old_old_chemical_grads,
								const std::vector<double>&			old_chemical_laplacians,
								const std::vector<double>&			old_old_chemical_laplacians,
								const std::vector<Tensor<1,dim> >&	old_velocity_values,
								// const std::vector<Tensor<1,dim> >&	old_old_velocity_values,
								const std::vector<double>&			gamma_values,
								const double 						global_u_infty,
								const double						global_chemical_variation,
								const double						cell_diameter
								) const;
		void solve_bdf2();
		void output_chemical_one();


		//-----------------------------------------------------------------------------
    	// EXTENSIONS ....

		// Chemicals<dim,NumberChemicals> chemicals; 
		// Sources<dim,NumberChemicals> sourceFunctions;
		// Fitness<dim,NumberChemicals> fitnessFunction;

		// METHODS:
  		void setup_system(const ArgParser& systemParameters);

  		void initialize_base_objects(const ArgParser& systemParameters);
  		void setup_constraints();
  		void initialize_vectors_matrices();

  		std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, 
  			unsigned int number_groups);
  		std::vector<Point<dim> > getMixerLocations(unsigned int number_groups);

	    void create_advection_matrix();

	    void update_chemicals();
	    void solve_time_step();
	    void adv_solve_time_step();
        void output_results() const;
        void output_bacteria() const; 

        void createGrid(unsigned int global_refinement, 
        	unsigned int sphere_refinement);
	    void refineMeshSpheres(unsigned int sphere_refinement);
	    void outputGrid();

        void updateMatricesWithPointSources();
    	void updatePointSources();

    	//DEBUGGING:
        Vector<double>       mass_check_vector; 

  		void setup_system_debug(const ArgParser& systemParameters);
  		void initialize_base_objects_debug(const ArgParser& systemParameters);

	    void create_mass_check_vector();
	    void printMassToFile(std::vector<double> massOne, 
    		std::vector<double> massTwo) const;
    	void projectInitialCondition(Point<dim> gaussian_center);
    	Point<dim> findGaussianCenterPoint() const;
    	void updateMatrices(bool usePointSources);

	};


// BDF-2 STUFF:
//--------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------

template<int dim>
void 
Simulator_BDF2<dim>::setup_base_objects(const ArgParser& systemParameters)
{
	time_step = systemParameters.getTimeStep();
	run_time = systemParameters.getRunTime();
	save_period = systemParameters.getSavePeriod();
	diffusion_constant1 = systemParameters.getGoodDiffusionConstant();
	diffusion_constant2 = systemParameters.getWasteDiffusionConstant();
	decay_constant1 = systemParameters.getGoodDecayConstant();
	decay_constant2 = systemParameters.getWasteDecayConstant();
	outputDirectory = systemParameters.getOutputDirectory();

	std::cout << "...Initializing geometry" << std::endl;
	geometry.initialize(systemParameters.getGeometryFile(), systemParameters.getMeshFile());
	std::string geo_out_file = outputDirectory + "/geometryInfo.dat";
	std::ofstream geo_out(geo_out_file);
	geometry.printInfo(geo_out);
	geometry.printInfo(std::cout);

	std::cout << "...Initializing advection" << std::endl;
	if( (systemParameters.getVelocityType() != VelocityType::NUMERICAL_FLOW)
		&& (systemParameters.getVelocityType() != VelocityType::TILE_FLOW) )
	{
		advectionField.initialize(systemParameters.getVelocityType(),
								geometry.getBottomLeftPoint(),
								geometry.getTopRightPoint(),
								systemParameters.getMaximumVelocity()
								);
						//  double vrad = 0, double vrotation = 0);
	}
	else
	{
		advectionField.initialize(systemParameters.getVelocityType(),
								geometry.getBottomLeftPoint(),
								geometry.getTopRightPoint(),
								geometry.getScaleFactor(),
								systemParameters.getMaximumVelocity(),
								systemParameters.getVelocityFile_X(),
								systemParameters.getVelocityFile_Y() );
	}

	advectionField.printInfo(std::cout);

	std::cout << "...Creating grid" << std::endl;
	createGrid(systemParameters.getGlobalRefinement(), systemParameters.getSphereRefinement());
	outputGrid();
}


template<int dim>
void 
Simulator_BDF2<dim>::setup_dofs()
{
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

// ------------------------- may be different...

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
									dsp,
									constraints,
									/*keep constrained_dofs = */ false); // *** not sure if false or true...

	sparsity_pattern.copy_from(dsp);

	mass_matrix.reinit(sparsity_pattern);
	stiffness_matrix.reinit(sparsity_pattern);
	system_matrix1.reinit(sparsity_pattern);


	solution1.reinit(dof_handler.n_dofs());
	old_solution1.reinit(dof_handler.n_dofs());
	old_old_solution1.reinit(dof_handler.n_dofs());
	system_rhs1.reinit(dof_handler.n_dofs());
	tmp1.reinit(dof_handler.n_dofs());
	temporary_one.reinit(dof_handler.n_dofs());
}


template<int dim>
void
Simulator_BDF2<dim>::assemble_system_matrices()
{
	mass_matrix = 0;
	stiffness_matrix = 0;

	QGauss<dim> quadrature_formula(chemical_fe_degree + 2);
	FEValues<dim> chemical_fe_values(fe, quadrature_formula,
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

		chemical_fe_values.reinit(cell);

		for(unsigned int q = 0; q < n_q_points; ++q)
		{
			for(unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_T[k] = chemical_fe_values.shape_grad(k,q);
				phi_T[k] = chemical_fe_values.shape_value(k,q);
			}

			for(unsigned int i = 0; i < dofs_per_cell; ++i)
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
				{
					local_mass_matrix(i,j) += (phi_T[i] * phi_T[j]
						*
						chemical_fe_values.JxW(q));

					local_stiffness_matrix(i,j) += (diffusion_constant1
						*grad_phi_T[i] * grad_phi_T[j]
						*
						chemical_fe_values.JxW(q));
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
} // build matrices


template<int dim>
double
Simulator_BDF2<dim>::get_maximal_velocity() const
{
	const QIterated<dim> quadrature_formula(QTrapez<1>(), chemical_fe_degree+1);
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
		advectionField.value_list(fe_values.get_quadrature_points(), velocity_values);

		for(unsigned int q = 0; q < n_q_points; ++q)
			max_velocity = std::max(max_velocity, velocity_values[q].norm());
	}

	return max_velocity;
}



template<int dim>
void Simulator_BDF2<dim>::assemble_chemical_system(const double maximal_velocity)
{
	const bool use_bdf2_scheme = (timeStepNumber != 0);

	if(use_bdf2_scheme == true)
	{
		system_matrix1.copy_from(mass_matrix);
		system_matrix1 *= (2*time_step + old_time_step) /
							(time_step + old_time_step); // ***need to add decay term...
		system_matrix1.add(time_step, stiffness_matrix);
	}
	else
	{
		system_matrix1.copy_from(mass_matrix);
		system_matrix1.add(time_step,stiffness_matrix);
	}

	system_rhs1 = 0;

	const QGauss<dim> quadrature_formula(chemical_fe_degree + 2);
	FEValues<dim> chemical_fe_values(fe,quadrature_formula,                                           
										   update_values    |
                                           update_gradients |
                                           update_hessians  |
                                           update_quadrature_points  |
                                           update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	Vector<double> 	local_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// solution values:
	std::vector<Tensor<1, dim> > 	old_velocity_values(n_q_points); // ***this is a constant!! -- need only find once
	std::vector<double> 			old_chemical_values(n_q_points);
	std::vector<double> 			old_old_chemical_values(n_q_points);
	std::vector<Tensor<1,dim> >		old_chemical_grads(n_q_points);
	std::vector<Tensor<1,dim> >		old_old_chemical_grads(n_q_points);
	std::vector<double>				old_chemical_laplacians(n_q_points);
	std::vector<double>				old_old_chemical_laplacians(n_q_points);

	ChemicalRightHandSide<dim> chemical_right_hand_side; // ***gaussian at origin
	std::vector<double> 	gamma_values(n_q_points);

	std::vector<double>				phi_T(dofs_per_cell);
	std::vector<Tensor<1,dim> >		grad_phi_T(dofs_per_cell);

	const std::pair<double,double> global_chem_range = get_extrapolated_chemical_range();

	// const FEValuesExtractors::Vector velocities(0); // used for stokes ... ** can use later!!

	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
													endc = dof_handler.end();

	for(; cell!=endc; ++cell)
	{
		local_rhs = 0;
		chemical_fe_values.reinit(cell);
		
		chemical_fe_values.get_function_values(old_solution1, old_chemical_values);
		chemical_fe_values.get_function_values(old_old_solution1, old_old_chemical_values);

		chemical_fe_values.get_function_gradients(old_solution1, old_chemical_grads);
		chemical_fe_values.get_function_gradients(old_old_solution1, old_old_chemical_grads);

		chemical_fe_values.get_function_laplacians(old_solution1, old_chemical_laplacians);
		chemical_fe_values.get_function_laplacians(old_old_solution1, old_old_chemical_laplacians);
			
		chemical_right_hand_side.value_list(chemical_fe_values.get_quadrature_points(), gamma_values);

		advectionField.value_list(chemical_fe_values.get_quadrature_points(), 
				old_velocity_values); 

		// get maximum of old velocity values:
		// double min_vel = old_velocity_values[0].norm();
		// double max_vel = old_velocity_values[0].norm();

		// for(unsigned int i = 0; i < old_velocity_values.size(); i++)
		// {
		// 	min_vel = std::min<double> (min_vel, old_velocity_values[i].norm());
		// 	max_vel = std::max<double> (max_vel, old_velocity_values[i].norm());
		// }

		// std::cout << "VELOCITY RANGE: "
		// 	<< min_vel << " " << max_vel
		// 	<< std::endl;

		// compute viscosity:
		const double nu 
			= compute_viscosity(old_chemical_values,
								old_old_chemical_values,
								old_chemical_grads,
								old_old_chemical_grads,
								old_chemical_laplacians,
								old_old_chemical_laplacians,
								old_velocity_values,
								gamma_values,
								maximal_velocity,
								global_chem_range.second - global_chem_range.first,
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
	               (old_chemical_values[q] *
	                (1 + time_step/old_time_step)
	                -
	                old_old_chemical_values[q] *
	                (time_step * time_step) /
	                (old_time_step * (time_step + old_time_step)))
	               :
	               old_chemical_values[q]);

		    const Tensor<1,dim> ext_grad_T
	            = (use_bdf2_scheme ?
	               (old_chemical_grads[q] *
	                (1 + time_step/old_time_step)
	                -
	                old_old_chemical_grads[q] *
	                time_step/old_time_step)
	               :
	               old_chemical_grads[q]);

	        const Tensor<1,dim> extrapolated_u = old_velocity_values[q];
	            // = (use_bdf2_scheme ?
	            //    (old_velocity_values[q] *
	            //     (1 + time_step/old_time_step)
	            //     -
	            //     old_old_velocity_values[q] *
	            //     time_step/old_time_step)
	            //    :
	            //    old_velocity_values[q]);

	        for(unsigned int i = 0; i < dofs_per_cell; ++i)
	        	local_rhs(i) +=  (T_term_for_rhs * phi_T[i]
	                             -
	                             time_step *
	                             extrapolated_u * ext_grad_T * phi_T[i]
	                             -
	                             time_step *
	                             nu * ext_grad_T * grad_phi_T[i]
	                             +
	                             time_step *
	                             gamma_values[q] * phi_T[i])
	                            *
	                            chemical_fe_values.JxW(q);
		}

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_rhs,
                                              local_dof_indices,
                                              system_rhs1);
    }
} // assemble chemical system


template<int dim>
std::pair<double,double> 
Simulator_BDF2<dim>::get_extrapolated_chemical_range () const
{
	const QIterated<dim> quadrature_formula(QTrapez<1>(),
											chemical_fe_degree);
	const unsigned int n_q_points = quadrature_formula.size();

	FEValues<dim> fe_values(fe, quadrature_formula, update_values);

	std::vector<double> old_chemical_values(n_q_points);
	std::vector<double> old_old_chemical_values(n_q_points);

	if(timeStepNumber != 0)
	{
		double min_chemical = std::numeric_limits<double>::max(),
				max_chemical = -std::numeric_limits<double>::max();

		typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();

		for(; cell != endc; ++cell)
		{
			fe_values.reinit(cell);
			fe_values.get_function_values(old_solution1,
											old_chemical_values);
			fe_values.get_function_values(old_old_solution1,
											old_old_chemical_values);

			for(unsigned int q = 0; q < n_q_points; ++q)
			{
				const double chemical =                 
					(1. + time_step/old_time_step) * old_chemical_values[q]-
	                time_step/old_time_step * old_old_chemical_values[q];
	              min_chemical = std::min (min_chemical, chemical);
	              max_chemical = std::max (max_chemical, chemical);
            }
		}
        return std::make_pair(min_chemical,max_chemical);
	}
	else
	{
		double min_chemical = std::numeric_limits<double>::max(),
				max_chemical = -std::numeric_limits<double>::max();

		typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();

		for(; cell != endc; ++cell)
		{
			fe_values.reinit(cell);
	         fe_values.get_function_values (old_solution1,
	                                         old_chemical_values);
	          for (unsigned int q=0; q < n_q_points; ++q)
	            {
	              const double chemical = old_chemical_values[q];
	              min_chemical = std::min (min_chemical, chemical);
	              max_chemical = std::max (max_chemical, chemical);
	            }
	    }
    	return std::make_pair(min_chemical,max_chemical);
	} // else

} // get range



template<int dim>
double 
Simulator_BDF2<dim>::compute_viscosity(
						const std::vector<double>&			old_chemical,
						const std::vector<double>& 			old_old_chemical,
						const std::vector<Tensor<1,dim> >&	old_chemical_grads,
						const std::vector<Tensor<1,dim> >&	old_old_chemical_grads,
						const std::vector<double>&			old_chemical_laplacians,
						const std::vector<double>&			old_old_chemical_laplacians,
						const std::vector<Tensor<1,dim> >&	old_velocity_values,
						// const std::vector<Tensor<1,dim> >&	old_old_velocity_values,
						const std::vector<double>&			gamma_values,
						const double 						global_u_infty,
						const double						global_chemical_variation,
						const double						cell_diameter
						) const
{
	const double beta = 0.017 * dim; // *** heuristic
	const double alpha = 1; // can be in range 1 to 2

	if(global_u_infty == 0)
		return 5e-3 * cell_diameter;

	const unsigned int n_q_points = old_chemical.size();

	double max_residual = 0;
	double max_velocity = 0;

	for(unsigned int q = 0; q < n_q_points; ++q)
	{
		const Tensor<1,dim> u = old_velocity_values[q];

		const double dT_dt = (old_chemical[q] - old_old_chemical[q])/old_time_step;

		const double u_grad_T = u*(old_chemical_grads[q] + old_old_chemical_grads[q]) /2;

		const double kappa_Delta_T = diffusion_constant1
						* (old_chemical_laplacians[q] + old_old_chemical_laplacians[q]) /2;

		const double residual 
			= std::abs( (dT_dt + u_grad_T - kappa_Delta_T - gamma_values[q])*
				std::pow((old_chemical[q] + old_old_chemical[q]) /2,
					alpha-1.) );

		max_residual = std::max(residual, max_residual);
		max_velocity = std::max(std::sqrt( u*u ), max_velocity);
	}

	const double c_R = std::pow(2., (4. - 2*alpha)/dim);
	const double global_scaling = c_R * global_u_infty *global_chemical_variation *
					std::pow(global_Omega_diameter, alpha - 2.);

	return (beta*
			max_velocity *
			std::min( cell_diameter,
						std::pow(cell_diameter,alpha) *
						max_residual /global_scaling));
}


template<int dim>
void 
Simulator_BDF2<dim>::solve_bdf2()
{
 // get time step from velocity and CFL condition, do this in setup since velocity is constant

	const double maximal_velocity = get_maximal_velocity();
	double cfl_time_step = 0;

	// std::cout << "Maximal velocity: " << maximal_velocity << std::endl;

	if(maximal_velocity >= 0.01)
		cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
			chemical_fe_degree *
			GridTools::minimal_cell_diameter(triangulation) /
			maximal_velocity;
	else
		cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
			chemical_fe_degree *
			GridTools::minimal_cell_diameter(triangulation) /
			0.01;

	// std::cout << "\n\tCFL TIME STEP: " << cfl_time_step
	// 	<< std::endl;

	solution1 = old_solution1;

	assemble_chemical_system(maximal_velocity);

	// solve using CG ...
  SolverControl solver_control(1000, 1e-8 * system_rhs1.l2_norm());
  SolverCG<> cg(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix1, 1.0);
  cg.solve(system_matrix1, solution1, system_rhs1,
           preconditioner);
  constraints.distribute(solution1);

  // std::cout << "     " << solver_control.last_step()
  //           << " CG iterations." << std::endl;

	// get value range:
    double min_chemical = solution1(0),
       max_chemical = solution1(0);
    
    for (unsigned int i=0; i< solution1.size(); ++i)
	{
		min_chemical = std::min<double> (min_chemical,
	                                    solution1(i));
		max_chemical = std::max<double> (max_chemical,
	                                    solution1(i));
	}
	// std::cout << "   Chemical range: "
	//       << min_chemical << ' ' << max_chemical
	//       << std::endl;

}

template<int dim>
void 
Simulator_BDF2<dim>::output_chemical_one()
{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);

	data_out.add_data_vector(solution1, "C1");

	data_out.build_patches();

	const std::string filename = outputDirectory
								 + "/solutionC"
	                             + Utilities::int_to_string(1,2) 
	                             + "-"
	                             + Utilities::int_to_string(saveStepNumber, 4) +
	                             ".vtk";
	std::ofstream output(filename.c_str());
	data_out.write_vtk(output);
}


// RUN TEST:


template<int dim>
void
Simulator_BDF2<dim>::run_bdf2_test(const ArgParser& systemParameters)
{
	// const unsigned int initial_refinement = 4;

    // GridGenerator::hyper_cube (triangulation);
    // global_Omega_diameter = GridTools::diameter (triangulation);
    // triangulation.refine_global (initial_refinement);

    setup_base_objects(systemParameters);
    setup_dofs();

	create_mass_check_vector();

	const unsigned int numtime = 
		ceil( systemParameters.getRunTime() / systemParameters.getTimeStep() );
	std::vector<double> total_mass_one, total_mass_two;
	total_mass_one.reserve(numtime);
	total_mass_two.reserve(numtime); 

    std::cout << "diffusion_constant1 = " << diffusion_constant1 << std::endl;

    // print velocity:
    std::string adv_file = outputDirectory + "/advection_field.dat";
    std::ofstream adv_out(adv_file);
    advectionField.print(geometry.getQuerryPoints(), adv_out);

	// initial condition:
	 VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(chemical_fe_degree+2),
	                      ChemicalInitialCondition<dim>(Point<2>(0.2,3)),
	                      old_solution1);
	
	assemble_system_matrices();


	 time_step = old_time_step = 0.0005;

 	do{
		solve_bdf2(); // includes -> assemble_chemical_system(get_maximal_velocity());    // update RHS
	
		time += time_step;
		++timeStepNumber;

		old_old_solution1 = old_solution1;
		old_solution1 = solution1;

		total_mass_one.push_back( solution1*mass_check_vector );
		// total_mass_two.push_back( solution2*mass_check_vector );

		if(timeStepNumber % save_period == 0)
		{
			++ saveStepNumber;
			std::cout << "time: " << time << std::endl;
			output_chemical_one();
		}
	} while(time <= run_time); 

	printMassToFile(total_mass_one, total_mass_two);
} // run test

	
















































//IMPLEMENTATION
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

	template<int dim>
	Simulator_BDF2<dim>::Simulator_BDF2()
	    :
        global_Omega_diameter (std::numeric_limits<double>::quiet_NaN()),
	    chemical_fe_degree(2),
	    fe(chemical_fe_degree),
	    dof_handler(triangulation),
	    time(0),
	    timeStepNumber(0),
	    reproduction_delay(5),
	    saveStepNumber(0),
	    outputDirectory("./")
	{}
	// 	:
	// 	chemicals(parameters)
	// 	// geometry(systemParameters),
	// 	// advectionField(&geometry, systemParameters),
	// 	// chemicals(NumberChemicals, systemParameters),
	// 	// bacteria(&geometry, &advectionField, systemParameters),
	// 	// sourceFunctions(&bacteria),
	// 	// fitnessFunction(&chemicals, systemParameters)
	// {}
	

	template<int dim>
	void Simulator_BDF2<dim>::setup_system(const ArgParser& systemParameters)
	{
  		initialize_base_objects(systemParameters);
  		setup_constraints();
  		initialize_vectors_matrices();
	} 


	template<int dim>
	void Simulator_BDF2<dim>::initialize_base_objects(const ArgParser& systemParameters)
	{
		time_step = systemParameters.getTimeStep();
		run_time = systemParameters.getRunTime();
		save_period = systemParameters.getSavePeriod();
		diffusion_constant1 = systemParameters.getGoodDiffusionConstant();
		diffusion_constant2 = systemParameters.getWasteDiffusionConstant();
		decay_constant1 = systemParameters.getGoodDecayConstant();
		decay_constant2 = systemParameters.getWasteDecayConstant();
		outputDirectory = systemParameters.getOutputDirectory();

		std::cout << "...Initializing geometry" << std::endl;
		geometry.initialize(systemParameters.getGeometryFile(), systemParameters.getMeshFile());
		std::string geo_out_file = outputDirectory + "/geometryInfo.dat";
		std::ofstream geo_out(geo_out_file);
		geometry.printInfo(geo_out);
		geometry.printInfo(std::cout);

		std::cout << "...Initializing advection" << std::endl;
		if( (systemParameters.getVelocityType() != VelocityType::NUMERICAL_FLOW)
			&& (systemParameters.getVelocityType() != VelocityType::TILE_FLOW) )
		{
			advectionField.initialize(systemParameters.getVelocityType(),
									geometry.getBottomLeftPoint(),
									geometry.getTopRightPoint(),
									systemParameters.getMaximumVelocity()
									);
							//  double vrad = 0, double vrotation = 0);
		}
		else
		{
			advectionField.initialize(systemParameters.getVelocityType(),
									geometry.getBottomLeftPoint(),
									geometry.getTopRightPoint(),
									geometry.getScaleFactor(),
									systemParameters.getMaximumVelocity(),
									systemParameters.getVelocityFile_X(),
									systemParameters.getVelocityFile_Y() );
		}

		advectionField.printInfo(std::cout);

		// get bacteria locations:
		std::cout << "...Initializing bacteria" << std::endl;
		std::vector<Point<dim> > locations ={Point<2>(2,2)}; //, Point<2>(2,4)};

		// if(geometry.getMeshType() == MeshType::MIXER)
			// locations = getMixerLocations(systemParameters.getNumberGroups());
		// else
			// locations = getBacteriaLocations(systemParameters.getNumberBacteria(), 
			// 	systemParameters.getNumberGroups());
		// *** doesn't work if mixer from file mesh

		bacteria.initialize(systemParameters.getBacteriaDiffusionConstant(), 
			systemParameters.getNumberBacteria(),
			locations,
			systemParameters.getGoodSecretionRate(),
			 systemParameters.getWasteSecretionRate() );

		std::cout << "...Setting fitness constants" << std::endl;
		bacteria.setFitnessConstants(systemParameters.getAlphaGood(),
									systemParameters.getAlphaWaste(),
									systemParameters.getGoodSaturation(),
									systemParameters.getWasteSaturation(),
									systemParameters.getSecretionCost() );

		std::cout << "...Creating grid" << std::endl;
		createGrid(systemParameters.getGlobalRefinement(), systemParameters.getSphereRefinement());
		outputGrid();

	}
	
	template<int dim>
	void Simulator_BDF2<dim>::setup_constraints()
	{
		dof_handler.distribute_dofs(fe);

		std::cout << std::endl << std::endl
		      << "================================================"
		      << std::endl
		      << "Number of active cells: " << triangulation.n_active_cells()
		      << std::endl
		      << "Number of degrees of freedom: " << dof_handler.n_dofs()
		      << std::endl
		      << std::endl;

		constraints.clear ();


		// MAKE CONSTRAINTS
		DoFTools::make_hanging_node_constraints (dof_handler, constraints);
		// ADD PERIODICITY IF ASKING ... NEED TO CONFIRM MESH BOUNDARY IDS...
		// if(systemParameters.getXBoundary() == Geometry::wrap)
		// {

//		std::array<BoundaryCondition, dim> boundaryConditions = geometry.getBoundaryConditions();
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
	void Simulator_BDF2<dim>::initialize_vectors_matrices()
	{

		DynamicSparsityPattern dsp(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler,
		                              dsp,
		                              constraints,
		                              /*keep_constrained_dofs = */ true);

		sparsity_pattern.copy_from(dsp);

		mass_matrix.reinit(sparsity_pattern);
		stiffness_matrix.reinit(sparsity_pattern);
		advection_matrix.reinit(sparsity_pattern);
		// robin_matrix.reinit(sparsity_pattern);

		// rhs_matrix1.reinit(sparsity_pattern);
		// rhs_matrix2.reinit(sparsity_pattern);
		system_matrix1.reinit(sparsity_pattern);
  		system_matrix2.reinit(sparsity_pattern);

		MatrixCreator::create_mass_matrix(dof_handler,
		                                QGauss<dim>(fe.degree+1),
		                                mass_matrix);
		MatrixCreator::create_laplace_matrix(dof_handler,
		                                   QGauss<dim>(fe.degree+1),
		                                   stiffness_matrix);

		if(advectionField.getVelocityType() != VelocityType::NO_FLOW)
		{
			std::cout << "\n...CREATING ADVECTION MATRIX\n" << std::endl;
			create_advection_matrix();
		}


		solution1.reinit(dof_handler.n_dofs());
		old_solution1.reinit(dof_handler.n_dofs());
		system_rhs1.reinit(dof_handler.n_dofs());
		tmp1.reinit(dof_handler.n_dofs());
		temporary_one.reinit(dof_handler.n_dofs());
		

		solution2.reinit(dof_handler.n_dofs());
		old_solution2.reinit(dof_handler.n_dofs());
		system_rhs2.reinit(dof_handler.n_dofs());
		tmp2.reinit(dof_handler.n_dofs());
		temporary_two.reinit(dof_handler.n_dofs());
	}



	template<int dim>
	std::vector<Point<dim> > Simulator_BDF2<dim>::getBacteriaLocations(unsigned int number_bacteria, 
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
	std::vector<Point<dim> > Simulator_BDF2<dim>::getMixerLocations(unsigned int number_groups)
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
	void Simulator_BDF2<dim>::create_advection_matrix()
	{
	  const QGauss<dim> quad((fe.degree+1));
	  const QGauss<dim-1> face_quad((fe.degree+1));

	  FEValues<dim> fe_values(fe, quad,
	        update_values | update_gradients | update_quadrature_points | update_JxW_values);
	  FEFaceValues<dim> fe_face_values(fe, face_quad,
	        update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

	  const unsigned int dofs_per_cell = fe.dofs_per_cell;
	  const unsigned int n_q_points = quad.size();
	  // const unsigned int n_face_q_points = face_quad.size();


	  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	  // FullMatrix<double> robin_cell_matrix(dofs_per_cell, dofs_per_cell);

	  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();

	  for(; cell != endc; ++cell){
	    cell_matrix = 0;

	    fe_values.reinit(cell);

	    for(unsigned int q_index = 0; q_index < n_q_points; ++q_index)
	    {
	      const Tensor<1,dim> velVal = advectionField.value(fe_values.quadrature_point(q_index)); // advection field value *****
	        // CAN MAYBE USE VECTORIZED VERSION...

	      for(unsigned int i=0; i < dofs_per_cell; ++i){
	        for(unsigned int j=0; j < dofs_per_cell; ++j){
	          cell_matrix(i,j) += ( fe_values.shape_value(i,q_index) * 
	            velVal * fe_values.shape_grad(j, q_index) * 
	            fe_values.JxW(q_index) ); // not quite right just yet *****
	        }
	      } // for cell indicies

	    } // for number of quadrature points -- could be more if using larger element space

	// *** CREATE SEPARATE ROBIN MATRIX ***
	    // Neumann BC enforcement:
	    //  LOOP OVER CELL FACES:
	      // for(unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f){
	      //   if( (cell ->face(f)->boundary_id() == 5) || (cell ->face(f)->boundary_id() == 6) ){

	      //       // std::cout << " at boundary 5" << std::endl;
	      //         fe_face_values.reinit(cell,f);

	      //         for(unsigned int q_index = 0; q_index < n_face_q_points; ++q_index){
	      //           const Tensor<1,2> velVal = advField.value(fe_face_values.quadrature_point(q_index)); // advection field value *****
	      //           const double neumann_value  = 1.0*(velVal * fe_face_values.normal_vector(q_index));

	      //           // // if( neumann_value > 0)
	      //           //   std::cout << "Boundary " << cell->face(f)->boundary_id()
	      //           //     << " Neumann value: " << neumann_value << std::endl;
	      //         // PRINT VALUE AND LOCATION TO FILE!!!

	      //             for(unsigned int i=0; i < dofs_per_cell; ++i){
	      //               for(unsigned int j=0; j < dofs_per_cell; ++j){
	      //                 robin_cell_matrix(i,j) += ( 
	      //                   fe_face_values.shape_value(i,q_index) *
	      //                    neumann_value * fe_face_values.shape_value(j, q_index) *
	      //                   fe_face_values.JxW(q_index) );               // may need a minus sign *****
	      //               }
	      //             } // for cell indicies
	      //           } // for quadrature points

	      //   } // if at a circle boundary
	      // } // for cell faces (for boundary condition)

	    // LOCAL TO GLOBAL:
	    cell->get_dof_indices(local_dof_indices);
	      // constraints.distribute_local_to_global(cell_matrix, local_dof_indices, advection_matrix); 

	    for(unsigned int i=0; i < dofs_per_cell; ++i){
	      for(unsigned int j=0; j < dofs_per_cell; ++j){
	        advection_matrix.add(local_dof_indices[i], 
	                            local_dof_indices[j], 
	                            cell_matrix(i,j) );
	      }
	    }


	    // for(unsigned int i=0; i < dofs_per_cell; ++i){
	    //   for(unsigned int j=0; j < dofs_per_cell; ++j){
	    //     robin_matrix.add(local_dof_indices[i], 
	    //                         local_dof_indices[j], 
	    //                         robin_cell_matrix(i,j) );
	    //   }
	    // }


	  } // for each cell


    }


    template<int dim>
    void Simulator_BDF2<dim>::solve_time_step()
    {
		SolverControl solver_control(1000, 1e-8 * system_rhs1.l2_norm());
		SolverCG<> cg(solver_control);

		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(system_matrix1, 1.0);

		cg.solve(system_matrix1, solution1, system_rhs1,
		       preconditioner);

		constraints.distribute(solution1);

		// std::cout << "Chem1:     " << solver_control.last_step()
		//           << " CG iterations." << std::endl;

		// === SOLVE CHEMICAL 2 ===
		SolverControl solver_control2(1000, 1e-8 * system_rhs2.l2_norm());
		SolverCG<> cg2(solver_control2);

		PreconditionSSOR<> preconditioner2;
		preconditioner2.initialize(system_matrix2, 1.0);

		cg2.solve(system_matrix2, solution2, system_rhs2,
		       preconditioner2);

		constraints.distribute(solution2);
    }


    template<int dim>
    void Simulator_BDF2<dim>::adv_solve_time_step()
    {
		SolverControl solver_control(1000, 1e-8 * system_rhs1.l2_norm());
		SolverBicgstab<> bicgstab(solver_control);

		PreconditionJacobi<> preconditioner;
		preconditioner.initialize(system_matrix1, 1.0);

		bicgstab.solve(system_matrix1, solution1, system_rhs1,
		       preconditioner);

		constraints.distribute(solution1);

		// std::cout << "Chem1:     " << solver_control.last_step()
		//           << " BICGSTAB iterations." << std::endl;

		SolverControl solver_control2(1000, 1e-8 * system_rhs2.l2_norm());
		SolverBicgstab<> bicgstab2(solver_control2);

		PreconditionJacobi<> preconditioner2;
		preconditioner2.initialize(system_matrix2, 1.0);

		bicgstab2.solve(system_matrix2, solution2, system_rhs2,
		       preconditioner2);

		constraints.distribute(solution2);
	}


	template<int dim>
	void Simulator_BDF2<dim>::update_chemicals()
	{
		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
			solve_time_step();
		else
			adv_solve_time_step();
	}

    template<int dim>
    void Simulator_BDF2<dim>::output_results() const
    {
		for(unsigned int i = 1; i <= 2; i++)
			{
			DataOut<dim> data_out;

			data_out.attach_dof_handler(dof_handler);
			if(i == 1)
			{
			  data_out.add_data_vector(solution1, "C1");
			}
			else
			{
			  data_out.add_data_vector(solution2, "C2");
			}

			data_out.build_patches();

			const std::string filename = outputDirectory
										 + "/solutionC"
			                             + Utilities::int_to_string(i,2) 
			                             + "-"
			                             + Utilities::int_to_string(saveStepNumber, 4) +
			                             ".vtk";
			std::ofstream output(filename.c_str());
			data_out.write_vtk(output);
		} // for each chemical

    }

	
	template<int dim>    
    void Simulator_BDF2<dim>::output_bacteria() const
    {
		std::string outfile = outputDirectory
			+ "/bacteria-" + std::to_string(saveStepNumber) + ".dat";
		std::ofstream out(outfile);
		bacteria.print(out);
    }



    template<int dim>
    void Simulator_BDF2<dim>::createGrid(unsigned int global_refinement, 
    	unsigned int sphere_refinement)
    {
    	MyGridGenerator<dim> gridGen;
    	gridGen.generateGrid(geometry,triangulation,sphere_refinement); // should take care of coloring boundaries

    	// // refine:
    	if(dim == 2)
	    	refineMeshSpheres(sphere_refinement);

	    triangulation.refine_global(global_refinement); 
        std::cout << "...Mesh refined globally: " << global_refinement << " times" << std::endl;
    }

	
	template<int dim>
    void Simulator_BDF2<dim>::outputGrid()
    {
    	std::string grid_out_file = outputDirectory + "/grid.eps";

		std::ofstream out (grid_out_file);
		GridOut grid_out;
		grid_out.write_eps (triangulation, out);
		std::cout << "Grid written to " << grid_out_file << std::endl;
    }


    template<int dim>
    void Simulator_BDF2<dim>::refineMeshSpheres(unsigned int sphere_refinement)
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
    void Simulator_BDF2<dim>::updateMatricesWithPointSources()
    {
		mass_matrix.vmult(system_rhs1, old_solution1);  // Matrix vector multiplication (MU^{n-1}) -- overwrites system_rhs
		mass_matrix.vmult(system_rhs2, old_solution2);


		// LEFT HAND SIDE TERMS:
		system_matrix1.copy_from(mass_matrix); // LHS = M
		system_matrix1 *= 1 + decay_constant1*time_step; //  for decay term -> LHS = (1+k \lam) M
		system_matrix1.add(time_step * diffusion_constant1, stiffness_matrix);   //  for diffusion term -> LHS = (1+k \lam) M + kd*L
		
		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
			system_matrix1.add(time_step, advection_matrix); // with a minus sign?
		// system_matrix1.add(-1.0*time_step, robin_matrix);

		system_matrix2.copy_from(mass_matrix);
		system_matrix2 *= 1 + decay_constant2*time_step; // *** for decay term
		system_matrix2.add(time_step * diffusion_constant2, stiffness_matrix);

		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
			system_matrix2.add(time_step, advection_matrix); // with a minus sign?
		// system_matrix2.add(-1.0*time_step, robin_matrix);


		// use point sources
		updatePointSources(); // updates tmp and tmp2

		system_rhs1.add(time_step, tmp1); // RHS = k*n*s
		system_rhs2.add(time_step, tmp2);

		constraints.condense (system_matrix1, system_rhs1); 
		constraints.condense (system_matrix2, system_rhs2);
    }
    

    template<int dim>
	void Simulator_BDF2<dim>::updatePointSources()
	{
		const unsigned int num_bacteria = bacteria.getSize();
		const double preFactor = 0.00160;

		// // reset tmp:
		// tmp1.reinit(dof_handler.n_dofs()); 
		// tmp2.reinit(dof_handler.n_dofs()); // probably too slow...

		tmp1 *= 0;
		tmp2 *= 0; // zero out ??

		for(unsigned int i = 0; i < num_bacteria; i++)
		{
			// clear tmp, tmp2:
			double sec_one = bacteria.getGoodSecretionRate(i);
			double sec_two = bacteria.getWasteSecretionRate(i);

			VectorTools::create_point_source_vector(dof_handler,
			                                        bacteria.getLocation(i),
			                                        temporary_one);


			tmp1.add(preFactor*sec_one,temporary_one);

			// c2:
			VectorTools::create_point_source_vector(dof_handler,
			                                        bacteria.getLocation(i),
			                                        temporary_two);

			tmp2.add(preFactor*sec_two,temporary_two);
		}

	}




// RUN
//----------------------------------------------------------------------------------------------------
	template<int dim>
	void Simulator_BDF2<dim>::run(const ArgParser& systemParameters)
	{
		// setup system with parameters:
		setup_system(systemParameters); 
	
		std::cout << "\n...Running FEM (CG) Simulator_BDF2 in " 
			<< dim << " dimensions...\n" << std::endl;

		// output initial condition:
		output_bacteria();
		output_results();	

		do{
			// update chemicals:
			updateMatricesWithPointSources(); // @todo have one update function with update sources
				// as a parameter read in from file
			update_chemicals();
			old_solution1 = solution1;
			old_solution2 = solution2;

			// update bacteria:
		    if(timeStepNumber > reproduction_delay)
				bacteria.reproduce(time_step, dof_handler, solution1, solution2);
//			std::cout << "now with " << bacteria.getSize() << " bacteria" << std::endl;
			// bacteria.mutateBacteria(mutation_rate,mutation_diff);
			bacteria.randomWalk(time_step, &geometry, &advectionField);
		
			// update time:
			time += time_step;
			++timeStepNumber;

			// output:
			if(timeStepNumber % save_period == 0) // could put into a function output_results()
			{
				++saveStepNumber;
				std::cout << "time: " << time << std::endl;
				output_bacteria();
				output_results();	
			}			
		} while(time <= run_time /*&& bacteria.isAlive() */);

	} // run






















//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// DEBUGGING:
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------














	
	template<int dim>
    void Simulator_BDF2<dim>::create_mass_check_vector()
	{
		mass_check_vector.reinit(dof_handler.n_dofs());

		const QGauss<dim>  quadrature_formula((fe.degree+1));
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients | update_JxW_values);
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points    = quadrature_formula.size();

		Vector<double>       cell_rhs (dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

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
			for (unsigned int i=0; i < dofs_per_cell; ++i)
				mass_check_vector(local_dof_indices[i]) += cell_rhs(i);
		}
    }


    template<int dim>
	void Simulator_BDF2<dim>::initialize_base_objects_debug(const ArgParser& systemParameters)
	{
		time_step = systemParameters.getTimeStep();
		run_time = systemParameters.getRunTime();
		save_period = systemParameters.getSavePeriod();
		outputDirectory = systemParameters.getOutputDirectory();

		if(systemParameters.isCheckMass() || systemParameters.isReproduceBacteria())
		{
			std::cout << "...setting chemical diffusion and decay constants" << std::endl;
			diffusion_constant1 = systemParameters.getGoodDiffusionConstant();
			diffusion_constant2 = systemParameters.getWasteDiffusionConstant();
			decay_constant1 = systemParameters.getGoodDecayConstant();
			decay_constant2 = systemParameters.getWasteDecayConstant();
		}

		std::cout << "...Initializing geometry" << std::endl;
		geometry.initialize(systemParameters.getGeometryFile(), systemParameters.getMeshFile());
		std::string geo_out_file = outputDirectory + "/geometry.dat";
		std::ofstream geo_out(geo_out_file);
		geometry.printInfo(geo_out);
		geometry.printInfo(std::cout);

		std::cout << "...Initializing advection" << std::endl;
		if( (systemParameters.getVelocityType() != VelocityType::NUMERICAL_FLOW)
			&& (systemParameters.getVelocityType() != VelocityType::TILE_FLOW) )
		{
			advectionField.initialize(systemParameters.getVelocityType(),
									geometry.getBottomLeftPoint(),
									geometry.getTopRightPoint(),
									systemParameters.getMaximumVelocity()
									);
							//  double vrad = 0, double vrotation = 0);
		}
		else
		{
			advectionField.initialize(systemParameters.getVelocityType(),
									geometry.getBottomLeftPoint(),
									geometry.getTopRightPoint(),
									geometry.getScaleFactor(),
									systemParameters.getMaximumVelocity(),
									systemParameters.getVelocityFile_X(),
									systemParameters.getVelocityFile_Y() 
									);
		}
		advectionField.printInfo(std::cout);

		if(systemParameters.isPrintVelocity())
		{
			std::string adv_file = outputDirectory + "/advection_field.dat";
			std::ofstream adv_out(adv_file);
			advectionField.print(geometry.getQuerryPoints(), adv_out);
		}

		// bacteria:
		if(systemParameters.isPointSource() || systemParameters.isReproduceBacteria())
		{
			std::vector<Point<dim> > locations = getBacteriaLocations(systemParameters.getNumberBacteria(), 
				systemParameters.getNumberGroups());

			std::cout << "...Initializing bacteria" << std::endl;
			bacteria.initialize(systemParameters.getBacteriaDiffusionConstant(), 
				systemParameters.getNumberBacteria(),
				locations,
				systemParameters.getGoodSecretionRate(),
				 systemParameters.getWasteSecretionRate() );
		}

		createGrid(systemParameters.getGlobalRefinement(), systemParameters.getSphereRefinement());

		outputGrid();
		// initialize advection and bacteria...
	}


	template<int dim>
	void Simulator_BDF2<dim>::setup_system_debug(const ArgParser& systemParameters)
	{
  		initialize_base_objects_debug(systemParameters);
  		if(systemParameters.isCheckMass() || systemParameters.isReproduceBacteria())
  		{
	  		setup_constraints();
	  		initialize_vectors_matrices();
	  	}
	} // setup system() *** maybe split assemble system


	template<int dim>
	void Simulator_BDF2<dim>::printMassToFile(std::vector<double> massOne, 
		std::vector<double> massTwo) const
	{
	  const std::string filename = outputDirectory + "/massVsTime.dat";

	  std::ofstream output(filename.c_str());

	  unsigned int n = massOne.size();
	  for(unsigned int i = 0; i < n; i++)
	    output << massOne[i] << "\t" << massTwo[i] << std::endl;
	} // printMassToFile()


	template<int dim>
	void Simulator_BDF2<dim>::projectInitialCondition(Point<dim> gaussian_center)
	{
		  GaussianFE<dim> initialFunction(gaussian_center);

            // ConstraintMatrix constraints;
            // constraints.close();
            VectorTools::project (dof_handler,
                                  constraints,
                                  QGauss<dim>(fe.degree+2),
                                  initialFunction,
                                  old_solution1); 

            VectorTools::project (dof_handler,
                              constraints,
                              QGauss<dim>(fe.degree+2),
                              initialFunction,
                              old_solution2); 

            solution1 = old_solution1;
            solution2 = old_solution2;
	}


	template<int dim>
	void Simulator_BDF2<dim>::updateMatrices(bool usePointSources)
	{
		mass_matrix.vmult(system_rhs1, old_solution1);  // Matrix vector multiplication (MU^{n-1}) -- overwrites system_rhs
		mass_matrix.vmult(system_rhs2, old_solution2);


		// LEFT HAND SIDE TERMS:
		system_matrix1.copy_from(mass_matrix); // LHS = M
		system_matrix1 *= 1 + decay_constant1*time_step; //  for decay term -> LHS = (1+k \lam) M
		system_matrix1.add(time_step * diffusion_constant1, stiffness_matrix);   //  for diffusion term -> LHS = (1+k \lam) M + kd*L

		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
		{
			system_matrix1.add(time_step, advection_matrix); // with a minus sign?
			std::cout << "updating chemical 1 with advection " << std::endl;
		}
		// system_matrix1.add(-1.0*time_step, robin_matrix);

		system_matrix2.copy_from(mass_matrix);
		system_matrix2 *= 1 + decay_constant2*time_step; // *** for decay term
		system_matrix2.add(time_step * diffusion_constant2, stiffness_matrix);

		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
			system_matrix2.add(time_step, advection_matrix); // with a minus sign?
		// system_matrix2.add(-1.0*time_step, robin_matrix);


		// use point sources
		if(usePointSources)
		{
			updatePointSources(); // updates tmp and tmp2 

			system_rhs1.add(time_step, tmp1); // RHS = k*n*s
			system_rhs2.add(time_step, tmp2);
		}

		constraints.condense (system_matrix1, system_rhs1); 
		constraints.condense (system_matrix2, system_rhs2);
	}


	template<int dim>
	Point<dim> Simulator_BDF2<dim>::findGaussianCenterPoint() const
	{
		Point<dim> result;
		// center y and z:
		for(unsigned int dim_itr = 1; dim_itr < dim; dim_itr++)
			result[dim_itr] = geometry.getBottomLeftPoint()[dim_itr] 
				+ 0.5*geometry.getWidth(dim_itr);

		// start guess point at 10% of way next to left wall, 
		// increment by 1% until point is in domain ...
		const unsigned int x_dim = 0;

		result[x_dim] = geometry.getBottomLeftPoint()[x_dim]
			+ 0.1*geometry.getWidth(x_dim);

		const double increment = 0.01*geometry.getWidth(x_dim);
		
		do{
			if(geometry.isInDomain(result))
				return result;

			result[x_dim] += increment;

		} while( result[x_dim] < geometry.getTopRightPoint()[x_dim] );

		throw std::runtime_error("Error: initial gaussian center not found");

	}




	template<int dim>
	void Simulator_BDF2<dim>::run_debug(const ArgParser& systemParameters)
	{
		std::cout << "\n...Running DEBUG FEM (CG) in " 
			<< dim << " dimensions...\n" << std::endl;

		setup_system_debug(systemParameters); 

		if(systemParameters.isCheckMass())
			create_mass_check_vector();

		const unsigned int numtime = 
			ceil( systemParameters.getRunTime() / systemParameters.getTimeStep() );
		std::vector<double> total_mass_one, total_mass_two;
		total_mass_one.reserve(numtime);
		total_mass_two.reserve(numtime); 
		// get from system parameters:

		// can create a debug parameter file and debug enums ...
		// const bool testing_chemicals = true;
		// const bool testing_bacteria = false;
		// const bool bacteria_source = false;
		// const bool initial_gaussian = true;

		if(systemParameters.isInitialGaussian())
			projectInitialCondition(findGaussianCenterPoint());

		output_results();

		do{
			if(systemParameters.isCheckMass() || systemParameters.isReproduceBacteria())
			{
				// update chemicals:
				updateMatrices(systemParameters.isPointSource()); 
				update_chemicals(); // @todo use velocity type to switch between solvers
				old_solution1 = solution1;
				old_solution2 = solution2;	

			    // === UPDATE CHECK VECTORS: ===
			    if(systemParameters.isCheckMass())
			    {
					total_mass_one.push_back( solution1*mass_check_vector );
					total_mass_two.push_back( solution2*mass_check_vector );
				}
			}
			if(systemParameters.isReproduceBacteria())
			{
			    if(time > systemParameters.getReproductionDelay())
					bacteria.reproduce(time_step, dof_handler, solution1, solution2);
				// if(time > systemParameters.getMutationDelay())
				// 	bactera.mutate()
			}

			if(time > systemParameters.getFlowDelay())
				bacteria.randomWalk(time_step, &geometry, &advectionField);
			else		
				bacteria.randomWalk(time_step, &geometry);
			
			// update time:
			time += time_step;
			++timeStepNumber;

			// output:
			if(timeStepNumber % save_period == 0) // could put into a function output_results()
			{
				++saveStepNumber;
				std::cout << "time: " << time << std::endl;
				output_bacteria();
				output_results(); // output chemicals
			}		

		} while(time <= run_time);


	  // OUTPUT CHECK
		if(systemParameters.isCheckMass())
			printMassToFile(total_mass_one, total_mass_two);

	} // run debug


}

#endif // Simulator_BDF2.h




