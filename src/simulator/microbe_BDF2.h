#ifndef MICROBE_SIMULATOR_MICROBE_BDF2_H
#define MICROBE_SIMULATOR_MICROBE_BDF2_H


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


// solution class for checking error:
template<int dim>
class GaussSolution : public Function<dim>
{
public:
	GaussSolution() :Function<dim>()
		,time(0)
		,diffusion_constant(0)
		,decay_constant(0)
		,velocity(0) 
		{}

	virtual double value(const Point<dim>& p,
							const unsigned int component = 0) const;
	virtual Tensor<1,dim> gradient(const Point<dim>& p, 
							const unsigned int component = 0) const;

	void set_solution_constants(double diff, double decay, double vel);
	void update_solution_time(double t);

private:
	double time;
	double diffusion_constant;
	double decay_constant;
	double velocity;
};

template<int dim>
double
GaussSolution<dim>::value(const Point<dim>& p, const unsigned int /* component */) const
{
	if(dim != 2)
		throw std::runtime_error("solution not implemented for dim != 2");

	const double exponent = (-1.0/ (1.0 + 4.0*diffusion_constant*time) )
		*( decay_constant*time
			*(1.0 + 4.0*diffusion_constant*time)
			+ time*time*velocity*velocity 
			- 2.*time*velocity*p[0] 
			+ p[0]*p[0]
			+ p[1]*p[1]
			);
	const double scale_factor = 1.0/ (1.0 + 4.0*diffusion_constant*time);


	return scale_factor*std::exp( exponent );
}


template<int dim>
Tensor<1,dim>
GaussSolution<dim>::gradient(const Point<dim>& p, const unsigned int /*component*/) const
{
	if(dim != 2)
		throw std::runtime_error("solution not implemented for dim != 2");
	
	Tensor<1,dim> return_value = p;
	return_value[0] += -velocity*time;

	const double exponent = (-1.0/ (1.0 + 4.0*diffusion_constant*time) )
		*( decay_constant*time
			*(1.0 + 4.0*diffusion_constant*time)
			+ time*time*velocity*velocity 
			- 2.*time*velocity*p[0] 
			+ p[0]*p[0]
			+ p[1]*p[1]
			);
	const double scale_factor = 1.0/ (1.0 + 4.0*diffusion_constant*time);

	return_value *= 2*scale_factor*scale_factor*std::exp( exponent );

	return return_value;
}


template<int dim>
void 
GaussSolution<dim>::set_solution_constants(double diff, double decay, double vel)
{
	diffusion_constant = diff;
	decay_constant = decay;
	velocity = vel;
}


template<int dim>
void 
GaussSolution<dim>::update_solution_time(double t)
{
	time = t;
}



//-----------------------------------------------------------------------------------------------
	template<int dim>
	class InitialChemical : public Function<dim>{
	public:
		InitialChemical() : Function<dim>(1), center() {}
		InitialChemical(const Point<dim> & c) : Function<dim>(1), center(c) {}

		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

		virtual void vector_value(const Point<dim> &p, Vector<double>   &values) const;

	private:
		Point<dim> center;
	};

	template<int dim>
	double InitialChemical<dim>::value(const Point<dim> &p,
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
  InitialChemical<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = InitialChemical<dim>::value (p, c);
  }







	// template<int dim>
	// class ZeroFunction : public Function<dim>{
	// public:
	// 	ZeroFunction() : Function<dim>(1), center() {}
	// 	ZeroFunction(const Point<dim> & c) : Function<dim>(1), center(c) {}

	// 	virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

	// 	virtual void vector_value(const Point<dim> &p, Vector<double>   &values) const;

	// private:
	// 	Point<dim> center;
	// };

	// template<int dim>
	// double ZeroFunction<dim>::value(const Point<dim> &p,
	// 									const unsigned int component) const
	// {
	// 	(void) component;
	// 	Assert(component == 0,
	// 		ExcMessage("Invalid operation for a scalar function."));
	//     Assert ((dim==2) || (dim==3), ExcNotImplemented());
	
	// 	return 0; //exp( -(p-center)*(p-center) ); 
	// } 

 //  template <int dim>
 //  void
 //  ZeroFunction<dim>::vector_value (const Point<dim> &p,
 //                                               Vector<double>   &values) const
 //  {
 //    for (unsigned int c=0; c<this->n_components; ++c)
 //      values(c) = ZeroFunction<dim>::value (p, c);
 //  }











	template<int dim>
	class MicrobeBDF2{
	public:
		// enum -- chemical type ? ... or polymorphism
		MicrobeBDF2();
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

	    SparseMatrix<double> goods_system_matrix;
	    Vector<double>       public_goods;
	    Vector<double>       old_public_goods;
	    Vector<double> 		 old_old_public_goods;

	    Vector<double>       goods_rhs;
	    Vector<double>		 goods_source;


	    SparseMatrix<double> waste_system_matrix;
	    Vector<double>       waste_chemical;
	    Vector<double>       old_waste_chemical;
	    Vector<double>       old_old_waste_chemical;

	    Vector<double>       waste_rhs;
	    Vector<double>		 waste_source;

	    Vector<double> 		 temporary;

        double good_diffusion_constant;
	    double waste_diffusion_constant;
	    double good_decay_constant;
	    double waste_decay_constant;

	    double time; // t
	    double time_step; // dt
	    double old_time_step; // usually should equal dt ...
	    unsigned int timeStepNumber; // n
		const unsigned int reproduction_delay;

		unsigned int save_period;
		unsigned int saveStepNumber;
		double run_time;

		std::string outputDirectory;

		ConvergenceTable                        convergence_table;

		//-----------------------------------------------------------------------------
		// stabilized BDF-2:
		// void setup_base_objects(const ArgParser& systemParameters); // including time step
		void setupSystem(const ArgParser& systemParameters);
			void setup_system_constants(const ArgParser& systemParameters);
				double get_CFL_time_step();
			void setupGeometry(const ArgParser& systemParameters);
			void setupAdvection(const ArgParser& systemParameters);
			void setupBacteria(const ArgParser& systemParameters);					
		  		std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, 
		  			unsigned int number_groups);
		  		std::vector<Point<dim> > getMixerLocations(unsigned int number_groups);
			void setupGrid(const ArgParser& systemParameters, unsigned int cycle = 0);
				void createGrid();
				void refineGrid(unsigned int global, unsigned int sphere);
				void refineMeshSpheres(unsigned int sphere_refinement);
				void outputGrid();
			void setup_dofs();
	  			void initialize_vectors_matrices();
			void assemble_system_matrices();

    	double get_maximal_velocity () const;

		void setupSystemTest(const ArgParser& systemParameters,
								unsigned int cycle);


		void assemble_chemical_system(const double maximal_velocity);

		void updateSources();

		std::pair<double,double> get_extrapolated_goods_range () const;
		std::pair<double,double> get_extrapolated_waste_range () const;

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
								const double						cell_diameter,
								const double						diffusion_constant,
								const double						decay_constant
								) const;
		void solve_bdf2();

        void output_chemicals() const;
        void output_chemicals(unsigned int cycle) const;
        void output_bacteria() const; 
		//-----------------------------------------------------------------------------
    	//DEBUGGING:
    	Vector<double> projected_exact_goods;
    	Vector<double> projected_exact_waste;

    	void output_single_vector(const Vector<double>& field, std::string filename) const;

    	// void outputFieldsDifferences(GaussSolution<dim> goods, GaussSolution<dim> waste);

        void process_goods_solution(unsigned int cycle, 
        	GaussSolution<dim> exact_solution);
        void output_error_analysis();

    //     Vector<double>       mass_check_vector; 

  		// void setup_system_debug(const ArgParser& systemParameters);
  		// void initialize_base_objects_debug(const ArgParser& systemParameters);

	   //  void create_mass_check_vector();
	   //  void printMassToFile(std::vector<double> massOne, 
    // 		std::vector<double> massTwo) const;
    // 	void projectInitialCondition(Point<dim> gaussian_center);
    // 	Point<dim> findGaussianCenterPoint() const;
    // 	void updateMatrices(bool usePointSources);

	};


// BDF-2 STUFF:
//--------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------
template<int dim>
MicrobeBDF2<dim>::MicrobeBDF2()
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


template<int dim>
void
MicrobeBDF2<dim>::setupSystem(const ArgParser& systemParameters)
{
	setup_system_constants(systemParameters);
	setupGeometry(systemParameters);
	setupAdvection(systemParameters);
	setupBacteria(systemParameters);
	setupGrid(systemParameters);
	setup_dofs();
	initialize_vectors_matrices();
	assemble_system_matrices();

	time_step = get_CFL_time_step(); 
	old_time_step = time_step;
}


template<int dim>
void
MicrobeBDF2<dim>::setupSystemTest(const ArgParser& systemParameters,
									unsigned int cycle)
{
	setup_system_constants(systemParameters);
	setupGeometry(systemParameters);
	setupAdvection(systemParameters);
	// setupBacteria(systemParameters);
	setupGrid(systemParameters, cycle);
	setup_dofs();
	initialize_vectors_matrices();
	assemble_system_matrices();

	time_step = get_CFL_time_step(); 
	old_time_step = time_step;
}


template<int dim>
void 
MicrobeBDF2<dim>::setup_system_constants(const ArgParser& systemParameters)
{
	std::cout << "...setting up system constants" << std::endl;
	run_time = systemParameters.getRunTime();
	save_period = systemParameters.getSavePeriod(); // may want to use a rate if time step is variable..
	good_diffusion_constant = systemParameters.getGoodDiffusionConstant();
	waste_diffusion_constant = systemParameters.getWasteDiffusionConstant();
	good_decay_constant = systemParameters.getGoodDecayConstant();
	waste_decay_constant = systemParameters.getWasteDecayConstant();
	outputDirectory = systemParameters.getOutputDirectory();
}

	
template<int dim>
double 
MicrobeBDF2<dim>::get_CFL_time_step()
{
	const double min_time_step = 0.001;

	const double maximal_velocity = get_maximal_velocity();
	double cfl_time_step = 0;

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

	cfl_time_step = std::min(min_time_step,cfl_time_step);

	std::cout << "...using time step: " << cfl_time_step << std::endl;

	return cfl_time_step;
}


template<int dim>
void 
MicrobeBDF2<dim>::setupGeometry(const ArgParser& systemParameters)
{
	std::cout << "...Initializing geometry" << std::endl;
	geometry.initialize(systemParameters.getGeometryFile(), 
		systemParameters.getMeshFile());
	
	std::string geo_out_file = outputDirectory + "/geometryInfo.dat";
	std::ofstream geo_out(geo_out_file);
	geometry.printInfo(geo_out);
	geometry.printInfo(std::cout);
}


template<int dim>
void 
MicrobeBDF2<dim>::setupAdvection(const ArgParser& systemParameters)
{
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
}


template<int dim>
void
MicrobeBDF2<dim>::setupBacteria(const ArgParser& systemParameters)
{
	std::cout << "...Initializing bacteria" << std::endl;
	std::vector<Point<dim> > locations ={Point<2>(2,1), Point<2>(2,5)}; 

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
    bacteria.printInfo(std::cout);
}


template<int dim>
std::vector<Point<dim> > 
MicrobeBDF2<dim>::getBacteriaLocations(unsigned int number_bacteria, 
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
MicrobeBDF2<dim>::getMixerLocations(unsigned int number_groups)
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
MicrobeBDF2<dim>::setupGrid(const ArgParser& systemParameters, unsigned int cycle)
{
	std::cout << "...Creating grid" << std::endl;
	createGrid();
		std::cout << "grid is set " << std::endl;

	global_Omega_diameter = GridTools::diameter(triangulation);

		std::cout << "global_Omega_diameter is " << global_Omega_diameter << std::endl;

	refineGrid(systemParameters.getGlobalRefinement() + cycle, systemParameters.getSphereRefinement());
	std::cout << "grid is refined " << std::endl;
	outputGrid();
	std::cout << "grid output" << std::endl;
}


template<int dim>
void
MicrobeBDF2<dim>::createGrid()
{
	MyGridGenerator<dim> gridGen;
	gridGen.generateGrid(geometry,triangulation); // should take care of coloring boundaries
}


template<int dim>
void 
MicrobeBDF2<dim>::refineGrid(unsigned int global_refinement, 
	unsigned int sphere_refinement)
{
	if(dim == 2)
    	refineMeshSpheres(sphere_refinement);

    triangulation.refine_global(global_refinement); 
    std::cout << "...Mesh refined globally: " << global_refinement << " times" << std::endl;
}


template<int dim>
void MicrobeBDF2<dim>::refineMeshSpheres(unsigned int sphere_refinement)
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
MicrobeBDF2<dim>::outputGrid()
{
	std::string grid_out_file = outputDirectory + "/grid.eps";

	std::ofstream out (grid_out_file);
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "Grid written to " << grid_out_file << std::endl;
}


template<int dim>
void 
MicrobeBDF2<dim>::setup_dofs()
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
	initialize_vectors_matrices();
}


template<int dim>
void 
MicrobeBDF2<dim>::initialize_vectors_matrices()
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

	waste_chemical.reinit(dof_handler.n_dofs());
	old_waste_chemical.reinit(dof_handler.n_dofs());
	old_old_waste_chemical.reinit(dof_handler.n_dofs());
	waste_rhs.reinit(dof_handler.n_dofs());
	waste_source.reinit(dof_handler.n_dofs());

	temporary.reinit(dof_handler.n_dofs());
}


template<int dim>
void
MicrobeBDF2<dim>::assemble_system_matrices()
{
	std::cout << "...assembling system matrices" << std::endl;
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
						*chemical_fe_values.JxW(q));

					local_stiffness_matrix(i,j) += (grad_phi_T[i] * grad_phi_T[j]
						*chemical_fe_values.JxW(q));
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
MicrobeBDF2<dim>::get_maximal_velocity() const
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
void MicrobeBDF2<dim>::assemble_chemical_system(const double maximal_velocity)
{
	const bool use_bdf2_scheme = (timeStepNumber != 0);

	if(use_bdf2_scheme == true)
	{
		goods_system_matrix.copy_from(mass_matrix);
		goods_system_matrix *= (time_step*good_decay_constant +
							(2*time_step + old_time_step) /
							(time_step + old_time_step) ); // ***need to add decay term...
		goods_system_matrix.add(time_step*good_diffusion_constant, stiffness_matrix);

		waste_system_matrix.copy_from(mass_matrix);
		waste_system_matrix *= (time_step*waste_decay_constant +
							(2*time_step + old_time_step) /
							(time_step + old_time_step) ); // ***need to add decay term...
		waste_system_matrix.add(time_step*waste_diffusion_constant, stiffness_matrix);
	}
	else
	{
		goods_system_matrix.copy_from(mass_matrix);
		goods_system_matrix *= (1.0 + time_step*good_decay_constant);
		goods_system_matrix.add(time_step*good_diffusion_constant,stiffness_matrix);

		waste_system_matrix.copy_from(mass_matrix);
		waste_system_matrix *= (1.0 + time_step*waste_decay_constant);
		waste_system_matrix.add(time_step*waste_diffusion_constant,stiffness_matrix);
	}

	goods_rhs = 0;
	waste_rhs = 0;

	const QGauss<dim> quadrature_formula(chemical_fe_degree + 2);
	FEValues<dim> chemical_fe_values(fe,quadrature_formula,                                           
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

	// solution values:
	std::vector<Tensor<1, dim> > 	old_velocity_values(n_q_points); // ***this is a constant!! -- need only find once

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
	std::vector<double> 	good_source_values(n_q_points);
	std::vector<double> 	waste_source_values(n_q_points);
	// updateSources(); // need to pass a bool or something ***

	// shape functions:
	std::vector<double>				phi_T(dofs_per_cell);
	std::vector<Tensor<1,dim> >		grad_phi_T(dofs_per_cell);

	// total ranges:
	const std::pair<double,double> global_goods_range = get_extrapolated_goods_range();
	const std::pair<double,double> global_waste_range = get_extrapolated_waste_range();

	// const FEValuesExtractors::Vector velocities(0); // used for stokes ... ** can use later!!

	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
													endc = dof_handler.end();

	for(; cell!=endc; ++cell)
	{
		goods_local_rhs = 0;
		waste_local_rhs = 0;

		chemical_fe_values.reinit(cell);
		
		// update goods:
		chemical_fe_values.get_function_values(old_public_goods, old_goods_values);
		chemical_fe_values.get_function_values(old_old_public_goods, old_old_goods_values);

		chemical_fe_values.get_function_gradients(old_public_goods, old_goods_grads);
		chemical_fe_values.get_function_gradients(old_old_public_goods, old_old_goods_grads);

		chemical_fe_values.get_function_laplacians(old_public_goods, old_goods_laplacians);
		chemical_fe_values.get_function_laplacians(old_old_public_goods, old_old_goods_laplacians);
			
		// update waste:
		chemical_fe_values.get_function_values(old_waste_chemical, old_waste_values);
		chemical_fe_values.get_function_values(old_old_waste_chemical, old_old_waste_values);

		chemical_fe_values.get_function_gradients(old_waste_chemical, old_waste_grads);
		chemical_fe_values.get_function_gradients(old_old_waste_chemical, old_old_waste_grads);

		chemical_fe_values.get_function_laplacians(old_waste_chemical, old_waste_laplacians);
		chemical_fe_values.get_function_laplacians(old_old_waste_chemical, old_old_waste_laplacians);

		// update sources:	
		// chemical_fe_values.get_function_values(goods_source, good_source_values);
		// chemical_fe_values.get_function_values(waste_source, waste_source_values);

		// get velocities:
		advectionField.value_list(chemical_fe_values.get_quadrature_points(), 
				old_velocity_values); 

		// compute viscosity:
		const double nu_goods 
			= 0;
			// compute_viscosity(old_goods_values,
			// 					old_old_goods_values,
			// 					old_goods_grads,
			// 					old_old_goods_grads,
			// 					old_goods_laplacians,
			// 					old_old_goods_laplacians,
			// 					old_velocity_values,
			// 					good_source_values,
			// 					maximal_velocity,
			// 					global_goods_range.second - global_goods_range.first, 
			// 					cell->diameter(),
			// 					good_diffusion_constant,
			// 					good_decay_constant);
		
		const double nu_waste 
			= 0;
			// compute_viscosity(old_waste_values,
			// 					old_old_waste_values,
			// 					old_waste_grads,
			// 					old_old_waste_grads,
			// 					old_waste_laplacians,
			// 					old_old_waste_laplacians,
			// 					old_velocity_values,
			// 					waste_source_values,
			// 					maximal_velocity,
			// 					global_waste_range.second - global_waste_range.first, 
			// 					cell->diameter(),
			// 					waste_diffusion_constant,
			// 					waste_decay_constant);

		// local to global:

		for(unsigned int q = 0; q < n_q_points; ++q)
		{
			for(unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_T[k] = chemical_fe_values.shape_grad(k,q);
				phi_T[k] = chemical_fe_values.shape_value(k,q);
			}

			// goods:
			const double good_T_term_for_rhs
		      = (use_bdf2_scheme ?
	               (old_goods_values[q] *
	                (1 + time_step/old_time_step)
	                -
	                old_old_goods_values[q] *
	                (time_step * time_step) /
	                (old_time_step * (time_step + old_time_step)))
	               :
	               old_goods_values[q]);

		    const Tensor<1,dim> good_ext_grad_T
	            = (use_bdf2_scheme ?
	               (old_goods_grads[q] *
	                (1 + time_step/old_time_step)
	                -
	                old_old_goods_grads[q] *
	                time_step/old_time_step)
	               :
	               old_goods_grads[q]);

            //waste:
			const double waste_T_term_for_rhs
		      = (use_bdf2_scheme ?
	               (old_waste_values[q] *
	                (1 + time_step/old_time_step)
	                -
	                old_old_waste_values[q] *
	                (time_step * time_step) /
	                (old_time_step * (time_step + old_time_step)))
	               :
	               old_waste_values[q]);

		    const Tensor<1,dim> waste_ext_grad_T
	            = (use_bdf2_scheme ?
	               (old_waste_grads[q] *
	                (1 + time_step/old_time_step)
	                -
	                old_old_waste_grads[q] *
	                time_step/old_time_step)
	               :
	               old_waste_grads[q]);

	        const Tensor<1,dim> extrapolated_u = old_velocity_values[q];


	        for(unsigned int i = 0; i < dofs_per_cell; ++i)
	        {
	        	goods_local_rhs(i) +=  (good_T_term_for_rhs * phi_T[i]
	                             -
	                             time_step *
	                             extrapolated_u * good_ext_grad_T * phi_T[i]
	                             -
	                             time_step *
	                             nu_goods * good_ext_grad_T * grad_phi_T[i]
	                             +
	                             time_step *
	                             good_source_values[q] * phi_T[i])
	                            *
	                            chemical_fe_values.JxW(q); 

	        	waste_local_rhs(i) +=  (waste_T_term_for_rhs * phi_T[i]
	                             -
	                             time_step *
	                             extrapolated_u * waste_ext_grad_T * phi_T[i]
	                             -
	                             time_step *
	                             nu_waste * waste_ext_grad_T * grad_phi_T[i]
	                             +
	                             time_step *
	                             waste_source_values[q] * phi_T[i])
	                            *
	                            chemical_fe_values.JxW(q);
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

    // update sources:
    //... ***


} // assemble chemical system


template<int dim>
void
MicrobeBDF2<dim>::updateSources()
{
	goods_source = 0;
	waste_source = 0;

	const unsigned int number_bacteria = bacteria.getSize();
	const double scale_factor = 0.00160;

	for(unsigned int i = 0; i < number_bacteria; i++)
	{
		const double good_secretion = bacteria.getGoodSecretionRate(i);
		const double waste_secretion = bacteria.getWasteSecretionRate(i);

		VectorTools::create_point_source_vector(dof_handler,
												bacteria.getLocation(i),
												temporary);

		goods_source.add(scale_factor*good_secretion, temporary);
		waste_source.add(scale_factor*waste_secretion, temporary);
	}
}


template<int dim>
std::pair<double,double> 
MicrobeBDF2<dim>::get_extrapolated_goods_range () const
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
			fe_values.get_function_values(old_public_goods,
											old_chemical_values);
			fe_values.get_function_values(old_old_public_goods,
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
	         fe_values.get_function_values (old_public_goods,
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
std::pair<double,double> 
MicrobeBDF2<dim>::get_extrapolated_waste_range () const
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
			fe_values.get_function_values(old_waste_chemical,
											old_chemical_values);
			fe_values.get_function_values(old_old_waste_chemical,
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
	         fe_values.get_function_values (old_waste_chemical,
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

}


template<int dim>
double 
MicrobeBDF2<dim>::compute_viscosity(
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
						const double						cell_diameter,
						const double 						diffusion_constant,
						const double						decay_constant
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

		const double kappa_Delta_T = diffusion_constant
						* (old_chemical_laplacians[q] + old_old_chemical_laplacians[q]) /2;

		const double lambda_T = 0.5*decay_constant*( old_chemical[q] + old_old_chemical[q] ); // decay term...

		const double residual 
			= std::abs( ( dT_dt + u_grad_T - kappa_Delta_T - gamma_values[q] + lambda_T )*
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
MicrobeBDF2<dim>::solve_bdf2()
{
 // get time step from velocity and CFL condition, do this in setup since velocity is constant

	const double maximal_velocity = get_maximal_velocity();
	// double cfl_time_step = 0;

	// std::cout << "Maximal velocity: " << maximal_velocity << std::endl;

	// if(maximal_velocity >= 0.01)
	// 	cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
	// 		chemical_fe_degree *
	// 		GridTools::minimal_cell_diameter(triangulation) /
	// 		maximal_velocity;
	// else
	// 	cfl_time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
	// 		chemical_fe_degree *
	// 		GridTools::minimal_cell_diameter(triangulation) /
	// 		0.01;

	// std::cout << "\n\tCFL TIME STEP: " << cfl_time_step
	// 	<< std::endl;

	// public_goods = old_public_goods;

	assemble_chemical_system(maximal_velocity);

	// solve using CG ...
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
		cg.solve(waste_system_matrix, waste_chemical, waste_rhs,
		       preconditioner);
		constraints.distribute(waste_chemical);
	}

	
  // std::cout << "     " << solver_control.last_step()
  //           << " CG iterations." << std::endl;

	// get value range:
 //    double min_chemical = public_goods(0),
 //       max_chemical = public_goods(0);
    
 //    for (unsigned int i=0; i< public_goods.size(); ++i)
	// {
	// 	min_chemical = std::min<double> (min_chemical,
	//                                     public_goods(i));
	// 	max_chemical = std::max<double> (max_chemical,
	//                                     public_goods(i));
	// }
	// std::cout << "   Chemical range: "
	//       << min_chemical << ' ' << max_chemical
	//       << std::endl;

}


template<int dim>
void 
MicrobeBDF2<dim>::output_chemicals() const
{
	// public goods:
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(public_goods,"public_goods");
		data_out.build_patches();
		const std::string filename = outputDirectory
									+ "/public_goods-"
									+ Utilities::int_to_string(saveStepNumber,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}

	// waste chemical:
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(waste_chemical,"waste");
		data_out.build_patches();
		const std::string filename = outputDirectory
									+ "/waste-"
									+ Utilities::int_to_string(saveStepNumber,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}	
}
        
template<int dim>
void 
MicrobeBDF2<dim>::output_chemicals(unsigned int cycle) const
{
	// public goods:
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(public_goods,"public_goods");
		data_out.build_patches();
		const std::string filename = outputDirectory
									+ "/cycle_" 
									+ Utilities::int_to_string(cycle,3) 
									+ "_public_goods-"
									+ Utilities::int_to_string(saveStepNumber,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}

	// waste chemical:
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(waste_chemical,"waste");
		data_out.build_patches();
		const std::string filename = outputDirectory
									+ "/cycle_" 
									+ Utilities::int_to_string(cycle,3) 
									+ "_waste-"
									+ Utilities::int_to_string(saveStepNumber,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
	}	
}


template<int dim>
void 
MicrobeBDF2<dim>::output_bacteria() const
{
	std::string filename = outputDirectory
						+ "/bacteria-" 
						+ Utilities::int_to_string(saveStepNumber,4)
						+ ".dat";
	std::ofstream out(filename);
	bacteria.print(out);
}


template<int dim>
void
MicrobeBDF2<dim>::run(const ArgParser& systemParameters)
{
    setupSystem(systemParameters);
    setup_dofs();

	assemble_system_matrices();


 	do{
		// update chemicals:
		solve_bdf2(); // includes -> assemble_chemical_system(get_maximal_velocity());    // update RHS
		old_old_public_goods = old_public_goods;
		old_public_goods = public_goods;
		old_old_waste_chemical = old_waste_chemical;
		old_waste_chemical = waste_chemical;

		// update bacteria:
		if(timeStepNumber > reproduction_delay)
			bacteria.reproduce(time_step, dof_handler, public_goods, waste_chemical);

		std::cout << "number alive: " << bacteria.getSize() << std::endl;
		bacteria.randomWalk(time_step, &geometry, &advectionField);
		//bacteria.mutate(mutation_rate);...

		// update time:
		time += time_step;
		++timeStepNumber;

		// save:
		if(timeStepNumber % save_period == 0)
		{
			++ saveStepNumber;
			std::cout << "time: " << time << std::endl;
			output_chemicals();
			output_bacteria();
		}
	} while(time <= run_time); 

	// printMassToFile(total_mass_one, total_mass_two);
} // run test



// DEBUGGING
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------

template<int dim>
void
MicrobeBDF2<dim>::process_goods_solution(unsigned int cycle, GaussSolution<dim> exact_solution)
{
	Vector<float> difference_per_cell(triangulation.n_active_cells());

	// compute L2-error
	VectorTools::integrate_difference(dof_handler,
										public_goods,
										exact_solution,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::L2_norm);

	// compute H1-error
	VectorTools::integrate_difference(dof_handler,
										public_goods,
										exact_solution,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::H1_seminorm);
	const double H1_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::H1_seminorm);;

	const QTrapez<1>		q_trapez;
	const QIterated<dim>	q_iterated(q_trapez, 5);
	VectorTools::integrate_difference(dof_handler,
										public_goods,
										exact_solution,
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
void
MicrobeBDF2<dim>::output_error_analysis()
{
	convergence_table.add_column_to_supercolumn("cycle", "n cells");
	convergence_table.add_column_to_supercolumn("cells", "n cells");

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
	convergence_table.write_text(std::cout); // *** can output in latex format
}


// RUN TEST:
template<int dim>
void
MicrobeBDF2<dim>::run_bdf2_test(const ArgParser& systemParameters)
{
	const unsigned int n_cycles = 6; // can get from system parameters...
	const bool usingPointSource = false;

	setup_system_constants(systemParameters);
	setupGeometry(systemParameters);
	setupAdvection(systemParameters);

	for(unsigned int cycle = 0; cycle < n_cycles; ++cycle)
	{
		saveStepNumber = 0;
		time = 0;
		timeStepNumber = 0;

		triangulation.clear();
		setupGrid(systemParameters, cycle + 2);
		{
			std::string grid_out_file = outputDirectory 
											+ "/cycle_" 
											+ Utilities::int_to_string(cycle,3)
											+ "grid.eps";

			std::ofstream out (grid_out_file);
			GridOut grid_out;
			grid_out.write_eps (triangulation, out);
			std::cout << "Grid written to " << grid_out_file << std::endl;
		}
		setup_dofs();
		initialize_vectors_matrices();
		assemble_system_matrices();

		time_step = get_CFL_time_step(); 
		old_time_step = time_step;

	    GaussSolution<dim> exact_goods_solution;

	    exact_goods_solution.set_solution_constants(good_diffusion_constant,
	    								good_decay_constant,
	    								systemParameters.getMaximumVelocity());

	    exact_goods_solution.update_solution_time(0); 

	    // output exact solution:
	    {
		     VectorTools::project (dof_handler,
			                      constraints,
			                      QGauss<dim>(chemical_fe_degree+2),
			                      exact_goods_solution,
			                      temporary);
	
			DataOut<dim> data_out;
			data_out.attach_dof_handler(dof_handler);
			data_out.add_data_vector(temporary,"exact_goods_solution");
			data_out.build_patches();
			const std::string filename = outputDirectory
										+ "/cycle_" 
										+ Utilities::int_to_string(cycle,3)
										+ "_exact_goods-"
										+ Utilities::int_to_string(saveStepNumber,4)
										+ ".vtk";
			std::ofstream output(filename.c_str());
			data_out.write_vtk(output);
		}

	    // print velocity:
	    std::string adv_file = outputDirectory + "/advection_field.dat";
	    std::ofstream adv_out(adv_file);
	    advectionField.print(geometry.getQuerryPoints(), adv_out);


	    if(usingPointSource == false)
	    {
		     VectorTools::project (dof_handler,
			                      constraints,
			                      QGauss<dim>(chemical_fe_degree+2),
			                      InitialChemical<dim>(Point<2>(0,0)),
			                      old_public_goods);
			

			 VectorTools::project (dof_handler,
		                      constraints,
		                      QGauss<dim>(chemical_fe_degree+2),
		                      InitialChemical<dim>(Point<2>(0,0)),
		                      old_waste_chemical);

			public_goods = old_public_goods;
			waste_chemical = old_waste_chemical;
			output_chemicals(cycle);
		}

		assemble_system_matrices();


	 	do{
			// update chemicals:
			solve_bdf2(); // includes -> assemble_chemical_system(get_maximal_velocity());    // update RHS
			old_old_public_goods = old_public_goods;
			old_public_goods = public_goods;
			old_old_waste_chemical = old_waste_chemical;
			old_waste_chemical = waste_chemical;

			// update bacteria:
			// if(timeStepNumber > reproduction_delay)
			// 	bacteria.reproduce(time_step, dof_handler, public_goods, waste_chemical);

			// std::cout << "number alive: " << bacteria.getSize() << std::endl;

			if(usingPointSource == true)
				bacteria.randomWalk(time_step, &geometry, &advectionField);
			//bacteria.mutate(mutation_rate);...

			// update time:
			time += time_step;
			++timeStepNumber;

			// save:
			if(timeStepNumber % save_period == 0)
			{
				++ saveStepNumber;
				std::cout << "time: " << time << std::endl;
				output_chemicals(cycle);
				// output_bacteria();

				// error analysis:

			}
		} while(time <= run_time); 

		exact_goods_solution.update_solution_time(time);
			    // output exact solution:
	    {
			++ saveStepNumber;
		     VectorTools::project (dof_handler,
			                      constraints,
			                      QGauss<dim>(chemical_fe_degree+2),
			                      exact_goods_solution,
			                      temporary);
	
			DataOut<dim> data_out;
			data_out.attach_dof_handler(dof_handler);
			data_out.add_data_vector(temporary,"exact_goods_solution");
			data_out.build_patches();
			const std::string filename = outputDirectory
										+ "/cycle_" 
										+ Utilities::int_to_string(cycle,3)
										+ "_exact_goods-"
										+ Utilities::int_to_string(saveStepNumber,4)
										+ ".vtk";
			std::ofstream output(filename.c_str());
			data_out.write_vtk(output);
		}

		output_chemicals(cycle);
		process_goods_solution(cycle, exact_goods_solution);
	} // for refinement cycles

	output_error_analysis();
} // run test



std::vector<Point<2> > boxQuerryPoints()
{
	std::vector<Point<2> > result;

	unsigned int n_ypoints = 3;
	unsigned int n_xpoints = 9;

	result.reserve(n_xpoints*n_ypoints);

	double y = -1;

	for(unsigned int i = 0; i < n_ypoints; i++)
	{
		double x = -4;
		for(unsigned int j = 0; j < n_xpoints; j++)
		{
			result.push_back(Point<2>(x,y));
			x += 1;
		}
		y += 1;
	}

	return result;
}


void printValues(std::ostream& out, std::vector<Point<2> > qpoints, std::vector<double> values)
{
	unsigned int size = qpoints.size();
	if( values.size() != size )
		throw std::runtime_error("vector of points and corresponding values must have same size");

	for(unsigned int i = 0; i < size; i++)
		out << qpoints[i] << " " << values[i] << std::endl;
}



template<int dim>
void 
MicrobeBDF2<dim>::output_single_vector(const Vector<double>& field, std::string outfile) const
{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(field,outfile);
		data_out.build_patches();
		const std::string filename = outputDirectory
									+ "/"
									+ outfile
									+ "-"
									+ Utilities::int_to_string(saveStepNumber,4)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);	
}



template<int dim>
void
MicrobeBDF2<dim>::run_debug(const ArgParser& systemParameters)
{
	const unsigned int grid_refinement = 6;
	const std::vector<Point<2> > querry_points = boxQuerryPoints();
	const unsigned int n_querry_points = querry_points.size();
	std::vector<double> values(n_querry_points);

	setup_system_constants(systemParameters);
	setupGeometry(systemParameters);
	setupAdvection(systemParameters);

	setupGrid(systemParameters, grid_refinement);
	{
		std::string grid_out_file = outputDirectory 
										+ "/recheck_grid.eps";
		std::ofstream out (grid_out_file);
		GridOut grid_out;
		grid_out.write_eps (triangulation, out);
		std::cout << "Grid written to " << grid_out_file << std::endl;
	}
	setup_dofs();
	initialize_vectors_matrices();
	assemble_system_matrices();

	time_step = get_CFL_time_step(); 
	old_time_step = time_step;

	projected_exact_goods.reinit(dof_handler.n_dofs());
	projected_exact_waste.reinit(dof_handler.n_dofs());

    // print velocity:
    std::string adv_file = outputDirectory + "/advection_field.dat";
    std::ofstream adv_out(adv_file);
    advectionField.print(geometry.getQuerryPoints(), adv_out);


    // setup initial condition:
     VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(chemical_fe_degree+2),
	                      InitialChemical<dim>(Point<2>(0,0)),
	                      old_public_goods);
	

	 VectorTools::project (dof_handler,
                      constraints,
                      QGauss<dim>(chemical_fe_degree+2),
                      InitialChemical<dim>(Point<2>(0,0)),
                      old_waste_chemical);

	public_goods = old_public_goods;
	waste_chemical = old_waste_chemical;
	
	
	// exact goods solution:
    GaussSolution<dim> exact_goods_solution;

    exact_goods_solution.set_solution_constants(good_diffusion_constant,
    								good_decay_constant,
    								systemParameters.getMaximumVelocity());
    
    exact_goods_solution.update_solution_time(time); 
    VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(chemical_fe_degree+2),
	                      exact_goods_solution,
	                      projected_exact_goods);

    // exact waste solution:
    GaussSolution<dim> exact_waste_solution;

    exact_waste_solution.set_solution_constants(waste_diffusion_constant,
    										waste_decay_constant,
    										systemParameters.getMaximumVelocity());

    exact_waste_solution.update_solution_time(time);
    VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(chemical_fe_degree+2),
	                      exact_waste_solution,
	                      projected_exact_waste);


	output_chemicals();
	output_single_vector(projected_exact_goods, "exact_goods");
	output_single_vector(projected_exact_waste, "exact_waste");

	// differences:
	temporary = projected_exact_goods;
	temporary -= public_goods;
	output_single_vector(temporary, "goods_difference");

	temporary = projected_exact_waste;
	temporary -= waste_chemical;
	output_single_vector(temporary, "waste_difference");


	// querry points and output:
	{
		std::ofstream out("projected_exact_goods-initial.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					projected_exact_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("numerical_goods-intitial.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					public_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("exact_goods_values-initial.dat");
		exact_goods_solution.value_list(querry_points, values);

		printValues(out, querry_points, values);
	}

	// wASTE FIELDS:

	{
		std::ofstream out("projected_exact_waste-initial.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					projected_exact_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("numerical_waste-intitial.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					public_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("exact_waste_values-initial.dat");
		exact_goods_solution.value_list(querry_points, values);

		printValues(out, querry_points, values);
	}

	assemble_system_matrices();


	// compute mass vs time:
	std::vector<double> mass_vs_time;

	/// evolve numerical solution:
 	do{
		// update chemicals:
		solve_bdf2(); // includes -> assemble_chemical_system(get_maximal_velocity());    // update RHS
		old_old_public_goods = old_public_goods;
		old_public_goods = public_goods;
		old_old_waste_chemical = old_waste_chemical;
		old_waste_chemical = waste_chemical;

		// update time:
		time += time_step;
		++timeStepNumber;
		std::cout << "time: " << time << std::endl;

		// compute mass:
		Vector<float> difference_per_cell(triangulation.n_active_cells());
		VectorTools::integrate_difference(dof_handler,
										 public_goods,
										 ZeroFunction<dim>(),
										 difference_per_cell,
										 QGauss<dim>(3),
										 VectorTools::L2_norm);

		const double mass = VectorTools::compute_global_error(triangulation,
														difference_per_cell,
														VectorTools::L2_norm);

		mass_vs_time.push_back(mass);
	} while(time <= run_time); 

	{
		std::ofstream out("mass_vs_time.dat");
		for(unsigned int i = 0; i < mass_vs_time.size(); i++)
		{
			out << mass_vs_time[i] << std::endl;
		}
	}

	// update exact solution:
 	 exact_goods_solution.update_solution_time(time); 
    VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(chemical_fe_degree+2),
	                      exact_goods_solution,
	                      projected_exact_goods);

    // exact waste solution:
    exact_waste_solution.update_solution_time(time);
    VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(chemical_fe_degree+2),
	                      exact_waste_solution,
	                      projected_exact_waste);

    ++saveStepNumber;

	output_chemicals();
	output_single_vector(projected_exact_goods, "exact_goods");
	output_single_vector(projected_exact_waste, "exact_waste");

	// differences:
	temporary = projected_exact_goods;
	temporary -= public_goods;
	output_single_vector(temporary, "goods_difference_final");

	temporary = projected_exact_waste;
	temporary -= waste_chemical;
	output_single_vector(temporary, "waste_difference_final");

// querry points and output:
	{
		std::ofstream out("projected_exact_goods-final.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					projected_exact_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("numerical_goods-final.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					public_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("exact_goods_values-final.dat");
		exact_goods_solution.value_list(querry_points, values);

		printValues(out, querry_points, values);
	}

	// wASTE FIELDS:

	{
		std::ofstream out("projected_exact_waste-final.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					projected_exact_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("numerical_waste-final.dat");
		for(unsigned int i = 0; i < n_querry_points; i++)
		{
			values[i] = dealii::VectorTools::point_value(dof_handler, 
					public_goods, querry_points[i]);
		}

		printValues(out, querry_points, values);
	}

	{
		std::ofstream out("exact_waste_values-final.dat");
		exact_goods_solution.value_list(querry_points, values);

		printValues(out, querry_points, values);
	}
} // run test
	
















































//IMPLEMENTATION
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
/*
    template<int dim>
    void MicrobeBDF2<dim>::solve_time_step()
    {
		SolverControl solver_control(1000, 1e-8 * goods_rhs.l2_norm());
		SolverCG<> cg(solver_control);

		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(goods_system_matrix, 1.0);

		cg.solve(goods_system_matrix, public_goods, goods_rhs,
		       preconditioner);

		constraints.distribute(public_goods);

		// std::cout << "Chem1:     " << solver_control.last_step()
		//           << " CG iterations." << std::endl;

		// === SOLVE CHEMICAL 2 ===
		SolverControl solver_control2(1000, 1e-8 * waste_rhs.l2_norm());
		SolverCG<> cg2(solver_control2);

		PreconditionSSOR<> preconditioner2;
		preconditioner2.initialize(waste_system_matrix, 1.0);

		cg2.solve(waste_system_matrix, waste_chemical, waste_rhs,
		       preconditioner2);

		constraints.distribute(waste_chemical);
    }

    template<int dim>
    void MicrobeBDF2<dim>::output_results() const
    {
		for(unsigned int i = 1; i <= 2; i++)
			{
			DataOut<dim> data_out;

			data_out.attach_dof_handler(dof_handler);
			if(i == 1)
			{
			  data_out.add_data_vector(public_goods, "C1");
			}
			else
			{
			  data_out.add_data_vector(waste_chemical, "C2");
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
	void MicrobeBDF2<dim>::updatePointSources()
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
	void MicrobeBDF2<dim>::run(const ArgParser& systemParameters)
	{
		// setup system with parameters:
		setup_system(systemParameters); 
	
		std::cout << "\n...Running FEM (CG) MicrobeBDF2 in " 
			<< dim << " dimensions...\n" << std::endl;

		// output initial condition:
		output_bacteria();
		output_results();	

		do{
			// update chemicals:
			updateMatricesWithPointSources(); // @todo have one update function with update sources
				// as a parameter read in from file
			update_chemicals();
			old_public_goods = public_goods;
			old_waste_chemical = waste_chemical;

			// update bacteria:
		    if(timeStepNumber > reproduction_delay)
				bacteria.reproduce(time_step, dof_handler, public_goods, waste_chemical);
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
		} while(time <= run_time && bacteria.isAlive() );

	} // run






















//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// DEBUGGING:
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------














	
	template<int dim>
    void MicrobeBDF2<dim>::create_mass_check_vector()
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
	void MicrobeBDF2<dim>::initialize_base_objects_debug(const ArgParser& systemParameters)
	{
		time_step = systemParameters.getTimeStep();
		run_time = systemParameters.getRunTime();
		save_period = systemParameters.getSavePeriod();
		outputDirectory = systemParameters.getOutputDirectory();

		if(systemParameters.isCheckMass() || systemParameters.isReproduceBacteria())
		{
			std::cout << "...setting chemical diffusion and decay constants" << std::endl;
			good_diffusion_constant = systemParameters.getGoodDiffusionConstant();
			waste_diffusion_constant = systemParameters.getWasteDiffusionConstant();
			good_decay_constant = systemParameters.getGoodDecayConstant();
			waste_decay_constant = systemParameters.getWasteDecayConstant();
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
	void MicrobeBDF2<dim>::setup_system_debug(const ArgParser& systemParameters)
	{
  		initialize_base_objects_debug(systemParameters);
  		if(systemParameters.isCheckMass() || systemParameters.isReproduceBacteria())
  		{
	  		setup_constraints();
	  		initialize_vectors_matrices();
	  	}
	} // setup system() *** maybe split assemble system


	template<int dim>
	void MicrobeBDF2<dim>::printMassToFile(std::vector<double> massOne, 
		std::vector<double> massTwo) const
	{
	  const std::string filename = outputDirectory + "/massVsTime.dat";

	  std::ofstream output(filename.c_str());

	  unsigned int n = massOne.size();
	  for(unsigned int i = 0; i < n; i++)
	    output << massOne[i] << "\t" << massTwo[i] << std::endl;
	} // printMassToFile()


	template<int dim>
	void MicrobeBDF2<dim>::projectInitialCondition(Point<dim> gaussian_center)
	{
		  GaussianFE<dim> initialFunction(gaussian_center);

            // ConstraintMatrix constraints;
            // constraints.close();
            VectorTools::project (dof_handler,
                                  constraints,
                                  QGauss<dim>(fe.degree+2),
                                  initialFunction,
                                  old_public_goods); 

            VectorTools::project (dof_handler,
                              constraints,
                              QGauss<dim>(fe.degree+2),
                              initialFunction,
                              old_waste_chemical); 

            public_goods = old_public_goods;
            waste_chemical = old_waste_chemical;
	}


	template<int dim>
	void MicrobeBDF2<dim>::updateMatrices(bool usePointSources)
	{
		mass_matrix.vmult(goods_rhs, old_public_goods);  // Matrix vector multiplication (MU^{n-1}) -- overwrites system_rhs
		mass_matrix.vmult(waste_rhs, old_waste_chemical);


		// LEFT HAND SIDE TERMS:
		goods_system_matrix.copy_from(mass_matrix); // LHS = M
		goods_system_matrix *= 1 + good_decay_constant*time_step; //  for decay term -> LHS = (1+k \lam) M
		goods_system_matrix.add(time_step * good_diffusion_constant, stiffness_matrix);   //  for diffusion term -> LHS = (1+k \lam) M + kd*L

		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
		{
			goods_system_matrix.add(time_step, advection_matrix); // with a minus sign?
			std::cout << "updating chemical 1 with advection " << std::endl;
		}
		// goods_system_matrix.add(-1.0*time_step, robin_matrix);

		waste_system_matrix.copy_from(mass_matrix);
		waste_system_matrix *= 1 + waste_decay_constant*time_step; // *** for decay term
		waste_system_matrix.add(time_step * waste_diffusion_constant, stiffness_matrix);

		if(advectionField.getVelocityType()!= VelocityType::NO_FLOW)
			waste_system_matrix.add(time_step, advection_matrix); // with a minus sign?
		// waste_system_matrix.add(-1.0*time_step, robin_matrix);


		// use point sources
		if(usePointSources)
		{
			updatePointSources(); // updates tmp and tmp2 

			goods_rhs.add(time_step, tmp1); // RHS = k*n*s
			waste_rhs.add(time_step, tmp2);
		}

		constraints.condense (goods_system_matrix, goods_rhs); 
		constraints.condense (waste_system_matrix, waste_rhs);
	}


	template<int dim>
	Point<dim> MicrobeBDF2<dim>::findGaussianCenterPoint() const
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
	void MicrobeBDF2<dim>::run_debug(const ArgParser& systemParameters)
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
				old_public_goods = public_goods;
				old_waste_chemical = waste_chemical;	

			    // === UPDATE CHECK VECTORS: ===
			    if(systemParameters.isCheckMass())
			    {
					total_mass_one.push_back( public_goods*mass_check_vector );
					total_mass_two.push_back( waste_chemical*mass_check_vector );
				}
			}
			if(systemParameters.isReproduceBacteria())
			{
			    if(time > systemParameters.getReproductionDelay())
					bacteria.reproduce(time_step, dof_handler, public_goods, waste_chemical);
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
*/

}

#endif // MicrobeBDF2.h




