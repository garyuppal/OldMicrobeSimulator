#ifndef MICROBE_SIMULATOR_DG_SIMULATOR_H
#define MICROBE_SIMULATOR_DG_SIMULATOR_H


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/grid/grid_tools.h>

// mesh worker:
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_tools.h>



#include <deal.II/integrators/laplace.h>


#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>

// mutligrid methods:
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "../utility/argparser.h"
#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "../bacteria/bacteria.h"
#include "../utility/my_grid_generator.h"

#include "./exact_solutions.h"

namespace MicrobeSimulator{ namespace DGSim{
using namespace dealii;



// ===========================================================================
// FOR DIFFUSION MATRIX:

template <int dim>
class MatrixIntegrator : public MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(MeshWorker::DoFInfo<dim> &                 dinfo,
            typename MeshWorker::IntegrationInfo<dim> &info) const override;
  void
       boundary(MeshWorker::DoFInfo<dim> &                 dinfo,
                typename MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(MeshWorker::DoFInfo<dim> &                 dinfo1,
            MeshWorker::DoFInfo<dim> &                 dinfo2,
            typename MeshWorker::IntegrationInfo<dim> &info1,
            typename MeshWorker::IntegrationInfo<dim> &info2) const override;
};


template <int dim>
void 
MatrixIntegrator<dim>::cell(
  MeshWorker::DoFInfo<dim> &                 dinfo,
  typename MeshWorker::IntegrationInfo<dim> &info) const
{
  LocalIntegrators::Laplace::cell_matrix(dinfo.matrix(0, false).matrix,
                                         info.fe_values());
}


template <int dim>
void 
MatrixIntegrator<dim>::boundary(
  MeshWorker::DoFInfo<dim> &                 dinfo,
  typename MeshWorker::IntegrationInfo<dim> &info) const
{}
// {
//   const unsigned int degree = info.fe_values(0).get_fe().tensor_degree();
//   LocalIntegrators::Laplace::nitsche_matrix(
//     dinfo.matrix(0, false).matrix,
//     info.fe_values(0),
//     LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, degree, degree));
// } /// might use this for robin BCs *****

// interior faces:
template <int dim>
void 
MatrixIntegrator<dim>::face(
  MeshWorker::DoFInfo<dim> &                 dinfo1,
  MeshWorker::DoFInfo<dim> &                 dinfo2,
  typename MeshWorker::IntegrationInfo<dim> &info1,
  typename MeshWorker::IntegrationInfo<dim> &info2) const
{
  const unsigned int degree = info1.fe_values(0).get_fe().tensor_degree();
  LocalIntegrators::Laplace::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    LocalIntegrators::Laplace::compute_penalty(
      dinfo1, dinfo2, degree, degree));
}

// GLOBAL ADVECTION FIELD ==========================================================
AdvectionField<2> advection_field; // at least its in the namespace of this simulator -- but need dim...

template<int dim>
class DG_Simulator
{
public:
	DG_Simulator();
	void run(const ArgParser& parameters);
	void run_debug(const ArgParser& parameters);
private:
	Geometry<dim> geometry;
	// AdvectionField<dim> advection_field;
	Bacteria<dim> bacteria;

	// FINITE ELEMENTS:
	const unsigned int fe_degree;

	Triangulation<dim>    triangulation;
	const MappingQ1<dim>  mapping; //*** check if can improve this ..., also can we use Q2?

	FE_DGQ<dim>           fe;
	DoFHandler<dim>       dof_handler;

	SparsityPattern     sparsity_pattern;

	SparseMatrix<double>  mass_matrix;    // also acts as system matrix
	SparseMatrix<double>  advection_matrix;
	SparseMatrix<double>  diffusion_matrix;

	SparseMatrix<double>  goods_system_matrix;

	Vector<double>        public_goods;
	Vector<double>        old_public_goods;

	Vector<double>        goods_rhs;

	Vector<double>		  exact_solution;

    double good_diffusion_constant;
    double good_decay_constant;


    // time stepping and saving:
    double time;
    double time_step;
    unsigned int time_step_number;

    unsigned int save_period;
    unsigned int save_step_number;

    double run_time;
    std::string output_directory;

    ConvergenceTable 		convergence_table;

    //functions
    //------------------------------------------------------
	void setup_system(const ArgParser& parameters);
	void setup_system_constants(const ArgParser& parameters);
	void setup_geometry(const ArgParser& parameters);
	void setup_advection(const ArgParser& parameters);
	// void setup_bacteria(const ArgParser& parameters);
	void setup_grid(const ArgParser& parameters, unsigned int cycle = 0);
	void setup_periodicity();
	void refine_grid(unsigned int global_refinement, 
	unsigned int sphere_refinement);
	void refineMeshSpheres(unsigned int sphere_refinement);
	void output_grid();
	void setup_dofs();
	void initialize_vectors_matrices();
	void assemble_matrices();
	void assemble_mass_matrix();
	void assemble_diffusion_matrix();
	void assemble_advection_matrix();

	void update_system();
	void solve();
    void output_chemicals();

	double get_CFL_time_step();
	double get_maximal_velocity();
    
    // debugging:
    void process_solution(unsigned int cycle, const Function<dim>& sol);
    void project_initial_condition(const Function<dim>& intial_condition);
    void output_convergence_table();

    void projectExactSolution(const Function<dim>& sol);

    // void update_exact_solution_vector(const Function<dim>& sol);
    void process_time_solution(const Function<dim>& sol,
    		std::vector<double>&	L_infty,
    		std::vector<double>&	L2,
    		std::vector<double>&	H1);
    void output_vector(const std::vector<double>& data, std::string file);

	/*
	// CAN KEEP THIS AS STATIC, BUT WILL NEED TO DECLARE ADVECTION OUTSIDE
	// OF CLASs -- but then we would need access to velocity field...
	*/
	/* static */ 
	static void integrate_advection_cell_term(MeshWorker::DoFInfo<dim> &dinfo,
	                                   MeshWorker::IntegrationInfo<dim> &info);
	/* static */ 
	static void integrate_advection_face_term(MeshWorker::DoFInfo<dim> & dinfo1,
	                              MeshWorker::DoFInfo<dim> & dinfo2,
	                              MeshWorker::IntegrationInfo<dim> &info1,
	                              MeshWorker::IntegrationInfo<dim> &info2);
	// using NULL for boundary terms

	/* static */ 
	static void integrate_mass_cell_term(MeshWorker::DoFInfo<dim> &dinfo, 
	                                MeshWorker::IntegrationInfo<dim> &info);
};


// IMPLEMENTATION
//--------------------------------------------------------------------------------------------

template<int dim>
DG_Simulator<dim>::DG_Simulator()
	:
	fe_degree(1),
	mapping(),
	fe(fe_degree),
	dof_handler(triangulation),
	time(0),
	time_step_number(0),
	save_step_number(0),
	output_directory("./")
{}


template<int dim>
void 
DG_Simulator<dim>::setup_system(const ArgParser& parameters)
{
	setup_system_constants(parameters);
	setup_geometry(parameters);
	setup_advection(parameters);
	setup_grid(parameters);
	setup_dofs();
	initialize_vectors_matrices();
	assemble_matrices();

	time_step = get_CFL_time_step(); // *** later use CFL condition function...
}


template<int dim>
double 
DG_Simulator<dim>::get_CFL_time_step()
{
	const double min_time_step = 0.01;

	const double maximal_velocity = get_maximal_velocity();
	double cfl_time_step = 0;

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
DG_Simulator<dim>::get_maximal_velocity()
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


template<int dim>
void 
DG_Simulator<dim>::setup_system_constants(const ArgParser& parameters)
{
	std::cout << "...Setting up system constants" << std::endl;
	run_time = parameters.getRunTime();
	save_period = parameters.getSavePeriod();
	good_diffusion_constant = parameters.getGoodDiffusionConstant();
	good_decay_constant = parameters.getGoodDecayConstant();
	output_directory = parameters.getOutputDirectory();
}


template<int dim>
void 
DG_Simulator<dim>::setup_geometry(const ArgParser& parameters)
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
DG_Simulator<dim>::setup_advection(const ArgParser& parameters)
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


// void setup_bacteria(const ArgParser& parameters);


template<int dim>
void 
DG_Simulator<dim>::setup_grid(const ArgParser& parameters, unsigned int cycle)
{
	std::cout << "...Setting up grid" << std::endl;
	MyGridGenerator<dim> gridGen;
	gridGen.generateGrid(geometry,triangulation); 

	setup_periodicity();
	refine_grid(parameters.getGlobalRefinement() + cycle, parameters.getSphereRefinement());
	output_grid();
}


template<int dim>
void
DG_Simulator<dim>::setup_periodicity()
{
	    // make periodic before refinement:
    std::vector<GridTools::PeriodicFacePair<TriaIterator<CellAccessor<dim,dim> > > >
	    periodicity_vector;

	for(unsigned int i = 0; i < dim; i++)
	{
		if(geometry.getBoundaryConditions()[i] == BoundaryCondition::WRAP)
		{
			std::cout << "\n...USING PERIODIC " << i <<"th BOUNDARY" << std::endl;

			const unsigned int direction = i;
			unsigned int bid_one = 0 + 2*i;
			unsigned int bid_two = 1 + 2*i;

			GridTools::collect_periodic_faces(triangulation, bid_one, bid_two, direction,
			                            periodicity_vector); 
		} // if periodic 
	} // for each dimension
  	triangulation.add_periodicity(periodicity_vector);
}

template<int dim>
void 
DG_Simulator<dim>::refine_grid(unsigned int global_refinement, 
	unsigned int sphere_refinement)
{
	if(dim == 2)
    	refineMeshSpheres(sphere_refinement);

    triangulation.refine_global(global_refinement); 
    std::cout << "...Mesh refined globally: " << global_refinement << " times" << std::endl;
}


template<int dim>
void 
DG_Simulator<dim>::refineMeshSpheres(unsigned int sphere_refinement)
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
DG_Simulator<dim>::output_grid()
{
	std::string grid_out_file = output_directory + "/grid.eps";

	std::ofstream out (grid_out_file);
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "...Grid written to " << grid_out_file << std::endl;
}


template<int dim>
void 
DG_Simulator<dim>::setup_dofs()
{
	dof_handler.distribute_dofs(fe);
	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);

	std::cout << std::endl << std::endl
	      << "================================================"
	      << std::endl
	      << "Number of active cells: " << triangulation.n_active_cells()
	      << std::endl
	      << "Number of degrees of freedom: " << dof_handler.n_dofs()
	      << std::endl
	      << std::endl;
}


template<int dim>
void 
DG_Simulator<dim>::initialize_vectors_matrices()
{
	mass_matrix.reinit(sparsity_pattern);
	advection_matrix.reinit(sparsity_pattern);
	diffusion_matrix.reinit(sparsity_pattern);
	
	goods_system_matrix.reinit(sparsity_pattern);

	public_goods.reinit(dof_handler.n_dofs());
	old_public_goods.reinit(dof_handler.n_dofs());
	goods_rhs.reinit(dof_handler.n_dofs());
	// temporary_holder.reinit(dof_handler.n_dofs());
	exact_solution.reinit(dof_handler.n_dofs());
}


template<int dim>
void 
DG_Simulator<dim>::assemble_matrices()
{
	assemble_mass_matrix();
	assemble_diffusion_matrix();
	assemble_advection_matrix();
}


template<int dim>	
void 
DG_Simulator<dim>::assemble_mass_matrix()
{
	MeshWorker::IntegrationInfoBox<dim> info_box;
	const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1; // should this be 2*n + 1? ***
	info_box.initialize_gauss_quadrature(n_gauss_points,
	                                   n_gauss_points,
	                                   n_gauss_points); // for cells, boundary, interior faces
	info_box.initialize_update_flags();
	UpdateFlags update_flags = 
	update_quadrature_points | update_values | update_gradients;
	info_box.add_update_flags(update_flags,true,true,true,true);

	// initialize objects:
	info_box.initialize(fe,mapping);

	// to recieve integrated data and forward to assembler:
	MeshWorker::DoFInfo<dim> dof_info(dof_handler);

	// create assembler objects and tell them where to put local data:
	MeshWorker::Assembler::MatrixSimple<SparseMatrix<double> >
	mass_assembler;
	mass_assembler.initialize(mass_matrix);

	// are we providing too much information? can we use same infobox for both matrices?
	MeshWorker::loop<dim,dim,
	              MeshWorker::DoFInfo<dim>,
	              MeshWorker::IntegrationInfoBox<dim> >(
	dof_handler.begin_active(),
	dof_handler.end(),
	dof_info,
	info_box,
	&DG_Simulator<dim>::integrate_mass_cell_term,
	NULL,
	NULL,
	mass_assembler);
}


template<int dim>
void 
DG_Simulator<dim>::assemble_diffusion_matrix()
{
	MeshWorker::IntegrationInfoBox<dim> info_box;
	UpdateFlags update_flags = update_values | update_gradients;
	info_box.add_update_flags_all(update_flags);
	info_box.initialize(fe, mapping);

	MeshWorker::DoFInfo<dim> dof_info(dof_handler);

	MeshWorker::Assembler::MatrixSimple<SparseMatrix<double> >
		diff_assembler;
	diff_assembler.initialize(diffusion_matrix);

	MatrixIntegrator<dim> integrator;

  	MeshWorker::integration_loop<dim,dim>(dof_handler.begin_active(),
                                         dof_handler.end(),
                                         dof_info,
                                         info_box,
                                         integrator,
                                         diff_assembler);
}


template<int dim>
void 
DG_Simulator<dim>::assemble_advection_matrix()
{
	MeshWorker::IntegrationInfoBox<dim> info_box;
	const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
	info_box.initialize_gauss_quadrature(n_gauss_points,
	                                   n_gauss_points,
	                                   n_gauss_points); // for cells, boundary, interior faces
	info_box.initialize_update_flags();
	UpdateFlags update_flags = 
	update_quadrature_points | update_values | update_gradients;
	info_box.add_update_flags(update_flags,true,true,true,true);

	// initialize objects:
	info_box.initialize(fe,mapping);

	// to recieve integrated data and forward to assembler:
	MeshWorker::DoFInfo<dim> dof_info(dof_handler);

	// create assembler objects and tell them where to put local data:
	MeshWorker::Assembler::MatrixSimple<SparseMatrix<double> >
	advection_assembler;
	advection_assembler.initialize(advection_matrix);

	// are we providing too much information? can we use same infobox for both matrices?
	MeshWorker::loop<dim,dim,
	              MeshWorker::DoFInfo<dim>,
	              MeshWorker::IntegrationInfoBox<dim> >(
	dof_handler.begin_active(),
	dof_handler.end(),
	dof_info,
	info_box,
	&DG_Simulator<dim>::integrate_advection_cell_term, // cell
	NULL, // boundary
	&DG_Simulator<dim>::integrate_advection_face_term, // face
	advection_assembler);
}


// helper functions for matrix construction:
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
template<int dim>
void 
DG_Simulator<dim>::integrate_mass_cell_term(MeshWorker::DoFInfo<dim> &dinfo, 
                                    MeshWorker::IntegrationInfo<dim> &info)
{
    const FEValuesBase<dim> &  fe_values    = info.fe_values();
    FullMatrix<double> &       local_matrix = dinfo.matrix(0).matrix; // alias for matrix in dinfo!
    const std::vector<double> &JxW          = fe_values.get_JxW_values();

    // with these objects, can continue integration as usual:

    // loop over quadrature points:
    for (unsigned int point = 0; point < fe_values.n_quadrature_points; ++point)
    {

      // this problem is homogeneous, hence no rhs
      // matrix entries:
      for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
          local_matrix(i, j) += fe_values.shape_value(i, point) *  
                                fe_values.shape_value(j, point) * 
                                JxW[point];
    }
} // integrate_adv_cell_term


template<int dim>
void 
DG_Simulator<dim>::integrate_advection_cell_term(MeshWorker::DoFInfo<dim> &dinfo,
                                       MeshWorker::IntegrationInfo<dim> &info)
{
    const FEValuesBase<dim> &  fe_values    = info.fe_values();
    FullMatrix<double> &       local_matrix = dinfo.matrix(0).matrix;
    const std::vector<double> &JxW          = fe_values.get_JxW_values();

    // with these objects, can continue integration as usual:

    // loop over quadrature points:
    for (unsigned int point = 0; point < fe_values.n_quadrature_points; ++point)
    {
      const Tensor<1, dim> beta_at_q_point =
        advection_field.value(fe_values.quadrature_point(point)); // my velocity function

      // std::cout << "beta at quadrature point:" << std::endl;
      // std::cout <<  beta_at_q_point << std::endl;

      // this problem is homogeneous, hence no rhs
      // matrix entries:
      for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
          local_matrix(i, j) += -beta_at_q_point *                
                                fe_values.shape_grad(i, point) *  
                                fe_values.shape_value(j, point) * 
                                JxW[point];
    }
} // integrate_adv_cell_term


// *** REVIEW FUNCTIONS USED BELOW AND CHECK WHAT NEEDS TO BE DONE TO MAKE PERIODIC
template<int dim>
void 
DG_Simulator<dim>::integrate_advection_face_term(MeshWorker::DoFInfo<dim> & dinfo1,
                                  MeshWorker::DoFInfo<dim> & dinfo2,
                                  MeshWorker::IntegrationInfo<dim> &info1,
                                  MeshWorker::IntegrationInfo<dim> &info2)
{
    const FEValuesBase<dim> &fe_face_values = info1.fe_values();
    const unsigned int       dofs_per_cell  = fe_face_values.dofs_per_cell;

    const FEValuesBase<dim> &fe_face_values_neighbor = info2.fe_values();
    const unsigned int       neighbor_dofs_per_cell =
      fe_face_values_neighbor.dofs_per_cell;

    FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0, false).matrix;
    FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0, true).matrix;
    FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0, true).matrix;
    FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0, false).matrix;

    const std::vector<double> &        JxW = fe_face_values.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals =
      fe_face_values.get_normal_vectors();
    for (unsigned int point = 0; point < fe_face_values.n_quadrature_points;
         ++point)
      {
        const double beta_dot_n =
          advection_field.value(fe_face_values.quadrature_point(point)) * normals[point];
        if (beta_dot_n > 0)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                u1_v1_matrix(i, j) += beta_dot_n *                           
                                      fe_face_values.shape_value(j, point) * 
                                      fe_face_values.shape_value(i, point) * 
                                      JxW[point];

        // also assemble (\beta \cdot n u, \hat{v})_{\partial \kappa_+}
          for (unsigned int k = 0; k < neighbor_dofs_per_cell; ++k)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                u1_v2_matrix(k, j) +=
                  -beta_dot_n *                                   
                  fe_face_values.shape_value(j, point) *          
                  fe_face_values_neighbor.shape_value(k, point) * 
                  JxW[point];
        }
        else
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int l = 0; l < neighbor_dofs_per_cell; ++l)
                u2_v1_matrix(i, l) +=
                  beta_dot_n *                                    
                  fe_face_values_neighbor.shape_value(l, point) * 
                  fe_face_values.shape_value(i, point) *          
                  JxW[point];

          // also assemble (\beta \cdot n u, \hat{v})_{\partial \kappa_-}
          for (unsigned int k = 0; k < neighbor_dofs_per_cell; ++k)
            for (unsigned int l = 0; l < neighbor_dofs_per_cell; ++l)
              u2_v2_matrix(k, l) +=
                -beta_dot_n *                                   
                fe_face_values_neighbor.shape_value(l, point) * 
                fe_face_values_neighbor.shape_value(k, point) * 
                JxW[point];
        } // if-else : for direction of v 
    } // for quadrature points
}// integrate face term
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

template<int dim>
void
DG_Simulator<dim>::update_system()
{
	goods_system_matrix.copy_from(mass_matrix);
	goods_system_matrix.add(time_step, advection_matrix);
	goods_system_matrix.add(time_step*good_diffusion_constant, diffusion_matrix);

	// RHS:
	mass_matrix.vmult(goods_rhs, old_public_goods);
}

// *** FIGURE OUT WHAT SOLVER AND PRECONDITIONER IS BEST:
template<int dim>
void 
DG_Simulator<dim>::solve()
{
  SolverControl solver_control(1000, 1e-12);
  // SolverRichardson<> solver(solver_control);
  SolverBicgstab<> bicgstab(solver_control);

  PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
  preconditioner.initialize(goods_system_matrix,
                            fe.dofs_per_cell);

  bicgstab.solve(goods_system_matrix, public_goods, goods_rhs, preconditioner);
}


template<int dim>
void 
DG_Simulator<dim>::output_chemicals()
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


// ERROR ANALYSIS AND DEBUGGING: 
//-------------------------------------------------------------------------------------
template<int dim>
void 
DG_Simulator<dim>::process_solution(unsigned int cycle, 
										const Function<dim>& sol)
{
	Vector<float>	difference_per_cell(triangulation.n_active_cells());

	// compute L2-error
	VectorTools::integrate_difference(dof_handler,
										public_goods,
										sol,
										difference_per_cell,
										QGauss<dim>(3),
										VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation,
															difference_per_cell,
															VectorTools::L2_norm);

	// compute H1-error
	VectorTools::integrate_difference(dof_handler,
										public_goods,
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
										public_goods,
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
void 
DG_Simulator<dim>::project_initial_condition(const Function<dim>& intial_condition)
{
	ConstraintMatrix constraints;
	constraints.close();

     VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(fe_degree+2),
	                      intial_condition,
	                      old_public_goods);
     public_goods = old_public_goods;
}


template<int dim>
void 
DG_Simulator<dim>::projectExactSolution(const Function<dim>& sol)
{
	ConstraintMatrix constraints;
	constraints.close();

     VectorTools::project (dof_handler,
	                      constraints,
	                      QGauss<dim>(fe_degree+2),
	                      sol,
	                      exact_solution);
}


template<int dim>
void 
DG_Simulator<dim>::output_convergence_table()
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

	std::string filename = output_directory + "/convergence_table.tex";
	std::ofstream table_file(filename);
	convergence_table.write_tex(table_file);
}

template<int dim>
void 
DG_Simulator<dim>::process_time_solution(const Function<dim>& sol,
		std::vector<double>&	L_infty,
		std::vector<double>&	L2,
		std::vector<double>&	H1)
{
	Vector<float>	difference_per_cell(triangulation.n_active_cells());

	// compute L2-error
	VectorTools::integrate_difference(dof_handler,
										public_goods,
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
										public_goods,
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
										public_goods,
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
DG_Simulator<dim>::output_vector(const std::vector<double>& data, std::string file)
{
	const std::string filename = output_directory + "/" + file + ".dat";
	std::ofstream out(filename);

	for(unsigned int i = 0; i<data.size(); i++)
		out << data[i] << std::endl;
}













// RUN
//------------------------------------------------------------------------------------------
template<int dim>
void
DG_Simulator<dim>::run(const ArgParser& parameters)
{
	setup_system(parameters);

	project_initial_condition(	ExactSolutions::Gaussian<dim>(Point<dim>(2.,3.)) );

	output_chemicals();

    do{
    	update_system();
    	solve();

    	// old_old_public_goods = old_public_goods;
    	old_public_goods = public_goods;

    	time += time_step;
    	++time_step_number;
	    	if(time_step_number % save_period == 0)
	    	{
	    		++save_step_number;
	    		std::cout << "time: " << time << std::endl;
	    		output_chemicals();
	    	}
	    } while(time < run_time);
}



template<int dim>
void 
DG_Simulator<dim>::run_debug(const ArgParser& parameters)
{
	const unsigned int n_cycles = 5;
    
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
	    	update_system();
	    	solve();

	    	// old_old_public_goods = old_public_goods;
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
	    	ExactSolutions::GaussXVelocityPeriodic<dim>(good_diffusion_constant,
	    		parameters.getMaximumVelocity(),time));
	} // for cycles

	output_convergence_table();
}


}} // close namespace

#endif // DG_SIMULATOR
