#ifndef MICROBESIMULATOR_CUSTOM_STOKES_SOLVER_H
#define MICROBESIMULATOR_CUSTOM_STOKES_SOLVER_H



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

// #include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <memory>

#include "../utility/argparser.h"
#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "../bacteria/bacteria.h"
#include "../bacteria/fitness.h"

#include "../utility/my_grid_generator.h"

#include "../utility/grid_generation_tools.h"


#include "./exact_solutions.h"
#include "./cell_iterator_map.h"

namespace MicrobeSimulator{ namespace VelocityStokes{
	using namespace dealii;
  // using PointCellMap::CellIteratorMap;


template <int dim>
struct InnerPreconditioner;

template <>
struct InnerPreconditioner<2>
{
    typedef SparseILU<double> type;
//	typedef SparseDirectUMFPACK type;
};
template <>
struct InnerPreconditioner<3>
{
	typedef SparseILU<double> type;
};



// Bacteria Fintess:
template<int dim>
class OnePG_Fitness : public FitnessBase<dim, 2>
{
public:
    OnePG_Fitness() {}
    ~OnePG_Fitness() {}

    virtual double value(const Point<dim>& location, 
        const std::array<double, 2>& secretion_rates) const;

    void attach_chemicals(DoFHandler<dim>* dofPtr,
                          Vector<double>* goodsPtr,
                          Vector<double>* wastePtr);

    void setup_fitness_constants(double pg_bene, double w_harm, 
      double pg_sat, double w_sat, 
      double beta);

    void printInfo(std::ostream& out) const;

  private: 
    DoFHandler<dim>* dof_pointer;
    Vector<double>* goods_pointer;
    Vector<double>* waste_pointer; // can probably also make friends...
    // std::array<FDMChemical<dim>*, NumberChemicals> chemical_pointers; 
    // @ todo -- could do array of smart pointers?

    double public_good_benefit;
    double waste_harm;

    double public_good_saturation;
    double waste_saturation;

    double secretion_cost; // same for 1 and 2 -- zero for 3
};


template<int dim>
double 
OnePG_Fitness<dim>::value(const Point<dim>& location, 
        const std::array<double, 2>& secretion_rates) const
{
  const double c1 = VectorTools::point_value(*dof_pointer,
                                            *goods_pointer,
                                            location);

  const double c2 = VectorTools::point_value(*dof_pointer,
                                            *waste_pointer,
                                            location);

  const double return_value =
      public_good_benefit * c1 / ( c1 + public_good_saturation )
      - waste_harm * c2 / ( c2 + waste_saturation )
      - secretion_cost * secretion_rates[0];

  return return_value;
}


template<int dim>
void 
OnePG_Fitness<dim>::attach_chemicals(DoFHandler<dim>* dofPtr,
                      Vector<double>* goodsPtr,
                      Vector<double>* wastePtr)
{
  dof_pointer = dofPtr;
  goods_pointer = goodsPtr;
  waste_pointer = wastePtr;
}


template<int dim>
void 
OnePG_Fitness<dim>::setup_fitness_constants(double pg_bene, double w_harm, 
  double pg_sat, double w_sat, 
  double beta)
{
  public_good_benefit = pg_bene;
  waste_harm = w_harm;
  public_good_saturation = pg_sat;
  waste_saturation = w_sat;
  secretion_cost = beta;
}


template<int dim>
void 
OnePG_Fitness<dim>::printInfo(std::ostream& out) const
{
  out << "\n\n-----------------------------------------------------" << std::endl
    << "\t\tFITNESS FUNCTION (for " << 2 << " chemicals)" << std::endl
    << "-----------------------------------------------------" << std::endl
    << "\t public good benefit: " << public_good_benefit << std::endl
    << "\t waste harm: " << waste_harm << std::endl
    << "\t public good saturation: " << public_good_saturation << std::endl
    << "\t waste saturation: " << waste_saturation << std::endl
    << "\t secretion cost: " << secretion_cost << std::endl;
}


// SOLVER:
  template <int dim>
  class StokesSolver
  {
  public:
    StokesSolver (const unsigned int stokes_degree);
    void run (const ArgParser& parameters);
    void test_initial_field(const ArgParser& parameters);
    void check_source();
    void run_check_map_sources(const ArgParser& parameters);
    void check_grid(const ArgParser& parameters);

    void run_pipe_splitter(const ArgParser& parameters);

    void run_check_stokes(const ArgParser& parameters);

  private:
    void setup_dofs (unsigned int number_spheres, unsigned int number_rectangles);
    void assemble_stokes_system ();
    void solve_stokes ();
    void output_results (const unsigned int refinement_cycle) const;
    void refine_mesh ();

    double get_maximal_velocity() const;
    void assemble_chemical_matrices();

    double get_CFL_time_step();

    void assemble_chemical_system(const double maximal_velocity, const bool use_sources);
    void updateSources();

    void update_sources_from_map();

    double compute_viscosity(const std::vector<Tensor<1,dim> >& velocity_values,
                const double cell_diameter);

    void solve_chemicals();
    void output_chemicals() const;
    void output_bacteria() const;
    void output_sources() const;

    // STOKES:
    // --------------------------------------------------------------------------

    const unsigned int   stokes_degree;
    Triangulation<dim>   triangulation;

    // STOKES:
    FESystem<dim>        stokes_fe;
    DoFHandler<dim>      stokes_dof_handler;

    ConstraintMatrix          stokes_constraints;
    BlockSparsityPattern      stokes_sparsity_pattern;
    BlockSparseMatrix<double> stokes_system_matrix;

    BlockSparsityPattern      preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;

    BlockVector<double> stokes_solution;
    BlockVector<double> stokes_rhs;

    std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;


    // CHEMICALS:
    //-------------------------------------------------------------------------
      Geometry<dim> geometry;
      AdvectionField<dim> advection_field;
      Bacteria<dim, 2> bacteria; // two chemicals

      OnePG_Fitness<dim> fitness_function;

      PointCellMap<dim> point_cell_map;

      const unsigned int      chemical_fe_degree;
      FE_Q<dim>               chemical_fe;
      DoFHandler<dim>         chemical_dof_handler;
      // AffineConstraints<dim>  chemical_constraints;

      ConstraintMatrix        chemical_constraints;
      SparsityPattern         chemical_sparsity_pattern;

      SparseMatrix<double>    chemical_mass_matrix;
      SparseMatrix<double>    chemical_diffusion_matrix;

      SparseMatrix<double>    goods_system_matrix;
      Vector<double>          public_goods;
      Vector<double>          old_public_goods;
      Vector<double>          old_old_public_goods;
      Vector<double>          goods_rhs;
      Vector<double>          goods_source;

      SparseMatrix<double>    waste_system_matrix;
      Vector<double>          waste;
      Vector<double>          old_waste;
      Vector<double>          old_old_waste;
      Vector<double>          waste_rhs;
      Vector<double>          waste_source;

      Vector<double>          temporary;

      // SYSTEM CONSTANTS:
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

      // FUNCTIONS:    
      // -------------------------------------------------------------------------
      // void setup_system(const ArgParser& parameters);
      void setup_system_constants(const ArgParser& parameters);
      void setup_geometry(const ArgParser& parameters);
      void setup_point_cell_map(const double resolution); 
      void setup_advection(const ArgParser& parameters);
      void setup_bacteria(const ArgParser& parameters);
      void setup_fitness(const ArgParser& parameters);

      std::vector<Point<dim> > getBacteriaLocations(unsigned int number_bacteria, unsigned int number_groups);
      std::vector<Point<dim> > getMixerLocations(unsigned int number_groups);

      void setup_grid(const ArgParser& parameters, unsigned int cycle = 0);
      void refine_grid(unsigned int global_refinement, 
        unsigned int sphere_refinement);
      void refineMeshSpheres(unsigned int sphere_refinement);
      void output_grid();
      void output_grid_vtk();

      void project_initial_condition(const Function<dim> &initial_condition,
                                        Vector<double>& numerical_field);
  };



//BOUNDARY CONDITIONS
//--------------------------------------------------------------------------------------------

  // INLET BOUNDARY:
  template <int dim>
  class InletBoundary : public Function<dim>
  {
  public:
    InletBoundary () : Function<dim>(dim+1) {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  InletBoundary<dim>::value (const Point<dim>& /* p */,
                              const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));
    if (component == 0)
      return 1; // along x direction
    return 0;
  }


  template <int dim>
  void
  InletBoundary<dim>::vector_value (const Point<dim> &p,
                                     Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = InletBoundary<dim>::value (p, c);
  }



  // NO SLIP:
  template <int dim>
  class NoSlipBoundary : public Function<dim>
  {
  public:
    NoSlipBoundary () : Function<dim>(dim+1) {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  NoSlipBoundary<dim>::value (const Point<dim>& /* p */,
                              const unsigned int  component ) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));
    // if (component == 0)
    //   return 1; // (p[0] < 0 ? -1 : (p[0] > 0 ? 1 : 0));
    return 0;
  }

  template <int dim>
  void
  NoSlipBoundary<dim>::vector_value (const Point<dim> &p,
                                     Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = NoSlipBoundary<dim>::value (p, c);
  }

//RIGHT HAND SIDE
//--------------------------------------------------------------------------------------------

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(dim+1) {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  double
  RightHandSide<dim>::value (const Point<dim>  &/*p*/,
                             const unsigned int /*component*/) const
  {
    return 0;
  }

  template <int dim>
  void
  RightHandSide<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = RightHandSide<dim>::value (p, c);
  }


// INVERSE MATRIX
//--------------------------------------------------------------------------------------------
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix (const MatrixType         &m,
                   const PreconditionerType &preconditioner);
    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;
  private:
    const SmartPointer<const MatrixType> matrix;
    const SmartPointer<const PreconditionerType> preconditioner;
  };


  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType,PreconditionerType>::InverseMatrix
  (const MatrixType         &m,
   const PreconditionerType &preconditioner)
    :
    matrix (&m),
    preconditioner (&preconditioner)
  {}


  template <class MatrixType, class PreconditionerType>
  void InverseMatrix<MatrixType,PreconditionerType>::vmult
  (Vector<double>       &dst,
   const Vector<double> &src) const
  {
    SolverControl solver_control (src.size(), 1e-6*src.l2_norm()); // *** large tolerance for speed
    SolverCG<>    cg (solver_control);
    dst = 0;
    cg.solve (*matrix, dst, src, *preconditioner);
  }



// SCHUR COMPLEMENT
//--------------------------------------------------------------------------------------------
  template <class PreconditionerType>
  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement (const BlockSparseMatrix<double> &stokes_system_matrix,
                     const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);
    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;
  private:
    const SmartPointer<const BlockSparseMatrix<double> > stokes_system_matrix;
    const SmartPointer<const InverseMatrix<SparseMatrix<double>, PreconditionerType> > A_inverse;
    mutable Vector<double> tmp1, tmp2;
  };


  template <class PreconditionerType>
  SchurComplement<PreconditionerType>::SchurComplement
  (const BlockSparseMatrix<double>                              &stokes_system_matrix,
   const InverseMatrix<SparseMatrix<double>,PreconditionerType> &A_inverse)
    :
    stokes_system_matrix (&stokes_system_matrix),
    A_inverse (&A_inverse),
    tmp1 (stokes_system_matrix.block(0,0).m()),
    tmp2 (stokes_system_matrix.block(0,0).m())
  {}


  template <class PreconditionerType>
  void SchurComplement<PreconditionerType>::vmult (Vector<double>       &dst,
                                                   const Vector<double> &src) const
  {
    stokes_system_matrix->block(0,1).vmult (tmp1, src);
    A_inverse->vmult (tmp2, tmp1);
    stokes_system_matrix->block(1,0).vmult (dst, tmp2);
  }


//IMPLEMENTATION
//--------------------------------------------------------------------------------------------

template <int dim>
StokesSolver<dim>::StokesSolver (const unsigned int stokes_degree)
  :
  stokes_degree (stokes_degree),
  triangulation (Triangulation<dim>::maximum_smoothing),
  stokes_fe (FE_Q<dim>(stokes_degree+1), dim, FE_Q<dim>(stokes_degree), 1), // taylor hood elements ***
  stokes_dof_handler (triangulation)
  ,
  chemical_fe_degree(1),
  chemical_fe( chemical_fe_degree ),
  chemical_dof_handler(triangulation)
  ,
  time(0),
  time_step_number(0),
  save_step_number(0),
  output_directory("."),
  reproduction_delay(5)
{}


template <int dim>
double 
StokesSolver<dim>::get_maximal_velocity() const
{
  const QIterated<dim> quadrature_formula(QTrapez<1>(), stokes_degree + 1);
  const unsigned int   n_q_points = quadrature_formula.size();
  FEValues<dim> fe_values(stokes_fe, quadrature_formula, update_values);
  std::vector<Tensor<1, dim> > velocity_values(n_q_points);

  double max_velocity = 0;

  const FEValuesExtractors::Vector velocities(0);
  for (const auto &cell : stokes_dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[velocities].get_function_values(stokes_solution,
                                                velocity_values);
      for (unsigned int q = 0; q < n_q_points; ++q)
        max_velocity = std::max(max_velocity, velocity_values[q].norm());
    }
  return max_velocity;
}


template <int dim>
void 
StokesSolver<dim>::setup_dofs (unsigned int number_spheres, unsigned int number_rectangles)
{
  A_preconditioner.reset ();
  stokes_system_matrix.clear ();
  preconditioner_matrix.clear ();
  stokes_dof_handler.distribute_dofs (stokes_fe);
  DoFRenumbering::Cuthill_McKee (stokes_dof_handler); // renumber DOFS to help ILU ***

  std::vector<unsigned int> block_component (dim+1,0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise (stokes_dof_handler, block_component);

  // DIRECHLET BOUNDARY CONDITIONS:
  {
    const unsigned int inlet_boundary_id = 0; // 1 is open
    std::vector<unsigned int> no_slip_boundary_id = {2, 3}; //, 10, 11};
    if(dim == 3)
    {
      no_slip_boundary_id.push_back(4);
      no_slip_boundary_id.push_back(5);
    }

    unsigned int sphere_id = GridGenerationTools::id_sphere_begin;
    for(unsigned int i = 0; i < number_spheres; ++i)
    {
      no_slip_boundary_id.push_back(sphere_id);
      ++sphere_id;
    }
    unsigned int rect_id = GridGenerationTools::id_rectangle_begin;
    for(unsigned int i = 0; i < number_rectangles; ++i)
    {
      no_slip_boundary_id.push_back(rect_id);
      ++rect_id; 
    }

    const unsigned int n_noslip_boundaries = no_slip_boundary_id.size(); 

    stokes_constraints.clear ();
    FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                                             stokes_constraints);

    VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                              inlet_boundary_id,  // boundary component (boundary_id???) ***
                                              InletBoundary<dim>(),
                                              stokes_constraints,
                                              stokes_fe.component_mask(velocities)); // to only apply to velocities
 
    for(unsigned int i = 0; i < n_noslip_boundaries; i++)
    {
      std::cout << " no slip bnds: " << no_slip_boundary_id[i] << std::endl;
      VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                                no_slip_boundary_id[i],  // boundary component (boundary_id???) ***
                                                NoSlipBoundary<dim>(),
                                                stokes_constraints,
                                                stokes_fe.component_mask(velocities)); // to only apply to velocities
    }
  }
  stokes_constraints.close ();

  // chemical constraints:
  {
    chemical_dof_handler.distribute_dofs(chemical_fe);
    chemical_constraints.clear();
    DoFTools::make_hanging_node_constraints(chemical_dof_handler,
                                            chemical_constraints);
    chemical_constraints.close();
  }

  std::vector<types::global_dof_index> dofs_per_block (2);
  DoFTools::count_dofs_per_block (stokes_dof_handler, 
                                  dofs_per_block, 
                                  block_component);
  const unsigned int n_u = dofs_per_block[0],
                     n_p = dofs_per_block[1];
  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Number of degrees of freedom: "
            << stokes_dof_handler.n_dofs()
            << " (" << n_u << '+' << n_p << ')'
            << std::endl;

// scoping helps to release memory after operation complete: ******
  {
    BlockDynamicSparsityPattern dsp (2,2);
    dsp.block(0,0).reinit (n_u, n_u);
    dsp.block(1,0).reinit (n_p, n_u);
    dsp.block(0,1).reinit (n_u, n_p);
    dsp.block(1,1).reinit (n_p, n_p);
    dsp.collect_sizes();
    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (! ((c==dim) && (d==dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;
    DoFTools::make_sparsity_pattern (stokes_dof_handler, 
                                    coupling, dsp, 
                                    stokes_constraints, 
                                    false);
    stokes_sparsity_pattern.copy_from (dsp);
  }

  {
    BlockDynamicSparsityPattern preconditioner_dsp (2,2);
    preconditioner_dsp.block(0,0).reinit (n_u, n_u);
    preconditioner_dsp.block(1,0).reinit (n_p, n_u);
    preconditioner_dsp.block(0,1).reinit (n_u, n_p);
    preconditioner_dsp.block(1,1).reinit (n_p, n_p);
    preconditioner_dsp.collect_sizes();
    Table<2,DoFTools::Coupling> preconditioner_coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (((c==dim) && (d==dim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;
    DoFTools::make_sparsity_pattern (stokes_dof_handler, preconditioner_coupling,
                                     preconditioner_dsp, stokes_constraints, false);
    preconditioner_sparsity_pattern.copy_from (preconditioner_dsp);
  }

  // initialize objects:
  stokes_system_matrix.reinit (stokes_sparsity_pattern);
  preconditioner_matrix.reinit (preconditioner_sparsity_pattern);

  stokes_solution.reinit (2);
  stokes_solution.block(0).reinit (n_u);
  stokes_solution.block(1).reinit (n_p);
  stokes_solution.collect_sizes ();

  stokes_rhs.reinit (2);
  stokes_rhs.block(0).reinit (n_u);
  stokes_rhs.block(1).reinit (n_p);
  stokes_rhs.collect_sizes ();

  // chemical objects:
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
  goods_system_matrix.reinit(chemical_sparsity_pattern);
  waste_system_matrix.reinit(chemical_sparsity_pattern);

  public_goods.reinit(chemical_dof_handler.n_dofs());
  old_public_goods.reinit(chemical_dof_handler.n_dofs());
  old_old_public_goods.reinit(chemical_dof_handler.n_dofs());
  goods_rhs.reinit(chemical_dof_handler.n_dofs());
  goods_source.reinit(chemical_dof_handler.n_dofs());

  waste.reinit(chemical_dof_handler.n_dofs());
  old_waste.reinit(chemical_dof_handler.n_dofs());
  old_old_waste.reinit(chemical_dof_handler.n_dofs());
  waste_rhs.reinit(chemical_dof_handler.n_dofs());
  waste_source.reinit(chemical_dof_handler.n_dofs());

  temporary.reinit(chemical_dof_handler.n_dofs());
  // exact_solution.reinit(chemical_dof_handler.n_dofs());

}


template <int dim>
void 
StokesSolver<dim>::assemble_stokes_system ()
{
  stokes_system_matrix=0;
  stokes_rhs=0;
  preconditioner_matrix = 0;

  QGauss<dim>   quadrature_formula(stokes_degree+2);
  FEValues<dim> stokes_fe_values (stokes_fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>   local_preconditioner_matrix (dofs_per_cell, dofs_per_cell);

  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const RightHandSide<dim>          right_hand_side; // *** defined in program above
  std::vector<Vector<double> >      rhs_values (n_q_points,
                                                Vector<double>(dim+1));

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
  std::vector<double>                  div_phi_u   (dofs_per_cell);
  std::vector<double>                  phi_p       (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = stokes_dof_handler.begin_active(),
  endc = stokes_dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      stokes_fe_values.reinit (cell);
      local_matrix = 0;
      local_preconditioner_matrix = 0;
      local_rhs = 0;
      right_hand_side.vector_value_list(stokes_fe_values.get_quadrature_points(),
                                        rhs_values);
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              symgrad_phi_u[k] = stokes_fe_values[velocities].symmetric_gradient (k, q);
              div_phi_u[k]     = stokes_fe_values[velocities].divergence (k, q);
              phi_p[k]         = stokes_fe_values[pressure].value (k, q);
            }
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            { // *** BILINEAR FORMS ***
              for (unsigned int j=0; j<=i; ++j) // *** loop upto i only, because symmetric ***
                {
                  local_matrix(i,j) += (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])
                                        - div_phi_u[i] * phi_p[j]
                                        - phi_p[i] * div_phi_u[j])
                                       * stokes_fe_values.JxW(q);
                  local_preconditioner_matrix(i,j) += (phi_p[i] * phi_p[j])
                                                      * stokes_fe_values.JxW(q);
                }
              const unsigned int component_i =
                stokes_fe.system_to_component_index(i).first;
              local_rhs(i) += stokes_fe_values.shape_value(i,q) *
                              rhs_values[q](component_i) *
                              stokes_fe_values.JxW(q);
            }
        }
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=i+1; j<dofs_per_cell; ++j)
          {
            local_matrix(i,j) = local_matrix(j,i);
            local_preconditioner_matrix(i,j) = local_preconditioner_matrix(j,i); 
            // flip indices to fill in rest ***
          }
      cell->get_dof_indices (local_dof_indices);
      stokes_constraints.distribute_local_to_global (local_matrix, local_rhs,
                                              local_dof_indices,
                                              stokes_system_matrix, stokes_rhs);
      stokes_constraints.distribute_local_to_global (local_preconditioner_matrix,
                                              local_dof_indices,
                                              preconditioner_matrix);
    }
  // *** make preconditioner  
  std::cout << "   Computing preconditioner..." << std::endl << std::flush;

  A_preconditioner
    = std::make_shared<typename InnerPreconditioner<dim>::type>();
  A_preconditioner->initialize (stokes_system_matrix.block(0,0),
                                typename InnerPreconditioner<dim>::type::AdditionalData());
}


template <int dim>
void 
StokesSolver<dim>::solve_stokes ()
{
  const InverseMatrix<SparseMatrix<double>,
        typename InnerPreconditioner<dim>::type>
        A_inverse (stokes_system_matrix.block(0,0), *A_preconditioner);

  Vector<double> tmp (stokes_solution.block(0).size());

  {
    Vector<double> schur_rhs (stokes_solution.block(1).size());
    A_inverse.vmult (tmp, stokes_rhs.block(0));
    stokes_system_matrix.block(1,0).vmult (schur_rhs, tmp);
    schur_rhs -= stokes_rhs.block(1);
    SchurComplement<typename InnerPreconditioner<dim>::type>
    schur_complement (stokes_system_matrix, A_inverse);
    SolverControl solver_control (stokes_solution.block(1).size(),
                                  1e-6*schur_rhs.l2_norm());
    SolverCG<>    cg (solver_control);
    SparseILU<double> preconditioner;
    preconditioner.initialize (preconditioner_matrix.block(1,1),
                               SparseILU<double>::AdditionalData());
    InverseMatrix<SparseMatrix<double>,SparseILU<double> >
    m_inverse (preconditioner_matrix.block(1,1), preconditioner);
    cg.solve (schur_complement, stokes_solution.block(1), schur_rhs,
              m_inverse);
    stokes_constraints.distribute (stokes_solution);
    std::cout << "  "
              << solver_control.last_step()
              << " outer CG Schur complement iterations for pressure"
              << std::endl;
  }
  {
    stokes_system_matrix.block(0,1).vmult (tmp, stokes_solution.block(1));
    tmp *= -1;
    tmp += stokes_rhs.block(0);
    A_inverse.vmult (stokes_solution.block(0), tmp);
    stokes_constraints.distribute (stokes_solution);
  }
}


template <int dim>
void
StokesSolver<dim>::output_results (const unsigned int refinement_cycle)  const
{
  std::vector<std::string> stokes_solution_names (dim, "velocity");
  stokes_solution_names.emplace_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  data_component_interpretation
  (dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation
  .push_back (DataComponentInterpretation::component_is_scalar);
  DataOut<dim> data_out;
  data_out.attach_dof_handler (stokes_dof_handler);
  data_out.add_data_vector (stokes_solution, stokes_solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
  data_out.build_patches ();
  std::ofstream output (output_directory
                        + "/stokes_solution-"
                        + Utilities::int_to_string(refinement_cycle, 2)
                        + ".vtk");
  data_out.write_vtk (output);
}


// *** use KellyErrorEstimator, only for pressure to flag refinement ***
template <int dim>
void
StokesSolver<dim>::refine_mesh ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
  FEValuesExtractors::Scalar pressure(dim);
  KellyErrorEstimator<dim>::estimate (stokes_dof_handler,
                                      QGauss<dim-1>(stokes_degree+1),
                                      typename FunctionMap<dim>::type(),
                                      stokes_solution,
                                      estimated_error_per_cell,
                                      stokes_fe.component_mask(pressure));
  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.0);
  triangulation.execute_coarsening_and_refinement ();
}

// NEW STUFF
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// template<int dim>
// void 
// StokesSolver<dim>::setup_system(const ArgParser& parameters)
// {
//   setup_system_constants(parameters);
//   setup_geometry(parameters);
//   setup_advection(parameters);
//   setup_bacteria(parameters);
//   setup_grid(parameters);
//   setup_dofs();
//   initialize_vectors_matrices();
//   assemble_matrices();

//   std::cout << "matrices assembled, getting time step" << std::endl;

//   time_step = get_CFL_time_step();
// }


template<int dim>
void 
StokesSolver<dim>::setup_system_constants(const ArgParser& parameters)
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
StokesSolver<dim>::setup_geometry(const ArgParser& parameters)
{
  std::cout << "...Initializing geometry" << std::endl;
  geometry.initialize(parameters.getGeometryFile(), 
    parameters.getMeshFile());
  
  std::string geo_out_file = output_directory + "/geometryInfo.dat";
  std::ofstream geo_out(geo_out_file);
  geometry.printInfo(geo_out);
  geometry.printInfo(std::cout);

  geometry.outputGeometry(output_directory); // boundary and obstacles for easy viewing
}


template<int dim>
void 
StokesSolver<dim>::setup_point_cell_map(const double resolution)
{
  std::cout << "...Setting up point-cell map" << std::endl;
  point_cell_map.initialize(geometry, chemical_dof_handler, resolution); // using default resolution
  point_cell_map.printInfo(std::cout); 
}


template<int dim>
void 
StokesSolver<dim>::setup_advection(const ArgParser& parameters)
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
StokesSolver<dim>::setup_bacteria(const ArgParser& parameters)
{
  const unsigned int number_bacteria = parameters.getNumberBacteria();

  if(number_bacteria < 1)
    return;

  std::cout << "...Initializing bacteria" << std::endl;
  // std::vector<Point<dim> > locations ={Point<2>(0,0)}; 

  std::cout << "\t... using mixer locations" << std::endl;

  const double scale_factor = 3.;

  std::vector<Point<2> > loc2d = {Point<2>(0.5 ,0.25), Point<2>(0.5 ,0.75)};
  std::vector<Point<3> > loc3d = {Point<3>(scale_factor*0.25,scale_factor*0.5,scale_factor*0.5)}; 
    //{Point<3>(0.5 ,0.25,0.25), Point<3>(0.5 ,0.75, 0.75)};
  // }

  std::vector<Point<dim> > locations = loc3d; // = (dim == 2 ? loc2d : loc3d); 



  const unsigned int initial_number_cheaters = 1; // need to readd this...

    std::array<double, 2> rates = {
        parameters.getGoodSecretionRate(),
        parameters.getWasteSecretionRate()
      };

  bacteria.initialize(parameters.getBacteriaDiffusionConstant(), 
            number_bacteria,
            rates,
            locations); 
            // parameters.getGoodSecretionRate(),
            // parameters.getWasteSecretionRate(),
            // initial_number_cheaters );

  // std::cout << "...Setting fitness constants" << std::endl;
  // bacteria.setFitnessConstants(parameters.getAlphaGood(),
  //               parameters.getAlphaWaste(),
  //               parameters.getGoodSaturation(),
  //               parameters.getWasteSaturation(),
  //               parameters.getSecretionCost() );
    bacteria.printInfo(std::cout);
}


template<int dim>
std::vector<Point<dim> > 
StokesSolver<dim>::getBacteriaLocations(unsigned int number_bacteria, 
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
StokesSolver<dim>::getMixerLocations(unsigned int number_groups)
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
StokesSolver<dim>::setup_fitness(const ArgParser& parameters)
{
  fitness_function.attach_chemicals(&chemical_dof_handler,
                                    &public_goods,
                                    &waste);

  fitness_function.setup_fitness_constants(parameters.getAlphaGood(),
                                          parameters.getAlphaWaste(),
                                          parameters.getGoodSaturation(),
                                          parameters.getWasteSaturation(),
                                          parameters.getSecretionCost());

  fitness_function.printInfo(std::cout);
}



template<int dim>
void 
StokesSolver<dim>::setup_grid(const ArgParser& parameters, 
  unsigned int cycle)
{
  std::cout << "...Setting up grid" << std::endl;
  MyGridGenerator::generateGrid<dim>(geometry,triangulation); 

  refine_grid(parameters.getGlobalRefinement() + cycle, parameters.getSphereRefinement());
  output_grid();
}


template<int dim>
void 
StokesSolver<dim>::refine_grid(unsigned int global_refinement, 
  unsigned int sphere_refinement)
{
  if(dim == 2)
      refineMeshSpheres(sphere_refinement);

    triangulation.refine_global(global_refinement); 
    std::cout << "...Mesh refined globally: " << global_refinement << " times" << std::endl;
}


template<int dim>
void 
StokesSolver<dim>::refineMeshSpheres(unsigned int sphere_refinement)
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
StokesSolver<dim>::output_grid()
{
  std::string grid_out_file = output_directory + "/grid.eps";

  std::ofstream out (grid_out_file);
  GridOut grid_out;
  grid_out.write_eps (triangulation, out);
  std::cout << "...Grid written to " << grid_out_file << std::endl;
}


template<int dim>
void 
StokesSolver<dim>::output_grid_vtk()
{
  std::string grid_out_file = output_directory + "/grid.vtk";

  std::ofstream out (grid_out_file);
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);
  std::cout << "...Grid written to " << grid_out_file << std::endl;
}



template<int dim>
void 
StokesSolver<dim>::assemble_chemical_matrices()
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
    }

    cell->get_dof_indices(local_dof_indices);
    chemical_constraints.distribute_local_to_global(local_mass_matrix,
                                            local_dof_indices,
                                            chemical_mass_matrix);
    chemical_constraints.distribute_local_to_global(local_diffusion_matrix,
                                            local_dof_indices,
                                            chemical_diffusion_matrix);
  }
}


template<int dim>
double 
StokesSolver<dim>::get_CFL_time_step()
{
  const double min_time_step = 0.01;

  const double maximal_velocity = get_maximal_velocity();
  double cfl_time_step = 0;

  std::cout << "maximal_velocity is: " << maximal_velocity << std::endl;

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
StokesSolver<dim>::assemble_chemical_system(const double maximal_velocity, const bool use_sources)
{
  // const bool use_sources = false;
  const bool use_bdf2_scheme = (time_step_number != 0);
  // const double beta = ;

  if(use_bdf2_scheme == true)
  {
    // goods:
    goods_system_matrix.copy_from(chemical_mass_matrix);
    goods_system_matrix *= (1.5 + time_step*good_decay_constant); 
    // not too sure about this..., else treat  decay explictly
    goods_system_matrix.add(time_step*good_diffusion_constant, 
                            chemical_diffusion_matrix);

    // waste:
    waste_system_matrix.copy_from(chemical_mass_matrix);
    waste_system_matrix *= (1.5 + time_step*waste_decay_constant); 
    waste_system_matrix.add(time_step*waste_diffusion_constant, 
                            chemical_diffusion_matrix);
  }
  else
  {
    // goods:
    goods_system_matrix.copy_from(chemical_mass_matrix); 
    goods_system_matrix *= (1.0 + time_step*good_decay_constant);
    goods_system_matrix.add(time_step*good_diffusion_constant, 
                            chemical_diffusion_matrix); 

    // waste:
    waste_system_matrix.copy_from(chemical_mass_matrix); 
    waste_system_matrix *= (1.0 + time_step*waste_decay_constant);
    waste_system_matrix.add(time_step*waste_diffusion_constant, 
                            chemical_diffusion_matrix); // diffusion treated explicitly
  }


  // RHS:
  goods_rhs = 0;
  waste_rhs = 0;

  const QGauss<dim> quadrature_formula(chemical_fe_degree + 2);

  FEValues<dim> chemical_fe_values(chemical_fe,
                            quadrature_formula,                                           
                            update_values    |
                            update_gradients |
                            update_hessians  |
                            update_quadrature_points  |
                            update_JxW_values);

  FEValues<dim> stokes_fe_values(stokes_fe,
                                quadrature_formula,
                                update_values);

  const unsigned int dofs_per_cell = chemical_fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  Vector<double>  goods_local_rhs(dofs_per_cell);
  Vector<double>  waste_local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // velocity:
  std::vector<Tensor<1, dim> >  velocity_values(n_q_points); 

  // goods:
  std::vector<double>           old_goods_values(n_q_points);
  std::vector<double>           old_old_goods_values(n_q_points);
  std::vector<Tensor<1,dim> >   old_goods_grads(n_q_points);
  std::vector<Tensor<1,dim> >   old_old_goods_grads(n_q_points);
  std::vector<double>           old_goods_laplacians(n_q_points);
  std::vector<double>           old_old_goods_laplacians(n_q_points);

  // waste:
  std::vector<double>           old_waste_values(n_q_points);
  std::vector<double>           old_old_waste_values(n_q_points);
  std::vector<Tensor<1,dim> >   old_waste_grads(n_q_points);
  std::vector<Tensor<1,dim> >   old_old_waste_grads(n_q_points);
  std::vector<double>           old_waste_laplacians(n_q_points);
  std::vector<double>           old_old_waste_laplacians(n_q_points);

  // sources:
  // std::vector<double>  good_source_values(n_q_points);
  // std::vector<double>  waste_source_values(n_q_points);
  // if(use_sources == true)
  //  updateSources();


  // shape functions:
  std::vector<double>           phi_T(dofs_per_cell);
  std::vector<Tensor<1,dim> >   grad_phi_T(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  

  //typename DoFHandler<dim>::active_cell_iterator 
  auto cell         = chemical_dof_handler.begin_active();
  const auto endc   = chemical_dof_handler.end();
  auto stokes_cell  = stokes_dof_handler.begin_active();

  for(; cell!=endc; ++cell, ++stokes_cell)
  {
    goods_local_rhs = 0;
    waste_local_rhs = 0;

    chemical_fe_values.reinit(cell);
    stokes_fe_values.reinit(stokes_cell);  // *** should only have to do this once!!!
    //RESTUCTURE CODE TO EXTRACT VELOCITIES ONLY ONCE ... -- can use timestep (ie using bdf),
      // compute only if timestep == 0, and store in scope of class
    // WILL ALSO NEED TO FIGURE OUT HOW TO PASS VELOCITY TO BACTERIA...

    // update goods:
    chemical_fe_values.get_function_values(old_public_goods, old_goods_values);
    chemical_fe_values.get_function_values(old_old_public_goods, old_old_goods_values);

    chemical_fe_values.get_function_gradients(old_public_goods, old_goods_grads);
    chemical_fe_values.get_function_gradients(old_old_public_goods, old_old_goods_grads);

    chemical_fe_values.get_function_laplacians(old_public_goods, old_goods_laplacians);
    chemical_fe_values.get_function_laplacians(old_old_public_goods, old_old_goods_laplacians);
      
    // update waste:
    chemical_fe_values.get_function_values(old_waste, old_waste_values);
    chemical_fe_values.get_function_values(old_old_waste, old_old_waste_values);

    chemical_fe_values.get_function_gradients(old_waste, old_waste_grads);
    chemical_fe_values.get_function_gradients(old_old_waste, old_old_waste_grads);

    chemical_fe_values.get_function_laplacians(old_waste, old_waste_laplacians);
    chemical_fe_values.get_function_laplacians(old_old_waste, old_old_waste_laplacians);

    // update sources:  
    // if(use_sources == true)
    // {
    //  fe_values.get_function_values(goods_source, good_source_values);
    //  fe_values.get_function_values(waste_source, waste_source_values);
    // }

    // update velocity:
    // advection_field.value_list(fe_values.get_quadrature_points(), velocity_values); 
    stokes_fe_values[velocities].get_function_values(stokes_solution, velocity_values);

    // compute viscosity:
    const double nu_goods = compute_viscosity(velocity_values, cell->diameter());
    const double nu_waste = nu_goods; // using same for now -- doesn't depend on residual,
    // only depends on velocity and cell size

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
                        chemical_fe_values.JxW(q); 

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
                        chemical_fe_values.JxW(q); 
          }
    }

      cell->get_dof_indices (local_dof_indices);
      chemical_constraints.distribute_local_to_global (goods_local_rhs,
                                              local_dof_indices,
                                              goods_rhs);
      chemical_constraints.distribute_local_to_global (waste_local_rhs,
                                              local_dof_indices,
                                              waste_rhs);
    } // loop over cells

    if(use_sources == true)
    {
      // updateSources();
      update_sources_from_map();
      goods_rhs.add(time_step, goods_source);
      waste_rhs.add(time_step, waste_source);
    }
}


template<int dim>
void
StokesSolver<dim>::updateSources() // use a mapping from a structured grid to speed up ***
{
  goods_source = 0;
  waste_source = 0;

  const unsigned int number_bacteria = bacteria.getSize();
  const double scale_factor = 0.0016; 
 
  for(unsigned int i = 0; i < number_bacteria; i++)
  {
    const double good_secretion = bacteria.getGoodSecretionRate(i);
    const double waste_secretion = bacteria.getWasteSecretionRate(i);
// std::cout << "gs: " << good_secretion << " ws: " << waste_secretion << std::endl;
    VectorTools::create_point_source_vector(chemical_dof_handler,
                                            bacteria.getLocation(i),
                                            temporary);

    goods_source.add(scale_factor*good_secretion, temporary);
    waste_source.add(scale_factor*waste_secretion, temporary);
  }
}


template<int dim>
void
StokesSolver<dim>::update_sources_from_map()
{
  goods_source = 0;
  waste_source = 0;

  const double scale_factor = 0.0016;
  const unsigned int number_bacteria = bacteria.getSize();

  const unsigned int dofs_per_cell = chemical_dof_handler.get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  // loop over bacteria:
  for(unsigned int i = 0; i < number_bacteria; i++)
  {
    const double good_secretion = bacteria.getSecretionRate(i,0);
    const double waste_secretion = bacteria.getSecretionRate(i,1);

    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
    cell_point = 
      point_cell_map.get_cell_point_pair(bacteria.getLocation(i), &geometry);

    Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim> fe_values(StaticMappingQ1<dim>::mapping, 
                        chemical_dof_handler.get_fe(),
                        q, 
                        UpdateFlags(update_values));

    fe_values.reinit(cell_point.first);

    cell_point.first->get_dof_indices (local_dof_indices);

    /* shoudlnt this be += ?? */

    for (unsigned int i=0; i<dofs_per_cell; i++)
    {
      goods_source(local_dof_indices[i]) += //check ***
          scale_factor*good_secretion*fe_values.shape_value(i,0);
      waste_source(local_dof_indices[i]) += 
          scale_factor*waste_secretion*fe_values.shape_value(i,0);
    }
  }

}


template<int dim>
double 
StokesSolver<dim>::compute_viscosity(const std::vector<Tensor<1,dim> >& velocity_values,
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
StokesSolver<dim>::solve_chemicals()
{
  {
    SolverControl solver_control(1000, 1e-8 * goods_rhs.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> chemical_preconditioner;
    chemical_preconditioner.initialize(goods_system_matrix, 1.0);
    cg.solve(goods_system_matrix, public_goods, goods_rhs,
           chemical_preconditioner);
    chemical_constraints.distribute(public_goods);
  }
  {
    SolverControl solver_control(1000, 1e-8 * waste_rhs.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> chemical_preconditioner;
    chemical_preconditioner.initialize(waste_system_matrix, 1.0);
    cg.solve(waste_system_matrix, waste, waste_rhs,
           chemical_preconditioner);
    chemical_constraints.distribute(waste);
  }

   // update solutions:
  old_old_public_goods = old_public_goods;
  old_public_goods = public_goods;
  old_old_waste = old_waste;
  old_waste = waste;
}


template<int dim>
void 
StokesSolver<dim>::output_chemicals() const
{
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(chemical_dof_handler);
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
    data_out.attach_dof_handler(chemical_dof_handler);
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
StokesSolver<dim>::output_bacteria() const
{
  std::string outfile = output_directory
    + "/bacteria-" + std::to_string(save_step_number) + ".dat";
  std::ofstream out(outfile);
  bacteria.print(out);
}


template<int dim>
void 
StokesSolver<dim>::output_sources() const
{
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(chemical_dof_handler);
    data_out.add_data_vector(goods_source,"goods_source");
    data_out.build_patches();
    const std::string filename = output_directory
                  + "/goods_source-"
                  + Utilities::int_to_string(save_step_number,4)
                  + ".vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(chemical_dof_handler);
    data_out.add_data_vector(waste_source,"waste_source");
    data_out.build_patches();
    const std::string filename = output_directory
                  + "/waste_source-"
                  + Utilities::int_to_string(save_step_number,4)
                  + ".vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }
}


template<int dim>
void
StokesSolver<dim>::project_initial_condition(const Function<dim> &initial_condition,
                                        Vector<double>& numerical_field)
{
       VectorTools::project (chemical_dof_handler,
                        chemical_constraints,
                        QGauss<dim>(chemical_fe_degree+2),
                        initial_condition,
                        numerical_field);
}


// RUN
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
template<int dim>
void
StokesSolver<dim>::run_pipe_splitter(const ArgParser& parameters)
{
  setup_system_constants(parameters);
  setup_geometry(parameters);
  setup_bacteria(parameters);

  const double scale_factor = 3.0;

  {
    MyGridGenerator::square_pipe_with_spherical_hole<dim>(triangulation, scale_factor);

    triangulation.refine_global(2);
    output_grid();
  }

  const unsigned int number_refinements = 3; // 5;

  for (unsigned int refinement_cycle = 0; refinement_cycle < number_refinements;
       ++refinement_cycle)
    {
      std::cout << "Refinement cycle " << refinement_cycle << std::endl;
      if (refinement_cycle > 0)
        refine_mesh ();
      setup_dofs ();
      std::cout << "   Assembling..." << std::endl << std::flush;
      assemble_stokes_system ();
      std::cout << "   Solving..." << std::flush;
      solve_stokes ();
      output_results (refinement_cycle);
      std::cout << std::endl;
    }

    time_step = get_CFL_time_step();

    run_time = 5.0; //100*time_step; //1.0; 
    save_period = std::ceil(run_time/(100*time_step));

    assemble_chemical_matrices();

    setup_point_cell_map(0.2); // might be too fine ...

    const double max_velocity = get_maximal_velocity();

    std::cout << "\nloop through time...\n" << std::endl;

    setup_fitness(parameters);

    do{
      // update chemicals:
      assemble_chemical_system(max_velocity, true);
      solve_chemicals();

      // // update bacteria:
      bacteria.randomWalk(time_step, 
                          &geometry,
                          stokes_dof_handler,
                          stokes_solution);

      bacteria.reproduce(time_step, fitness_function); 
                        // chemical_dof_handler,
                        // public_goods,
                        // waste);

      // mutate ...

      // update time:
      time += time_step;
      ++time_step_number;

      // output:
      if(time_step_number % save_period == 0)
      {
        ++save_step_number;
        std::cout << "time: " << time << std::endl;

        output_bacteria();
        output_chemicals();
        output_sources();
      } 

    } while( (time < run_time) && bacteria.isAlive() );


}


template <int dim>
void 
StokesSolver<dim>::run (const ArgParser& parameters)
{
  setup_system_constants(parameters);
  setup_geometry(parameters);
  setup_bacteria(parameters); // *** with 1 initial cheater ***

  {
    MyGridGenerator::generateGrid<dim>(geometry, triangulation);
    // gridGen.generateGridWithHoles(geometry,triangulation);

    triangulation.refine_global(3);
    output_grid();
  }

  // triangulation.refine_global (1);
  

  const unsigned int number_refinements = 3; // 5;

  for (unsigned int refinement_cycle = 0; refinement_cycle < number_refinements;
       ++refinement_cycle)
    {
      std::cout << "Refinement cycle " << refinement_cycle << std::endl;
      if (refinement_cycle > 0)
        refine_mesh ();
      setup_dofs ();
      std::cout << "   Assembling..." << std::endl << std::flush;
      assemble_stokes_system ();
      std::cout << "   Solving..." << std::flush;
      solve_stokes ();
      output_results (refinement_cycle);
      std::cout << std::endl;
    }

    // test random walk:
    time_step = get_CFL_time_step();

    run_time = 5.0; //100*time_step; //1.0; 
    save_period = 10;

    assemble_chemical_matrices();

    const double max_velocity = get_maximal_velocity();

    std::cout << "\nloop through time...\n" << std::endl;
    do{
      // update chemicals:
      assemble_chemical_system(max_velocity, true);
      solve_chemicals();

      // // update bacteria:
      // bacteria.randomWalk(time_step, 
      //                     &geometry,
      //                     stokes_dof_handler,
      //                     stokes_solution);

      // bacteria.reproduce(time_step,
      //                   chemical_dof_handler,
      //                   public_goods,
      //                   waste);

      // mutate ...

      // update time:
      time += time_step;
      ++time_step_number;

      // output:
      if(time_step_number % save_period == 0)
      {
        ++save_step_number;
        std::cout << "time: " << time << std::endl;

        output_bacteria();
        output_chemicals();
      } 

    } while( (time < run_time) && bacteria.isAlive() );



} // run()


template<int dim>
void
StokesSolver<dim>::test_initial_field(const ArgParser& parameters)
{
  setup_system_constants(parameters);
  setup_geometry(parameters);

  {
    MyGridGenerator::generateGrid<dim>(geometry, triangulation);
    // gridGen.generateGridWithHoles(geometry,triangulation);

    triangulation.refine_global(3);
    output_grid();
  }

  // triangulation.refine_global (1);
  

  const unsigned int number_refinements = 3; // 5;

  for (unsigned int refinement_cycle = 0; refinement_cycle < number_refinements;
       ++refinement_cycle)
    {
      std::cout << "Refinement cycle " << refinement_cycle << std::endl;
      if (refinement_cycle > 0)
        refine_mesh ();
      setup_dofs ();
      std::cout << "   Assembling..." << std::endl << std::flush;
      assemble_stokes_system ();
      std::cout << "   Solving..." << std::flush;
      solve_stokes ();
      output_results (refinement_cycle);
      std::cout << std::endl;
    }

  

    time_step = get_CFL_time_step();
    std::cout << "using time step " << time_step << std::endl;

    run_time = 10.1;
    save_period = 100;

    // TEST WITH INITIAL CONDITION:
    const double initial_width = 0.5;

    project_initial_condition(ExactSolutions::Gaussian<dim>(Point<dim>(1.,1.5),initial_width),
                              public_goods);

    project_initial_condition(ExactSolutions::Gaussian<dim>(Point<dim>(1.5,1.5),initial_width),
                              waste);

    old_public_goods = public_goods;
    old_waste = waste;
    output_chemicals();

    assemble_chemical_matrices();
    do{
      assemble_chemical_system(get_maximal_velocity());
      solve_chemicals();

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
StokesSolver<dim>::check_source()
{
  // generate grid:
      {
      std::vector<unsigned int> subdivisions (dim, 1);
      subdivisions[0] = 4;
      const Point<dim> bottom_left = (dim == 2 ?
                                      Point<dim>(-2,-1) :
                                      Point<dim>(-2,0,-1));
      const Point<dim> top_right   = (dim == 2 ?
                                      Point<dim>(2,0) :
                                      Point<dim>(2,1,0));
      GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                 subdivisions,
                                                 bottom_left,
                                                 top_right);
    }


    triangulation.refine_global(2);
    output_grid();

    setup_dofs(); // split chemicals and stokes ...

    Point<2> location_one(-1.25,-0.8); // left of center
    Point<2> location_two(1.0, -0.3); 

    std::vector<Point<2> > locations = {location_one, location_two};

    // mapping from point to cell:
    std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > >
    test_iterator_map;

    for(unsigned int point_itr = 0; point_itr < locations.size(); point_itr++)
    {
        std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
        cell_point =
          GridTools::find_active_cell_around_point (StaticMappingQ1<dim>::mapping, 
                                                    chemical_dof_handler, 
                                                    locations[point_itr]);
          test_iterator_map.push_back(cell_point);
    }




    const double good_secretion = 10000.;
    const double waste_secretion = 100.;
    const double scale_factor = 0.0016; 

    // before loop:
    public_goods = 0;
    waste = 0;

    const unsigned int dofs_per_cell = chemical_dof_handler.get_fe().dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


    // loop over locations:
    for(unsigned int point_itr = 0; point_itr < locations.size(); point_itr++)
    {
        std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
        cell_point = test_iterator_map[point_itr];

        Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

        FEValues<dim> fe_values(StaticMappingQ1<dim>::mapping, 
                                chemical_dof_handler.get_fe(),
                                q, 
                                UpdateFlags(update_values));
        
        fe_values.reinit(cell_point.first);


        cell_point.first->get_dof_indices (local_dof_indices);

        for (unsigned int i=0; i<dofs_per_cell; i++)
        {
          std::cout << "fe value: " << fe_values.shape_value(i,0) << std::endl;
          public_goods(local_dof_indices[i]) = 
                scale_factor*good_secretion*fe_values.shape_value(i,0);
          waste(local_dof_indices[i]) = 
                scale_factor*waste_secretion*fe_values.shape_value(i,0);
        }

    }// loop over points

    output_chemicals();

}


template<int dim>
void 
StokesSolver<dim>::run_check_map_sources(const ArgParser& parameters)
{
  setup_system_constants(parameters);
  setup_geometry(parameters);
  setup_bacteria(parameters); // *** with 1 initial cheater ***


  {
    MyGridGenerator::generateGrid<dim>(geometry, triangulation);
    // gridGen.generateGridWithHoles(geometry,triangulation);

    triangulation.refine_global(4);
    output_grid();
  }


  setup_dofs();

  // const double resolution = 0.1;
  // std::cout << "...setting up point-cell map" << std::endl;
  // setup_point_cell_map(resolution); 


  time_step = 0.001;
  run_time = 0.010;
  save_period = 1;

  std::cout << "\nRUNNING SOURCE CHECK\n" << std::endl;

  do{
      bacteria.randomWalk(time_step, 
                          &geometry); 
                          // ,
                          // stokes_dof_handler,
                          // stokes_solution);

      // update_sources_from_map();
      updateSources();

      // update time:
      time += time_step;
      ++time_step_number;

      // output:
      if(time_step_number % save_period == 0)
      {
        ++save_step_number;
        std::cout << "time: " << time << std::endl;

        output_sources();
        output_bacteria();
      }       

  }while( time < run_time );

}


template<int dim>
void
StokesSolver<dim>::check_grid(const ArgParser& parameters)
{

  // setup_geometry(parameters);

  // MyGridGenerator::square_pipe_with_spherical_hole(triangulation);

    // MyGridGenerator::cube_with_spherical_hole<dim>( triangulation);
    // {
    //   std::string grid_out_file = output_directory + "/grid_coarse.eps";

    //   std::ofstream out (grid_out_file);
    //   GridOut grid_out;
    //   grid_out.write_eps (triangulation, out);
    //   std::cout << "...Grid written to " << grid_out_file << std::endl;
    // }

    const double left_length = 2.5;
    const double right_length = 2.5;
    const double height = 2.;
    const double radius = 0.5;

    GridGenerationTools::build_mixer_mesh(left_length,
                                          right_length,
                                          height,
                                          radius,
                                          triangulation);

// void build_filter_mesh(const double left_length,
//                       const double center_filter_length,
//                       const double right_length,
//                       const unsigned int number_channels,
//                       const double wall_thickness,
//                       const double channel_thickness,
//                       Triangulation<dim>& tria)

  // const double left_length = 5;
  // const double center_filter_length = 7;
  // const double right_length = 4;
  // const unsigned int number_channels = 4;
  // const double wall_thickness = 0.5;
  // const double channel_thickness = 1.5;


  // GridGenerationTools::build_filter_mesh(left_length,
  //                                       center_filter_length,
  //                                       right_length,
  //                                       number_channels,
  //                                       wall_thickness,
  //                                       channel_thickness,
  //                                       triangulation);

  // const Point<dim> p_1(0,0);
  // const Point<dim> p_2(10,5);

  // std::vector<std::vector<double> > step_sizes = {{8,2},{2,3}};


  // GridGenerator::subdivided_hyper_rectangle(triangulation,
  //                                           step_sizes,
  //                                           p_1,
  //                                           p_2);


// oid GridGenerator::subdivided_hyper_rectangle ( Triangulation< dim > &  tria,
// const std::vector< std::vector< double > > &  step_sizes,
// const Point< dim > &  p_1,
// const Point< dim > &  p_2,
// const bool  colorize = false 
// ) 


    {
      std::string grid_out_file = output_directory + "/grid.eps";

      std::ofstream out (grid_out_file);
      GridOut grid_out;
      grid_out.write_eps (triangulation, out);
      std::cout << "...Grid written to " << grid_out_file << std::endl;
    }

    {
      std::string grid_out_file = output_directory + "/grid.vtk";

      std::ofstream out (grid_out_file);
      GridOut grid_out;
      grid_out.write_vtk (triangulation, out);
      std::cout << "...Grid written to " << grid_out_file << std::endl;
    }

    triangulation.refine_global(3);
    
    {
      std::string grid_out_file = output_directory + "/grid_refined.eps";

      std::ofstream out (grid_out_file);
      GridOut grid_out;
      grid_out.write_eps (triangulation, out);
      std::cout << "...Grid written to " << grid_out_file << std::endl;
    }

}


template<int dim>
void 
StokesSolver<dim>::run_check_stokes(const ArgParser& parameters)
{
  setup_geometry(parameters);

  // get parameters from geometry...
  // const double left_length = 5;
  // const double center_filter_length = 7;
  // const double right_length = 4;
  // const unsigned int number_channels = 4;
  // const double wall_thickness = 0.5;
  // const double channel_thickness = 1.5;


  // GridGenerationTools::build_filter_mesh(left_length,
  //                                       center_filter_length,
  //                                       right_length,
  //                                       number_channels,
  //                                       wall_thickness,
  //                                       channel_thickness,
  //                                       triangulation);

  GridGenerationTools::build_mesh_from_geometry(geometry, triangulation);

  triangulation.refine_global(1);

  output_grid();

  // const unsigned int number_walls = number_channels - 1;


  // const unsigned int number_refinements = 5; // 5;

  // for (unsigned int refinement_cycle = 0; refinement_cycle < number_refinements;
  //      ++refinement_cycle)
  //   {
  //     std::cout << "Refinement cycle " << refinement_cycle << std::endl;
  //     if (refinement_cycle > 0)
  //       refine_mesh ();
  //     setup_dofs (0, number_walls);
  //     std::cout << "   Assembling..." << std::endl << std::flush;
  //     assemble_stokes_system ();
  //     std::cout << "   Solving..." << std::flush;
  //     solve_stokes ();
  //     output_results (refinement_cycle);
  //     std::cout << std::endl;
  //   }



}





}} // close namespace

#endif

