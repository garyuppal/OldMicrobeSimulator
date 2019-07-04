#include "fem_base.h"

FEM_Base::FEM_Base() :
    fe(1),
    dof_handler(triangulation),
    initialized(false)
    {}


void FEM_Base::setup_system() 
{
  // set up matrices:
  dof_handler.distribute_dofs(fe);

  std::cout << std::endl
          << "==========================================="
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
  //   std::cout << "\n ...USING PERIODIC X BOUNDARY" << std::endl;
  //   // ADD PERIODICITY:
  //     std::vector<GridTools::PeriodicFacePair<typename DoFHandler<2>::cell_iterator> >
  //     periodicity_vector;

  //     const unsigned int direction = 0;
  //     GridTools::collect_periodic_faces(dof_handler, 1, 2, direction,
  //                                       periodicity_vector); //, offset, rotation_matrix);

  //     DoFTools::make_periodicity_constraints<DoFHandler<2> >
  //     (periodicity_vector, constraints); //, fe.component_mask(velocities), first_vector_components);
  // } // if periodic in x direction

  // if(systemParameters.getYBoundary() == Geometry::wrap)
  // {
  //   std::cout << "\n ...USING PERIODIC Y BOUNDARY" << std::endl;
  //   // ADD PERIODICITY:
  //     std::vector<GridTools::PeriodicFacePair<typename DoFHandler<2>::cell_iterator> >
  //     periodicity_vector;

  //     const unsigned int direction = 1; // understand this *** x vs y component
  //     GridTools::collect_periodic_faces(dof_handler, 3, 4, direction,
  //                                       periodicity_vector); //, offset, rotation_matrix);

  //     DoFTools::make_periodicity_constraints<DoFHandler<2> >
  //     (periodicity_vector, constraints); //, fe.component_mask(velocities), first_vector_components);
  // } // if periodic in y direction

  constraints.close();
  // --------------------------------

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  advection_matrix.reinit(sparsity_pattern);

  // system_matrix.reinit(sparsity_pattern);
  // system_matrix2.reinit(sparsity_pattern);
  // system_matrix_waste.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<2>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<2>(fe.degree+1),
                                       laplace_matrix);

  // If provided velocity function
  // if(advField.getVelocityType() != AdvectionField::None)
  // {
  //   std::cout << "\n ...CREATING ADVECTION MATRIX" << std::endl;
  //   create_advection_matrix();
  // }

  // solution.reinit(dof_handler.n_dofs());
  // old_solution.reinit(dof_handler.n_dofs());
  // system_rhs.reinit(dof_handler.n_dofs());
  // tmp.reinit(dof_handler.n_dofs());

  // solution2.reinit(dof_handler.n_dofs());
  // old_solution2.reinit(dof_handler.n_dofs());
  // system_rhs2.reinit(dof_handler.n_dofs());
  // tmp2.reinit(dof_handler.n_dofs());

  // solution_waste.reinit(dof_handler.n_dofs());
  // old_solution_waste.reinit(dof_handler.n_dofs());
  // system_rhs_waste.reinit(dof_handler.n_dofs());
  // tmp_waste.reinit(dof_handler.n_dofs());

  initialized = true;
} // setup_system()


void FEM_Base::create_advection_matrix() {}


void FEM_Base::createGrid() 
{
  	const unsigned int initial_global_refinement = 5; //systemParameters.getGRef(); 

    const Point<2> p1(0.0,0.0); //(systemParameters.getXMin(),systemParameters.getYMin()); //xmin,ymin);
    const Point<2> p2(5.0,5.0); //(systemParameters.getXMax(),systemParameters.getYMax()); //xmax,ymax);
    GridGenerator::hyper_rectangle (triangulation,p1,p2, /*colorize = */ true);  
    std::cout << "using rectangle mesh with points: " << p1 << " ; " << p2 << std::endl;

    triangulation.refine_global (initial_global_refinement); 
}


const Triangulation<2>& FEM_Base::getTriangulation() const { return triangulation; }
const FE_Q<2>& FEM_Base::getFE() const { return fe; }
const DoFHandler<2>& FEM_Base::getDofHandler() const { return dof_handler; }
const ConstraintMatrix& FEM_Base::getConstraintMatrix() const { return constraints; }
const SparsityPattern& FEM_Base::getSparsityPattern() const { return sparsity_pattern; }
const SparseMatrix<double>& FEM_Base::getMassMatrix() const { return mass_matrix; }
const SparseMatrix<double>& FEM_Base::getLaplaceMatrix() const { return laplace_matrix; }
const SparseMatrix<double>& FEM_Base::getAdvectionMatrix() const { return advection_matrix; }
bool FEM_Base::isInitialized() const { return initialized; }


// fem_base.cc

