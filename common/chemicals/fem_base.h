#ifndef FEM_BASE_H
#define FEM_BASE_H


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
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
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>

using namespace dealii;

class FEM_Base
{
public:
	FEM_Base(); 

    void setup_system();
    void create_advection_matrix();
    void createGrid(); // can create using geometry....

    const Triangulation<2>& getTriangulation() const;
    const FE_Q<2>& getFE() const;
    const DoFHandler<2>& getDofHandler() const;
    const ConstraintMatrix& getConstraintMatrix() const; 
    const SparsityPattern& getSparsityPattern() const;
    const SparseMatrix<double>& getMassMatrix() const;
    const SparseMatrix<double>& getLaplaceMatrix() const;
    const SparseMatrix<double>& getAdvectionMatrix() const;
    bool isInitialized() const;

private:
    Triangulation<2>    triangulation;
    FE_Q<2>             fe;
    DoFHandler<2>       dof_handler;

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix; 
    SparseMatrix<double> laplace_matrix;

    SparseMatrix<double> advection_matrix; 

    bool initialized;   
}; // class FEM_Base


#endif // FEM_Base.h
