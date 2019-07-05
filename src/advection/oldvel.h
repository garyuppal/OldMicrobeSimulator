#ifndef VELOCITYFILEDS_H
#define VELOCITYFILEDS_H


// #include <deal.II/base/utilities.h>
// #include <deal.II/base/quadrature_lib.h>
// #include <deal.II/base/function.h>
// #include <deal.II/base/logstream.h>
// #include <deal.II/lac/vector.h>
// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/sparse_matrix.h>
// // can probably remove cg...
// #include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/solver_bicgstab.h>

// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/constraint_matrix.h>
// #include <deal.II/grid/tria.h>
// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/grid_refinement.h>
// #include <deal.II/grid/grid_out.h>
// #include <deal.II/grid/tria_accessor.h>
// #include <deal.II/grid/tria_iterator.h>
// #include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_accessor.h>
// #include <deal.II/dofs/dof_tools.h>
// #include <deal.II/fe/fe_q.h>
// #include <deal.II/fe/fe_values.h>
// #include <deal.II/numerics/data_out.h>
// #include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/error_estimator.h>
// #include <deal.II/numerics/solution_transfer.h>
// #include <deal.II/numerics/matrix_tools.h>

// #include <deal.II/base/data_out_base.h>
// #include <deal.II/base/parameter_handler.h>

// #include <deal.II/base/tensor_function.h>
// #include <deal.II/grid/grid_in.h>

// #include <list>
// #include <fstream>
// #include <iostream>
// #include <cmath>
// #include <ctime>
// #include <string>
// #include <sstream>

#include "../geometry/geometry.h"

class VelocityFields
{
  private:
    std::vector<double> vx;
    std::vector<double> vy; // field classes??

    double xmin;
    double ymin; // bottom corner for reference

    // velocity field resolution:
    unsigned int Nx;
    unsigned int Ny;
    double invDX; 
    double invDY; 
  
  public:
    VelocityFields(); // default contructor

    VelocityFields(const VelocityFields& vf); // copy constructor

    // Functions:
    double getVelX(const Point<2> &p) const;
    double getVelY(const Point<2> &p) const;
    unsigned int getIndexFromPoint(const Point<2> &p) const; 

    void initializeVelocityField(std::ifstream& vxfile, std::ifstream& vyfile, 
        double xm, double ym, double vscale);
    void outputVelocities(std::ofstream& vxOut, std::ofstream& vyOut);

}; // class VelocityFields{}



#endif  // velocity_fields.h
