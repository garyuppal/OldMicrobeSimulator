#ifndef ADVECTIONFIELD_H
#define ADVECTIONFIELD_H


// #include <deal.II/base/utilities.h>
// #include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
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

#include <deal.II/base/tensor_function.h>
// #include <deal.II/grid/grid_in.h>

// #include <list>
// #include <fstream>
// #include <iostream>
// #include <cmath>
// #include <ctime>
// #include <string>
// #include <sstream>

#include "../geometry/geometry.h"
#include "velocity_fields.h"
#include "../testing/querrypoints.h"

using namespace dealii;


class AdvectionField : public TensorFunction<1,2>
{
public:
  enum VelType : unsigned int
  {
    None = 0, Numerical, ConstantVel, Couette, Poiseuille, Vortex
  }; // velocity type

  AdvectionField ();
  AdvectionField(double rot, double rad, Geometry* geoptr, VelType vt);



  virtual Tensor<1,2> value (const Point<2> &p) const;

  virtual void value_list (const std::vector<Point<2> > &points,
                           std::vector<Tensor<1,2> >    &values) const;

  DeclException2 (ExcDimensionMismatch,
                  unsigned int, unsigned int,
                  << "The vector has size " << arg1 << " but should have "
                  << arg2 << " elements.");


  void setVelocity(double vm, Geometry* geoptr, VelType vt);
  void setVelocity(double rot, double rad, Geometry* geoptr, VelType vt);
  void setVFields(std::ifstream& vxfile, std::ifstream& vyfile, // should overload above... keep same name...
    double xm, double ym, double vscale);

  void outputVelocities(std::ofstream& vxOut, std::ofstream& vyOut);

  void interpolateToFile(std::ostream& outFile, const Geometry &geo);
  void interpolateToFile(std::ostream& outFile_x, std::ostream& outFile_y, const Geometry &geo);

  void printQuerryVelocity(QuerryPoints qp, std::ostream& velOut);

  VelType getVelocityType() const;

private:
  VelocityFields vFields;
  double vmax;
  Geometry* gptr;
  VelType vtype;

  // for vortex:
  double vortex_rotation;
  double vortex_radius;    

  double getVelX(const Point<2> &p) const;
  double getVelY(const Point<2> &p) const;
};



#endif // advection_field.h
