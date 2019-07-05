#ifndef ADVECTION_H
#define ADVECTION_H


#include <deal.II/base/tensor_function.h>
using dealii::TensorFunction;

#include <deal.II/base/point.h>
using dealii::Point;
using dealii::Tensor;

#include <cmath>

#include "../utility/enum_types.h"
#include "../discrete_field/numerical_velocity.h" 

namespace MicrobeSimulator{

  template<int dim>
  class AdvectionField : public TensorFunction<1,dim>
  {
  public:

    AdvectionField();
    AdvectionField(VelocityType vt, const Point<dim>& lower, const Point<dim>& upper,
      double max_vel, double vrad = 0, double vrotation = 0);

    virtual Tensor<1,dim> value (const Point<dim> &p) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<1,dim> >    &values) const;

    DeclException2 (ExcDimensionMismatch,
                    unsigned int, unsigned int,
                    << "The vector has size " << arg1 << " but should have "
                    << arg2 << " elements.");

    VelocityType getVelocityType() const;

    void initialize(VelocityType vt, const Point<dim>& lower, const Point<dim>& upper,
      double max_vel, double vrad = 0, double vrotation = 0);
    void initialize(VelocityType vt, const Point<dim>& lower, const Point<dim>& upper,
      double geometry_scale, double max_vel, 
      std::string x_file, std::string y_file );

    void print(const std::vector<Point<dim> >& points, std::ostream& out);

    void printInfo(std::ostream& out) const;


  private:
    VelocityType velocity_type;

    // boundary:
    Point<dim> bottom_left;
    Point<dim> top_right; 
    // Geometry<dim>* geometry_pointer;

    NumericalVelocity<dim> numerical_velocity; // only initialize if needed

    // for constant, pipe, and couette flow:
    double maximum_velocity;

    // for vortex:
    double vortex_radius;
    double vortex_rotation;

    // functions:
    Point<dim> getConstantVelocity() const;
    Point<dim> getCouetteVelocity(const Point<dim>& p) const;
    Point<dim> getPipeVelocity(const Point<dim>& p) const;
    Point<dim> getVortexVelocity(const Point<dim>& p) const;
    Tensor<1,dim> getNumericalVelocity(const Point<dim>& p) const;
    Tensor<1,dim> getTileVelocity(const Point<dim>& p) const;
  };

// IMPLEMENTATION
// ------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------

  
  template<int dim>
  AdvectionField<dim>::AdvectionField ()
    :
    velocity_type(VelocityType::NO_FLOW),
    maximum_velocity(0),
    vortex_radius(0),
    vortex_rotation(0)
  {}

  template<int dim>
  AdvectionField<dim>::AdvectionField(VelocityType vt, const Point<dim>& lower, const Point<dim>& upper,
      double max_vel, double vradius, double vrotation)
    :
    velocity_type(vt),
    bottom_left(lower),
    top_right(upper),
    maximum_velocity(max_vel),
    vortex_radius(vradius),
    vortex_rotation(vrotation)
  {}


  template<int dim>
  Tensor<1,dim> AdvectionField<dim>::value (const Point<dim> &p) const
  {
    if(velocity_type == VelocityType::CONSTANT_FLOW)
      return getConstantVelocity();
    else if(velocity_type == VelocityType::COUETTE)
      return getCouetteVelocity(p);
    else if(velocity_type == VelocityType::PIPE_FLOW)
      return getPipeVelocity(p);
    else if(velocity_type == VelocityType::VORTEX_FLOW)
      return getVortexVelocity(p);
    else if(velocity_type == VelocityType::NUMERICAL_FLOW)
      return getNumericalVelocity(p);
    else if(velocity_type == VelocityType::TILE_FLOW)
      return getTileVelocity(p);
  
    // else // including (velocity_type == VelocityType::NO_FLOW)
    return Point<dim>();
  }

  template<int dim>
  void AdvectionField<dim>::value_list (const std::vector<Point<dim> > &points,
                           std::vector<Tensor<1,dim> >    &values) const
  {
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

    for (unsigned int i=0; i<points.size(); ++i)
      values[i] = AdvectionField<dim>::value (points[i]);
  }


  template<int dim>
  VelocityType AdvectionField<dim>::getVelocityType() const
  {
    return velocity_type;
  }

  template<int dim>
  void AdvectionField<dim>::initialize(VelocityType vt, 
    const Point<dim>& lower, const Point<dim>& upper,
    double max_vel, double vradius, double vrotation)
  {
    velocity_type = vt;
    bottom_left = lower;
    top_right = upper;
    maximum_velocity = max_vel;
    vortex_radius = vradius;
    vortex_rotation = vrotation;
  }

  template<int dim>
  void AdvectionField<dim>::initialize(VelocityType vt, 
    const Point<dim>& lower, const Point<dim>& upper,
    double geometry_scale, double max_vel, 
    std::string x_file, std::string y_file )
  {
    if(dim != 2)
      throw std::invalid_argument("Numerical velocity initialization not implemented for dim != 2");

    velocity_type = vt;
    bottom_left = lower; // *** CAREFUL, points given by geometry are already scaled
    top_right = upper;
    maximum_velocity = max_vel;

    std::ifstream x_file_in(x_file);
    std::ifstream y_file_in(y_file);
    numerical_velocity.initialize(x_file_in, y_file_in, bottom_left, 
      max_vel, geometry_scale);
  }


  template<int dim>
  void AdvectionField<dim>:: printInfo(std::ostream& out) const
  {
    out << "\n\n-----------------------------------------------------" << std::endl;
    out << "\t\tADVECTION FIELD INFO:";
    out << "\n-----------------------------------------------------" << std::endl;

    out << "\t Dimensions: " << dim << std::endl
      << "\t Type: " << getVelocityTypeString(velocity_type) << std::endl
      << "\t Maximum velocity: " << maximum_velocity << std::endl;
     
    if(velocity_type == VelocityType::NUMERICAL_FLOW ||
      velocity_type == VelocityType::TILE_FLOW)
    {
        numerical_velocity.printInfo(out);
    }
    else if(velocity_type == VelocityType::VORTEX_FLOW)
    {
      out << "\t Vortex radius: " << vortex_radius << std::endl
        << "\t Vortex rotation: " << vortex_rotation << std::endl;     
    }
    out << "\n-----------------------------------------------------\n\n" << std::endl;

  }


  // Velocity Functions:
  //-------------------------------------------------------------------------------
  template<int dim>
  Point<dim> AdvectionField<dim>::getConstantVelocity() const
  {
    Point<dim> value;

    value[0] = maximum_velocity; // along first component only
    for(unsigned int i = 1; i < dim; i++)
      value[i] = 0.0;

    return value;
  }

  template<int dim>
  Point<dim> AdvectionField<dim>::getCouetteVelocity(const Point<dim>& p) const
  {
    // should only be valid for square geometries
    // velocity along x direction, dependent on y position -- no z dependence
    // if(dim != 2 || dim != 3)
    //   throw exception // assuming this is done elsewhere
    Point<dim> value;
    const double y = p[1]; 
    const double height = top_right[1]; 

    value[0] = maximum_velocity*(y/height);

    for(unsigned int dim_itr = 1; dim_itr < dim; dim_itr++)
      value[dim_itr] = 0; 

    return value;
  }

  template<int dim>
  Point<dim> AdvectionField<dim>::getPipeVelocity(const Point<dim>& p) const
  {
    // will depend on cylinder vs square...

    Point<dim> value;

    if(dim == 2)
    {
      const double y = p[1];
      const double hy_sqr = top_right[1]*top_right[1];

      value[0] = maximum_velocity*(1 - y*y/hy_sqr);
      value[1] = 0;
    }
    else if(dim == 3)
    {
      const double y = p[1];
      const double z = p[2];
      const double hy_sqr = top_right[1]*top_right[1];
      const double hz_sqr = top_right[2]*top_right[2];

      value[0] = maximum_velocity*(1 - y*y/hy_sqr)*(1 - z*z/hz_sqr);
      value[1] = 0;
      value[2] = 0;
    }
    else
    {
      throw std::runtime_error("Pipe velocity not implemented for desired dimension");
    }

    return value;
    // for cylinder:
    // v = (1 - r^2/R^2) \hat{x}
  }

  template<int dim>
  Point<dim> AdvectionField<dim>::getVortexVelocity(const Point<dim>& p) const
  {
    // should only be valid for cylinder or sphere geometry, ... maybe only cylinder
    
    // assuming dim at least 2

    Point<dim> value;
    Point<dim> origin;

    for(unsigned int dim_itr = 0; dim_itr < dim; dim_itr++)
      origin[dim_itr] = 0;

    const double distance = p.distance(origin);
    const double theta = atan2(p[1],p[0]);
    double factor;

    if(distance < vortex_radius)
      factor = vortex_rotation*distance / (2*
        dealii::numbers::PI*vortex_radius*vortex_radius);
     // if inside vortex
    else
      factor = vortex_rotation/(2*dealii::numbers::PI*distance);
     // if outside vortex


    value[0] = - cos(theta)*factor;
    value[1] = sin(theta)*factor;

    if(dim == 3)
      value[2] = 0; // extrude in z direction

    return value;
  }
  
  template<int dim>
  Tensor<1,dim> AdvectionField<dim>::getNumericalVelocity(const Point<dim>& p) const
  {
    // std::cout << "in get numerical velocity -- not yet fully implemented" 
    //   << "\n make sure to check if numerical velocity field is valid and initialized"
    //   << std::endl;

    return numerical_velocity.value(p);
  }

  template<int dim>
  Tensor<1,dim> AdvectionField<dim>::getTileVelocity(const Point<dim>& p) const
  {
    std::cout << "in get numerical TILE velocity -- not yet fully implemented" 
      << "\n make sure to check if numerical velocity field is valid (initialized)"
      << std::endl;
    return numerical_velocity.value(p);
  }

  template<int dim>
  void AdvectionField<dim>::print(const std::vector<Point<dim> >& points, 
    std::ostream& out)
  {
    for(unsigned int i = 0; i < points.size(); i++)
      out << points[i] << " " << this->value(points[i]) << std::endl;
  }

}

#endif // advection_field.h
