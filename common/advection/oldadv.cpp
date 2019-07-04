#include "advection_field.h"


AdvectionField::AdvectionField () 
  : 
  TensorFunction<1,2> (),   
  vmax(0.0),
  gptr(0),
  vtype(None),
  vortex_rotation(0.0),
  vortex_radius(0.0)
  {}


AdvectionField::AdvectionField(double rot, double rad, Geometry* geoptr, VelType vt)
  :
  TensorFunction<1,2> (),   
  vmax(0.0),
  gptr(geoptr),
  vtype(vt),
  vortex_rotation(rot),
  vortex_radius(rad)
  {}

void AdvectionField::setVelocity(double rot, double rad, Geometry* geoptr, VelType vt)
{
  vortex_rotation = rot;
  vortex_radius = rad;
  gptr = geoptr;
  vtype = vt;
}


void AdvectionField::setVFields(std::ifstream& vxfile, std::ifstream& vyfile, 
   double xm, double ym, double vscale)
{ 
  vFields.initializeVelocityField(vxfile,vyfile,xm,ym,vscale); 
  vtype = Numerical;
} // AdvectionField::setVFields()


void AdvectionField::outputVelocities(std::ofstream& vxOut, std::ofstream& vyOut)
{
  vFields.outputVelocities(vxOut,vyOut);
}


Tensor<1,2> AdvectionField::value (const Point<2> &p) const
{
  Point<2> value;
  value[0] = getVelX(p); // x component
  value[1] = getVelY(p); // y component
  return value;
} // change to pouisuelle flow...


void AdvectionField::setVelocity(double vm, Geometry* geoptr, VelType vt)
{
  vmax = vm;
  gptr = geoptr;
  vtype = vt;
}


AdvectionField::VelType AdvectionField::getVelocityType() const
{
  return vtype;
}



double AdvectionField::getVelX(const Point<2> &p) const
{
  if(vtype == Numerical)
  {
    return vFields.getVelX(p);
  }
  else if(vtype == ConstantVel)
  {
    return vmax;
  }
  else if(vtype == Couette)
  {
    double width = 0.5*(gptr->getYMax() - gptr->getYMin());
    return vmax*(p[1]/width);
  }
  else if(vtype == Poiseuille)
  {
    return 0.0; // for now
  }
  else if(vtype == Vortex)
  {
    const double dist = sqrt( p[0]*p[0] + p[1]*p[1]);
    const double theta = atan2(p[1],p[0]); 
    double vxtmp;
    if(dist <= vortex_radius)
    {
      vxtmp =  vortex_rotation*dist/( 2*numbers::PI*dist );
    }
    else
    {
      vxtmp =  vortex_rotation/( 2*numbers::PI*dist );
    }

    return vxtmp*cos(theta);
  }
  else
  {
    return 0.0;
  }
} // get X velocity

double AdvectionField::getVelY(const Point<2> &p) const
{
  if(vtype == Numerical)
  {
    return vFields.getVelY(p);
  }
  else if(vtype == Vortex)
  {
    const double dist = sqrt( p[0]*p[0] + p[1]*p[1]);
    const double theta = atan2(p[1],p[0]); 
    double vytmp;
    if(dist <= vortex_radius)
    {
      vytmp =  vortex_rotation*dist/( 2*numbers::PI*dist );
    }
    else
    {
      vytmp =  vortex_rotation/( 2*numbers::PI*dist );
    }

    return -vytmp*sin(theta);
  }
  else
  {
    return 0.0; // same for each case
  }
}// get vel Y


void AdvectionField::value_list (const std::vector<Point<2> > &points,
                                   std::vector<Tensor<1,2> >    &values) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));

  for (unsigned int i=0; i<points.size(); ++i)
    values[i] = AdvectionField::value (points[i]);
}


// out put to debug:
void AdvectionField::interpolateToFile(std::ostream& outFile, const Geometry &geo)
{
  unsigned int ny = 0;

    outFile << "x = [" << geo.getXMin() << ", " << geo.getXMax() << "]; y = [" << geo.getYMin() << ", " << geo.getYMax() << "]; " << std::endl;
    outFile << "dx = " << geo.getDX() << " dy = " << geo.getDY() << std::endl;
    for(double y = geo.getYMin(); y < geo.getYMax(); y += geo.getDY()){
      for(double x = geo.getXMin(); x < geo.getXMax(); x += geo.getDX()){
        bool check = geo.isInDomain(x,y);
        if( check ){
          Point<2> p(x,y);
          outFile << value(p) << ", ";
        }else{
          outFile << "NaN" << ", ";
        }
      } // for x
      outFile << std::endl;
      ny++;
    } // for y

    outFile << "Ny = " << ny << std::endl;
    outFile << std::endl;
} // interpolateToFile()


void AdvectionField::interpolateToFile(std::ostream& outFile_x, std::ostream& outFile_y, const Geometry &geo)
{
  unsigned int ny = 0;

    outFile_x << "x = [" << geo.getXMin() << ", " << geo.getXMax() << "]; y = [" << geo.getYMin() << ", " << geo.getYMax() << "]; " << std::endl;
    outFile_x << "dx = " << geo.getDX() << " dy = " << geo.getDY() << std::endl;
    for(double y = geo.getYMin(); y < geo.getYMax(); y += geo.getDY()){
      for(double x = geo.getXMin(); x < geo.getXMax(); x += geo.getDX()){
        if( geo.isInDomain(x,y) ){
          Point<2> p(x,y);
          outFile_x << value(p)[0] << ", ";
        }else{
          outFile_x << "NaN" << ", ";
        }
      } // for x
      outFile_x << std::endl;
      ny++;
    } // for y

    outFile_x << "Ny = " << ny << std::endl;
    outFile_x << std::endl;


    // output y:
   ny = 0;

    outFile_y << "x = [" << geo.getXMin() << ", " << geo.getXMax() << "]; y = [" << geo.getYMin() << ", " << geo.getYMax() << "]; " << std::endl;
    outFile_y << "dx = " << geo.getDX() << " dy = " << geo.getDY() << std::endl;
    for(double y = geo.getYMin(); y < geo.getYMax(); y += geo.getDY()){
      for(double x = geo.getXMin(); x < geo.getXMax(); x += geo.getDX()){
        if( geo.isInDomain(x,y) ){
          Point<2> p(x,y);
          outFile_y << value(p)[1] << ", ";
        }else{
          outFile_y << "NaN" << ", ";
        }
      } // for x
      outFile_y << std::endl;
      ny++;
    } // for y

    outFile_y << "Ny = " << ny << std::endl;
    outFile_y << std::endl;
} // interpolateToFile()


void AdvectionField::printQuerryVelocity(QuerryPoints qp, std::ostream& velOut)
{
  unsigned int n = qp.getLength();

  for(unsigned int i = 0; i < n; i++)
  {
    Point<2> p(qp.getX(i),qp.getY(i));
    velOut << p[0] << " , " <<  p[1] << " , " << value(p)[0] << " , " << value(p)[1] << std::endl;
  } // for each point

}



// advection_field.cc
