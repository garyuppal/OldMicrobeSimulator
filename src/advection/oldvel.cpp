#include "velocity_fields.h"

VelocityFields::VelocityFields() : xmin(0.0), ymin(0.0), Nx(0), Ny(0), invDX(0.0), invDY(0.0)  {} // default contructor


VelocityFields::VelocityFields(const VelocityFields& vf)
{
  vx = vf.vx;
  vy = vf.vy;

  xmin = vf.xmin;
  ymin = vf.ymin;
  Nx = vf.Nx;
  Ny = vf.Ny;
  invDX = vf.invDX;
  invDY = vf.invDY;
} // copy constructor


// FUNCTION IMPLEMENTATIONS:
// accessors:
unsigned int VelocityFields::getIndexFromPoint(const Point<2> &p) const
{
  double x, y;
  int ix, iy;
  unsigned int i;

  //re-shift origin
  x = p[0] - xmin;
  y = p[1] - ymin;

  ix = round(x*invDX);
  iy = round(y*invDY);

  // check boundary cases: (could go over due to floating point errors)
  ix = (ix<0)?0:ix;
  ix = (ix>Nx-1)?Nx-1:ix;

  iy = (iy<0)?0:iy;
  iy = (ix>Ny-1)?Ny-1:iy;


  i = ix + Nx*iy;

  return i; // for now
}


double VelocityFields::getVelX(const Point<2> &p) const
{
  unsigned int i = getIndexFromPoint(p);
  return vx[i]; // for now
}


double VelocityFields::getVelY(const Point<2> &p) const
{
  unsigned int i = getIndexFromPoint(p);
  return vy[i];
}


// initialize from Files: 
void VelocityFields::initializeVelocityField(std::ifstream& vxfile, std::ifstream& vyfile, 
  double xm, double ym, double vscale)
{
  xmin = xm;
  ymin = ym;

  unsigned int ic = 0;
  std::string line;

  // FOR X VELOCITY:
  while(std::getline(vxfile,line)){
    if(ic < 4){
      // read first four lines give Nx, Ny, dx, dy: 
      std::istringstream stream(line);
      std::string varName;
      double value; 
      stream >> varName >> value;  

      // cases:
      if(varName.compare("Nx") == 0){Nx = value; ic++;}
      if(varName.compare("Ny") == 0){Ny = value; ic++;}
      if(varName.compare("dx") == 0){invDX = (1.0/value); ic++;}
      if(varName.compare("dy") == 0){invDY = (1.0/value); ic++;}
    } // for first 4 lines
    else{
      break;
    } // else
  } // while reading file

  // reserve enough space in advance:
  vx.reserve(Nx*Ny);

  while(std::getline(vxfile,line,',')){
    std::istringstream stream(line);
    double value;
    stream >> value;

    vx.push_back( vscale*value );
  }

    // ============================================================

  // FOR Y VELOCITY:
  while(std::getline(vyfile,line)){
    if(ic < 4){
      // skip first four lines 
      std::istringstream stream(line);
      std::string varName;
      double value; 
      stream >> varName >> value;  

      // cases:
      if(varName.compare("Nx") == 0){ic++;}
      if(varName.compare("Ny") == 0){ic++;}
      if(varName.compare("dx") == 0){ic++;}
      if(varName.compare("dy") == 0){ic++;}
    } // for first 4 lines
    else{
      break;
    } // else
  } // while reading file

  // reserve enough space in advance:
  vy.reserve(Nx*Ny);

  while(std::getline(vyfile,line,',')){
    std::istringstream stream(line);
    double value;
    stream >> value;

    vy.push_back( vscale*value );
  } // while reading y velocity

  std::cout << "VELOCITY FILES READ IN " << std::endl 
    << "NX: " << Nx << std::endl
    << "NY: " << Ny << std::endl
    << "invDX: " << invDX << std::endl
    << "invDY: " << invDY  << std::endl
    << "xmin: " << xmin <<std::endl
    << "ymin: " << ymin << std::endl
    << "vscale: " << vscale << std::endl;

} // VelocityFields::initializeVelocityField()


// output:
void VelocityFields::outputVelocities(std::ofstream& vxOut, std::ofstream& vyOut)
{
  // output X Velocity:
  std::ostringstream outputX;

  outputX << "\nNX = " << Nx << 
     "\nNY = " << Ny <<
     "\ninvDX = " << invDX <<
     "\ninvDY = " << invDY << "\n"; 

  unsigned int nvx = vx.size();
  for(unsigned int i = 0; i < nvx; i++){
    outputX << vx[i] << "\n";
  }

  vxOut << outputX.str();


  // output Y Velocity:
  std::ostringstream outputY;

  outputY << "\nNX = " << Nx << 
     "\nNY = " << Ny <<
     "\ninvDX = " << invDX <<
     "\ninvDY = " << invDY << "\n"; 

  unsigned int nvy = vy.size();
  for(unsigned int i = 0; i < nvy; i++){
    outputY << vy[i] << "\n";
  }

  vyOut << outputY.str();
} // VelocityFields::outputVelocities()



// velocity_fields.cc
