#ifndef MICROBE_SIMULATOR_SOURCE_FUNCTIONS_H
#define MICROBE_SIMULATOR_SOURCE_FUNCTIONS_H


#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
using dealii::Function;

#include "../bacteria/bacteria.h"

namespace MicrobeSimulator{
  
  template<int dim, int NumberGoods>
  class SecretionFunction : public Function<dim>
  {
  public:
    SecretionFunction(Bacteria<dim,NumberGoods>* bact_ptr, unsigned int chem_id);

    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
    
  private:
    const double width;
    const double height;
    Bacteria<dim, NumberGoods>* bacteria_pointer; 
    unsigned int chemical_id;
  }; // can similarly define Secretion Function class


  template<int dim, int NumberGoods>
  SecretionFunction<dim,NumberGoods>::SecretionFunction(Bacteria<dim,NumberGoods>* bact_ptr, unsigned int chem_id)
    :
    Function<dim>(),
    width ( 25 * 3.14159265 ), 
    height (0.04),
    bacteria_pointer(bact_ptr),
    chemical_id(chem_id)
    {}

  template<int dim, int NumberGoods>
  double SecretionFunction<dim,NumberGoods>::value (const Point<dim> &p,
                                    const unsigned int component) const
  {
    // (void) component;
    // Assert (component == 0, ExcIndexRange(component, 0, 1)); 

    double return_value = 0.0;
    unsigned int size = bacteria_pointer->getSize();

    if(chemical_id == 0)
    {
      const double secretionRate = bacteria_pointer->getWasteSecretionRate();

      for(unsigned int i = 0; i < size; i++)
      {
        
        double distance_squared = 0.0;
        for(unsigned int i = 0; i < dim; i++)
          distance_squared += std::pow( p[i] - bacteria_pointer->at(i).getLocation()[i] , 2.0);

        return_value += secretionRate*height*( std::exp( -width*distance_squared ));

      }
    }
    else
    {
      for(unsigned int i = 0; i < size; i++)
      {
        double secretionRate = bacteria_pointer->at(i).getGoodSecretionRate(chemical_id);

        double distance_squared = 0.0;
        for(unsigned int i = 0; i < dim; i++)
          distance_squared += std::pow( p[i] - bacteria_pointer->at(i).getLocation()[i] , 2.0);

        return_value += secretionRate*height*( std::exp( -width*distance_squared ));
      }

    } // mutable secretion rate...

    return return_value;
  }



} 

#endif // source_functions.h
