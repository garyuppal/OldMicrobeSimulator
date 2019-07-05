#ifndef SINGLE_BACTERIUM_H
#define SINGLE_BACTERIUM_H


#include <deal.II/base/point.h>
using dealii::Point;

#include <deal.II/dofs/dof_handler.h>
using dealii::DoFHandler;

#include <deal.II/lac/vector.h>
using dealii::Vector;

#include <deal.II/lac/block_vector.h>
using dealii::BlockVector;

#include <deal.II/numerics/vector_tools.h>


#include "../geometry/geometry.h"
#include "../advection/advection.h"
#include "fitness.h"

#include <iostream>
#include <string>
#include <sstream>
#include <array>

namespace MicrobeSimulator{

  template<int dim, int NumberChemicals>
  class SingleBacterium{
  private:
    Point<dim> location;

    std::array<double, NumberChemicals> secretionRates;

  public:
    // constructors:
    SingleBacterium();
    SingleBacterium(const Point<dim>& p, 
        const std::array<double, NumberChemicals>& rates);
    SingleBacterium(const SingleBacterium& b);

    // assignment
    SingleBacterium<dim, NumberChemicals>& operator=(const SingleBacterium<dim, NumberChemicals>& rhs);

    // accessors:
    Point<dim> getLocation() const; 
    std::array<double, NumberChemicals> getSecretionRates() const;
    double getSecretionRate(unsigned int i) const;

    // mutators:
    void setLocation(const Point<dim>& p);
    void setSecretionRates(const std::array<double, NumberChemicals>& rates);
    void setSecretionRate(unsigned int i, double rate); 
 
    // functions:

    void randomStep(double timeStep, double diffusionConstant, 
    	const Geometry<dim>* const geometry, const AdvectionField<dim>* const advection = NULL );

    void randomStep(double timeStep, double diffusionConstant, 
        const Geometry<dim>* const geometry, const DoFHandler<dim>& stokes_dof,
        const BlockVector<double> stokes_solution);
    // *** probably better to just pass a function pointer for advection and geometry boundary conditions ***

    double getFitness(const FitnessBase<dim,NumberChemicals>& fitness_function);

    void mutate(double deltaSecretion, double original_secretion_rate, bool binary_mutation); 
        // what about for multiple public goods ?

    void printBacterium(std::ostream& out) const; 

  }; // class Bacterium


// IMPLEMENTATION:
// ---------------------------------------------------------------------------------------------------
    template<int dim, int NumberChemicals>
    SingleBacterium<dim, NumberChemicals>::SingleBacterium()
        : 
        location()
    {}


    template<int dim, int NumberChemicals>
    SingleBacterium<dim, NumberChemicals>::SingleBacterium(
        const Point<dim>& p,  
        const std::array<double, NumberChemicals>& rates)
        : 
        location(p),
        secretionRates(rates)
    {}


    template<int dim, int NumberChemicals>
    SingleBacterium<dim, NumberChemicals>::SingleBacterium(
        const SingleBacterium& b)
    {
        location = b.location;
        secretionRates = b.secretionRates;
    }


    template<int dim, int NumberChemicals>
    SingleBacterium<dim, NumberChemicals>&
    SingleBacterium<dim, NumberChemicals>::operator=(
        const SingleBacterium<dim, NumberChemicals>& rhs)
    {
       // check for self copy:
        if(this == &rhs)
          return *this;

        // copy:
        location = rhs.location;
        secretionRates = rhs.secretionRates;

        return *this;
    }


    // accessors:
    template<int dim, int NumberChemicals>
    Point<dim> 
    SingleBacterium<dim, NumberChemicals>::getLocation() const
    {
        return location;
    }


    template<int dim, int NumberChemicals>
    std::array<double, NumberChemicals> 
    SingleBacterium<dim, NumberChemicals>::getSecretionRates() const
    {
        return secretionRates; 
    }

    
    template<int dim, int NumberChemicals>
    double 
    SingleBacterium<dim, NumberChemicals>::getSecretionRate(unsigned int i) const 
    {
        return secretionRates.at(i); 
    }

    // mutators:
    template<int dim, int NumberChemicals>
    void SingleBacterium<dim, NumberChemicals>::setLocation(const Point<dim>& p)
    {
        location = p;
    }


    template<int dim, int NumberChemicals>
    void 
    SingleBacterium<dim, NumberChemicals>::setSecretionRates(
        const std::array<double, NumberChemicals>& rates)
    {
        secretionRates = rates;
    }

    template<int dim, int NumberChemicals>
    void 
    SingleBacterium<dim, NumberChemicals>::setSecretionRate(unsigned int i,
        double rate)
    {
        secretionRates[i] = rate;
    }


    // functions:


    template<int dim, int NumberChemicals>
    void 
    SingleBacterium<dim, NumberChemicals>::randomStep(double timeStep, double diffusionConstant, 
        const Geometry<dim>* const geometry, const AdvectionField<dim>* const advection)
    {
        Point<dim> old_location = location;
        Point<dim> randomPoint;

        if(dim == 2)
        {
            const double theta = 2*dealii::numbers::PI*((double) rand() / (RAND_MAX));
            randomPoint(0) = std::cos(theta);
            randomPoint(1) = std::sin(theta);
        }
        else if(dim == 3)
        {
            const double theta = dealii::numbers::PI*((double) rand() / (RAND_MAX));
            const double phi = 2*dealii::numbers::PI*((double) rand() / (RAND_MAX));

            randomPoint(0) = std::cos(theta);
            randomPoint(1) = std::sin(theta)*std::cos(phi);
            randomPoint(2) = std::sin(theta)*std::sin(phi);
        }
        else
        {
            throw std::runtime_error("Random step not implemented for dim != 2,3");
        }

        location += std::sqrt(2*dim*timeStep*diffusionConstant)*randomPoint;

        if(advection != NULL)
            location += advection->value(old_location)*timeStep;

            // std::cout << " need to implement advection" << std::endl;
            // location += velocity(location)*dt

        geometry->checkBoundaries(old_location, location);
    }


    template<int dim, int NumberChemicals>
    void 
    SingleBacterium<dim, NumberChemicals>::randomStep(double timeStep, double diffusionConstant, 
        const Geometry<dim>* const geometry, const DoFHandler<dim>& stokes_dof,
        const BlockVector<double> stokes_solution)
    {
        Point<dim> old_location = location;
        Point<dim> randomPoint;

        Point<dim> advection_value; 

        if(dim == 2)
        {
            const double theta = 2*dealii::numbers::PI*((double) rand() / (RAND_MAX));
            randomPoint(0) = std::cos(theta);
            randomPoint(1) = std::sin(theta);
        }
        else if(dim == 3)
        {
            const double theta = dealii::numbers::PI*((double) rand() / (RAND_MAX));
            const double phi = 2*dealii::numbers::PI*((double) rand() / (RAND_MAX));

            randomPoint(0) = std::cos(theta);
            randomPoint(1) = std::sin(theta)*std::cos(phi);
            randomPoint(2) = std::sin(theta)*std::sin(phi);
        }
        else
        {
            throw std::runtime_error("Random step not implemented for dim != 2,3");
        }

        location += std::sqrt(2*dim*timeStep*diffusionConstant)*randomPoint;

        // compute velocity:
        Vector<double> velocity;
        velocity.reinit(dim + 1); // +1 for pressure

        dealii::VectorTools::point_value(stokes_dof,
                                stokes_solution,
                                old_location,
                                velocity);

        for(unsigned int dim_itr = 0; dim_itr<dim; dim_itr++)
            advection_value[dim_itr] = velocity[dim_itr];

        location += advection_value*timeStep;

        geometry->checkBoundaries(old_location, location);
    }


    template<int dim, int NumberChemicals>
    void 
    SingleBacterium<dim, NumberChemicals>::mutate(double deltaSecretion, 
        double original_secretion_rate, bool binary_mutation)
    {
        // for public goods -- up to NumberChemicals - 1
        // pick one to mutate

        int chem_id = rand() % (NumberChemicals-1); 

        if(binary_mutation == true)
        {
            secretionRates[chem_id] = (secretionRates[chem_id] == 0 ?
                                        original_secretion_rate :
                                        0);
        }
        else
        {
            const double rate = secretionRates[chem_id] 
                + deltaSecretion*( 2.*((double) rand() / (RAND_MAX)) - 1.);

            secretionRates[chem_id] = ( rate < 0 ?
                                        0 :
                                        rate );  // absorb at 0
        }
    }

    template<int dim, int NumberChemicals>
    double 
    SingleBacterium<dim, NumberChemicals>::getFitness(
        const FitnessBase<dim,NumberChemicals>& fitness_function)
    {
        return fitness_function.value(location, secretionRates);
    }


    template<int dim, int NumberChemicals>
    void 
    SingleBacterium<dim, NumberChemicals>::printBacterium(std::ostream& out) const
    {
        out << location;
        for(unsigned int i = 0; i < NumberChemicals; i++)
            out << " " << secretionRates[i];
        out << std::endl;
    }

}

#endif  // bacterium.h
