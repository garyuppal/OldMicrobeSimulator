#ifndef MICROBE_SIMULATOR_EXACT_SOLUTION_H
#define MICROBE_SIMULATOR_EXACT_SOLUTION_H

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>


namespace MicrobeSimulator{ namespace ExactSolutions{
	using namespace dealii;

template<int dim>
class Gaussian : public Function<dim>
{
public:
	Gaussian()
		:
		Function<dim>(),
		center(),
		width(1.)
	{}

	Gaussian(const Point<dim>& c, double w)
		:
		Function<dim>(),
		center(c),
		width(w)
	{}

	virtual double value(const Point<dim>& p,
							const unsigned int component = 0) const;
private:
	Point<dim> center;
	double width;
};


template<int dim>
double
Gaussian<dim>::value(const Point<dim>& p,
							const unsigned int component) const
{
	return std::exp( -(p-center)*(p-center)/width );
}


template<int dim>
class GaussNeumann : public Function<dim>
{
public:
	GaussNeumann() 
		: 
		Function<dim>(),
		diffusion_constant(0),
		decay_constant(0),
		time(0)
	{}

	GaussNeumann(double diff, double decay, double time)
		:
		Function<dim>(),
		diffusion_constant(diff),
		decay_constant(decay),
		time(time)
	{}

	virtual double value(const Point<dim>& p,
							const unsigned int component = 0) const;
	virtual Tensor<1,dim> gradient(const Point<dim>& p, 
							const unsigned int component = 0) const;

	void set_solution_constants(double diff, double decay) 
	{
		diffusion_constant = diff;
		decay_constant = decay;
	}
	void update_solution_time(double t) {time = t;}

private:
	double gaussCenter(const Point<dim>& p) const;
	Tensor<1,dim> gaussGradCenter(const Point<dim>& p) const;

	double diffusion_constant;
	double decay_constant;
	double time;
};


template<int dim>
double
GaussNeumann<dim>::gaussCenter(const Point<dim>& p) const
{
	const double factor = 4.*diffusion_constant*time + 1.0;

	return (1./factor)*std::exp( -(p*p)/factor )*std::exp( -decay_constant*time );	
}

template<int dim>
Tensor<1,dim> 
GaussNeumann<dim>::gaussGradCenter(const Point<dim>& p) const
{
	Tensor<1,dim> return_value = p;
	const double factor = -2./(1. + 4.*diffusion_constant*time);
	const double gaussValue = gaussCenter(p);

	return factor*gaussValue*return_value;
	// -2x/(1+4dt)*value;
}

template<int dim>
double 
GaussNeumann<dim>::value(const Point<dim>& p,
					const unsigned int /* componenet */) const
{
	if(dim != 2)
		throw std::runtime_error("not implemented");

	const Point<dim> left(p[0]+10,p[1]);
	const Point<dim> right(p[0]-10,p[1]);
	const Point<dim> bottom(p[0],p[1]+10);
	const Point<dim> top(p[0],p[1]-10);

	return gaussCenter(p) + gaussCenter(left) + gaussCenter(right)
		+ gaussCenter(top) + gaussCenter(bottom);
}

template<int dim>
Tensor<1,dim>
GaussNeumann<dim>::gradient(const Point<dim>& p, 
					const unsigned int /*component*/) const
{
	if(dim != 2)
		throw std::runtime_error("solution not implemented for dim != 2");
	
	if(dim != 2)
		throw std::runtime_error("not implemented");

	const Point<dim> left(p[0]+10,p[1]);
	const Point<dim> right(p[0]-10,p[1]);
	const Point<dim> bottom(p[0],p[1]+10);
	const Point<dim> top(p[0],p[1]-10);

	return gaussGradCenter(p) + gaussGradCenter(left) + gaussGradCenter(right)
		+ gaussGradCenter(top) + gaussGradCenter(bottom);
}



//-----------------------------------------------------------------------------
double wrapPoint(const double x, const double width)
{
	double value = x;

	// domain centered at 0, with width given
	const double left = -0.5*width;
	const double right = 0.5*width;

	while(value > right)
		value = value - width; //left + (x-right); // = x - width
	while(value < left)
		value = width + value; //right - (left - x); // = width + x

	return value;
}


template<int dim>
class GaussXVelocityPeriodic : public Function<dim>
{
public:
	GaussXVelocityPeriodic() 
		:
		Function<dim>(),
		diffusion_constant(0),
		velocity(0),
		time(0)
	{}

	GaussXVelocityPeriodic(double diff, double vel, double t)
		:
		Function<dim>(),
		diffusion_constant(diff),
		velocity(vel),
		time(t)
	{}

	virtual double value(const Point<dim>& p,
							const unsigned int component = 0) const;
	virtual Tensor<1,dim> gradient(const Point<dim>& p, 
							const unsigned int component = 0) const;

	void set_solution_constants(double diff, double vel)
	{
		diffusion_constant = diff; 
		velocity = vel;
	}
	void update_solution_time(double t)
	{
		time = t;
	}

private:
	double gaussCenter(const Point<dim>& p) const;
	Tensor<1,dim> gaussGradCenter(const Point<dim>& p) const;

	double diffusion_constant;
	double velocity;
	double time;

};


template<int dim>
double 
GaussXVelocityPeriodic<dim>::gaussCenter(const Point<dim>& p) const
{
	Point<dim> wrappedPoint = p;

	double x = wrappedPoint[0];
	x += -velocity*time;

	const double width = 10.0;
	x = wrapPoint(x, width);
	wrappedPoint[0] = x;

	const double factor = 4.*diffusion_constant*time + 1.0; 
	// (time	+ (1./(4.*diffusion_constant)));

	return (1./factor)*std::exp( -(wrappedPoint*wrappedPoint)/factor );	
}


template<int dim>
Tensor<1,dim> 
GaussXVelocityPeriodic<dim>::gaussGradCenter(const Point<dim>& p) const
{
	Tensor<1,dim> tensor_factor = p;
	double x = tensor_factor[0];
	x += -velocity*time;

	const double width = 10.0;
	x = wrapPoint(x,width);
	tensor_factor[0] = x;


	const double factor = -2./(1. + 4.*diffusion_constant*time);
	const double gaussValue = gaussCenter(p); // automatically wraps x

	return factor*gaussValue*tensor_factor;
	// -2x/(1+4dt)*value;
}

template<int dim>
double 
GaussXVelocityPeriodic<dim>::value(const Point<dim>& p,
					const unsigned int /* componenet */) const
{
	if(dim != 2)
		throw std::runtime_error("not implemented");

	const Point<dim> bottom(p[0],p[1]+10);
	const Point<dim> top(p[0],p[1]-10);

	return gaussCenter(p) + gaussCenter(top) + gaussCenter(bottom);
}

template<int dim>
Tensor<1,dim>
GaussXVelocityPeriodic<dim>::gradient(const Point<dim>& p, 
					const unsigned int /*component*/) const
{
	if(dim != 2)
		throw std::runtime_error("solution not implemented for dim != 2");
	
	if(dim != 2)
		throw std::runtime_error("not implemented");

	const Point<dim> bottom(p[0],p[1]+10);
	const Point<dim> top(p[0],p[1]-10);

	return gaussGradCenter(p) + gaussGradCenter(top) + gaussGradCenter(bottom);
}



// with DECAY
//--------------------------------------------------------------------------------------

template<int dim>
class GaussFull : public Function<dim>
{
public:
	GaussFull() 
		:
		Function<dim>(),
		diffusion_constant(0),
		decay_constant(0),
		velocity(0),
		time(0)
	{}

	GaussFull(double diff, double decay, double vel, double t)
		:
		Function<dim>(),
		diffusion_constant(diff),
		decay_constant(decay),
		velocity(vel),
		time(t)
	{}

	virtual double value(const Point<dim>& p,
							const unsigned int component = 0) const;
	virtual Tensor<1,dim> gradient(const Point<dim>& p, 
							const unsigned int component = 0) const;

	void set_solution_constants(double diff, double vel)
	{
		diffusion_constant = diff; 
		velocity = vel;
	}
	void update_solution_time(double t)
	{
		time = t;
	}

private:
	double gaussCenter(const Point<dim>& p) const;
	Tensor<1,dim> gaussGradCenter(const Point<dim>& p) const;

	double diffusion_constant;
	double decay_constant;
	double velocity;
	double time;

};


template<int dim>
double 
GaussFull<dim>::gaussCenter(const Point<dim>& p) const
{
	Point<dim> wrappedPoint = p;

	double x = wrappedPoint[0];
	x += -velocity*time;

	const double width = 10.0;
	x = wrapPoint(x, width);
	wrappedPoint[0] = x;

	const double factor = 4.*diffusion_constant*time + 1.0; 
	// (time	+ (1./(4.*diffusion_constant)));

	return (1./factor)*std::exp( -(wrappedPoint*wrappedPoint)/factor )
		*std::exp( - decay_constant*time );	
}


template<int dim>
Tensor<1,dim> 
GaussFull<dim>::gaussGradCenter(const Point<dim>& p) const
{
	Tensor<1,dim> tensor_factor = p;
	double x = tensor_factor[0];
	x += -velocity*time;

	const double width = 10.0;
	x = wrapPoint(x,width);
	tensor_factor[0] = x;


	const double factor = -2./(1. + 4.*diffusion_constant*time);
	const double gaussValue = gaussCenter(p); // automatically wraps x

	return factor*gaussValue*tensor_factor;
	// -2x/(1+4dt)*value;
}

template<int dim>
double 
GaussFull<dim>::value(const Point<dim>& p,
					const unsigned int /* componenet */) const
{
	if(dim != 2)
		throw std::runtime_error("not implemented");

	const Point<dim> bottom(p[0],p[1]+10);
	const Point<dim> top(p[0],p[1]-10);

	return gaussCenter(p) + gaussCenter(top) + gaussCenter(bottom);
}

template<int dim>
Tensor<1,dim>
GaussFull<dim>::gradient(const Point<dim>& p, 
					const unsigned int /*component*/) const
{
	if(dim != 2)
		throw std::runtime_error("solution not implemented for dim != 2");
	
	if(dim != 2)
		throw std::runtime_error("not implemented");

	const Point<dim> bottom(p[0],p[1]+10);
	const Point<dim> top(p[0],p[1]-10);

	return gaussGradCenter(p) + gaussGradCenter(top) + gaussGradCenter(bottom);
}


}}


#endif