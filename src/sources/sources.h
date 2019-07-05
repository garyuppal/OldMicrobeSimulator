#ifndef MICROBE_SIMULATOR_SOURCES_HANDLER_H
#define MICROBE_SIMULATOR_SOURCES_HANDLER_H

#include "../bacteria/bacteria.h"
// #include "source_functions.h"


namespace MicrobeSimulator{

	template<int dim, int NumberGoods>
	class Sources{
	public:
		Sources(Bacteria<dim,NumberGoods>* bptr);
		~Sources();

		Function<dim>** getSources() const;

		void printInfo();

	private:
		Function<dim>** sources;
		Bacteria<dim,NumberGoods>* bacteria_pointer;
	};

	template<int dim, int NumberGoods>
	Sources<dim,NumberGoods>::Sources(Bacteria<dim,NumberGoods>* bptr)
		:
		bacteria_pointer(bptr)
	{}

	template<int dim, int NumberGoods>
	Sources<dim,NumberGoods>::~Sources()
	{

	}

	template<int dim, int NumberGoods>
	void Sources<dim,NumberGoods>::printInfo()
	{
		std::cout << "\n\nSOURCES! in " << dim << " dimensions, for "
			<< NumberGoods << " goods." << std::endl;
	}
}


#endif


