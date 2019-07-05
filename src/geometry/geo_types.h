#ifndef MICROBESIMULATOR_GEOTYPES_H
#define MICROBESIMULATOR_GEOTYPES_H

#include <iostream>

namespace MicrobeSimulator{ namespace GeoTypes{

struct Filter{
	unsigned int number_channels;
	double channel_thickness;
	double wall_thickness;
	double left_length;
	double center_length;
	double right_length;

	void printInfo(std::ostream& out)
	{
		out << "FILTER:" << std::endl
			<< "Number channels: " << number_channels << std::endl
			<< "Channel thickness: " << channel_thickness << std::endl
			<< "Wall thickness: " << wall_thickness << std::endl
			<< "Left length: " << left_length << std::endl
			<< "Center length: " << center_length << std::endl
			<< "Right length: " << right_length << std::endl;
 	}
};


struct Mixer{
	double left_length;
	double right_length;
	double height;
	double radius;

	void printInfo(std::ostream& out)
	{
		out << "MIXER:" << std::endl
			<< "Left length: " << left_length << std::endl
			<< "Right length: " << right_length << std::endl
			<< "Height: " << height << std::endl
			<< "Radius: " << radius << std::endl;
 	}
};


}} // close namespace
#endif