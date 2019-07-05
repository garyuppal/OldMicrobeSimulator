#ifndef MICROBESIMULATOR_ENUM_TYPES_H
#define MICROBESIMULATOR_ENUM_TYPES_H

#include <string>

namespace MicrobeSimulator{

	enum class RunMode : int
	{
		FEM_CG = 0, FEM_DG = 1, FDM = 2, DEBUGGING = 3
	};

	enum class BoundaryCondition : int
	{
		WRAP = 0, REFLECT = 1, OPEN = 2 // open end will be upper end for now
	}; // only makes sense for hyper cube

  	enum class VelocityType : int
	{
		NO_FLOW = 0, NUMERICAL_FLOW = 1, TILE_FLOW = 2,
		CONSTANT_FLOW = 3, COUETTE = 4, PIPE_FLOW = 5, VORTEX_FLOW = 6
	}; // velocity type

	enum class MeshType : int
	{
		FILE_MESH = 0, BOX_MESH = 1, SQUARE_CHEESE = 2, HEX_CHEESE = 3,
		MIXER = 4, FILTER = 5
	};

	std::string getRunModeString(RunMode run_mode)
	{
		std::string result = "";
		if(run_mode == RunMode::FEM_CG)
			result = "Finite element (CG)";
		else if(run_mode == RunMode::FEM_DG)
			result = "Discontinuous Galerkin finite element (DG)";
		else if(run_mode == RunMode::FDM)
			result = "Finite difference - Forward Euler";
		else 
			result = "ERROR";

		return result;
	}

	std::string getVelocityTypeString(VelocityType vtype)
	{
		std::string result = "";
		if(vtype == VelocityType::NO_FLOW)
			result = "No flow";
		else if(vtype == VelocityType::NUMERICAL_FLOW)
			result = "Numerical flow field";
		else if(vtype == VelocityType::TILE_FLOW)
			result = "Tile flow";
		else if(vtype == VelocityType::CONSTANT_FLOW)
			result = "Constant flow";
		else if(vtype == VelocityType::COUETTE)
			result = "Planar Couette flow";
		else if(vtype == VelocityType::PIPE_FLOW)
			result = "Hagen-Poiseuille pipe flow";
		else if(vtype == VelocityType::VORTEX_FLOW)
			result = "Rankine vortex flow";
		else
			result = "ERROR";

		return result;
	}

	std::string getBoundaryConditionString(BoundaryCondition bc)
	{
		std::string result = "";
		if(bc == BoundaryCondition::WRAP)
			result = "Periodic";
		else if(bc == BoundaryCondition::REFLECT)
			result = "Neumann";
		else if(bc == BoundaryCondition::OPEN)
			result = "Open";
		else
			result = "ERROR";

		return result;
	}

	// enum class MeshType : int
	// {
	// 	FILE_MESH = 0, BOX_MESH = 1, SQUARE_CHEESE = 2, HEX_CHEESE = 3,
	// 	MIXER = 4
	// };
	std::string getMeshTypeString(MeshType mesh_type)
	{
		std::string result = "";
		if(mesh_type == MeshType::FILE_MESH)
			result = "Mesh from file";
		else if(mesh_type == MeshType::BOX_MESH)
			result = "Hyper rectangle";
		else if(mesh_type == MeshType::SQUARE_CHEESE)
			result = "Square spacing swiss cheese";
		else if(mesh_type == MeshType::HEX_CHEESE)
			result = "Hexagonal spacing swiss cheese";
		else if(mesh_type == MeshType::MIXER)
			result = "Mixer tube";
		else if(mesh_type == MeshType::FILTER)
			result = "Filter";
		else
			result = "ERROR";

		return result;
	}
}
#endif

