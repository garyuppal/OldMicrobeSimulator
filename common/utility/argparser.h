#ifndef MICROBE_SIMULATOR_ARGPARSER_H
#define MICROBE_SIMULATOR_ARGPARSER_H

#include <ctime>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <cstddef>

// to explore directory:
#include <dirent.h>
#include <sys/stat.h> 
#include "./enum_types.h"


namespace MicrobeSimulator{

	// DATE-TIME FUNCTION:
	const std::string currentDateTime() 
	{
	    time_t     now = time(0);
	    struct tm  tstruct;
	    char       buf[80];
	    tstruct = *localtime(&now);
	    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	    // for more information about date/time format
	    strftime(buf, sizeof(buf), "%a-%m-%d-%y_%H-%M-%S", &tstruct);

	    return buf;
	} // for outputting data --- used in ArgParser


	class ArgParser{
	public:
		ArgParser(int argc, char** argv);

		// ACCESSORS:
		// geometry:
		std::string getGeometryFile() const;

		// advection:
		VelocityType getVelocityType() const;
		std::string getVelocityFile_X() const;
		std::string getVelocityFile_Y() const;
		double getMaximumVelocity() const;
    	double getVortexRadius() const;
    	double getVortexRotation() const;

    	// common:
    	unsigned int getDimension() const;
    	double getTimeStep() const;
    	double getRunTime() const;
    	unsigned int getSavePeriod() const;
    	int getJobID() const;
    	std::string getOutputDirectory() const;

    	// mesh:
    	unsigned int getGlobalRefinement() const;
    	unsigned int getSphereRefinement() const;
    	std::string getMeshFile() const;

    	// chemicals:
    	double getGoodDiffusionConstant() const;
    	double getWasteDiffusionConstant() const;
    	double getGoodDecayConstant() const;
    	double getWasteDecayConstant() const;

    	// bacteria:
    	unsigned int getNumberBacteria() const;
		unsigned int getNumberGroups() const;
		double getBacteriaDiffusionConstant() const;
		double getWasteSecretionRate() const;
		double getGoodSecretionRate() const;
		double getMutationRate() const;

		// fitness:
		double getAlphaGood() const;
		double getAlphaWaste() const;
		double getSecretionCost() const;
		double getGoodSaturation() const;
		double getWasteSaturation() const;

		// for specialization:
		double getInefficiencyPenalty() const;

		// DEBUGGING:
		bool isDebuggingCode() const;
		bool isPrintGrid() const;
		bool isPrintVelocity() const;
		bool isCheckMass() const;
		bool isPointSource() const;
		bool isInitialGaussian() const;
		bool isReproduceBacteria() const;
		double getFlowDelay() const;
		double getReproductionDelay() const;
		double getMutationDelay() const;

		// output:
		void print(std::ostream& out);
		void outputParameters();
	
	private:
		void parse(int argc, char** argv);
		void checkParameters();
    	void parseDirectory(const char* directory);
    	void assignParameters(const char* parameterFile);

		// geometry:
		std::string geometry_file;

    	// advection:
    	VelocityType velocity_type;
    	std::string x_velocity_file;
    	std::string y_velocity_file;

    	double maximum_velocity;
    	double vortex_radius;
    	double vortex_rotation;

    	// common:
    	unsigned int dimension;
    	double time_step;
    	double run_time;
    	unsigned int save_period;
    	int job_ID;
    	std::string output_directory;

    	// mesh:
    	unsigned int global_refinement;
    	unsigned int sphere_refinement;
    	std::string mesh_file;

    	// chemicals:
    	double good_diffusion_constant;
    	double waste_diffusion_constant;
    	double good_decay_constant;
    	double waste_decay_constant;

    	// bacteria:
    	unsigned int number_bacteria;
    	unsigned int number_groups;
    	double bacteria_diffusion_constant;
    	double waste_secretion_rate;
    	double good_secretion_rate;
    	double mutation_rate;


    	// fitness:
		double alpha_good;
		double alpha_waste;
		double secretion_cost;
		double good_saturation;
		double waste_saturation;

		double inefficiency_penalty;

		// DEBUGGING:
		bool debug_code;
		bool print_grid;
		bool print_velocity;
		bool check_mass;
		bool point_source;
		bool initial_gaussian;
		bool reproduce_bacteria;

		double flow_delay;
		double reproduction_delay;
		double mutation_delay;

	};


// IMPLEMENTATIONS:
// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------
	ArgParser::ArgParser(int argc, char** argv)
		:
		velocity_type(VelocityType::NO_FLOW), maximum_velocity(0), 
		vortex_radius(0), vortex_rotation(0), 
		job_ID(0), output_directory("./"), 
		global_refinement(0), sphere_refinement(0), mesh_file(""),
		mutation_rate(0),
		debug_code(false), print_grid(false), print_velocity(false), check_mass(false),
		point_source(false), initial_gaussian(false), reproduce_bacteria(false)
	{
		parse(argc,argv);
	}



// ACCESSORS:
//------------------------------------------------------------------------------------------
	// geometry:
	std::string ArgParser::getGeometryFile() const {return geometry_file;}

	// advection:
	VelocityType ArgParser::getVelocityType() const {return velocity_type;}
	std::string ArgParser::getVelocityFile_X() const {return x_velocity_file;}
	std::string ArgParser::getVelocityFile_Y() const {return y_velocity_file;}
	double ArgParser::getMaximumVelocity() const {return maximum_velocity;}
	double ArgParser::getVortexRadius() const {return vortex_radius;}
	double ArgParser::getVortexRotation() const {return vortex_rotation;}

	// common:
	unsigned int ArgParser::getDimension() const {return dimension;}
	double ArgParser::getTimeStep() const {return time_step;}
	double ArgParser::getRunTime() const {return run_time;}
	unsigned int ArgParser::getSavePeriod() const {return save_period;}
	int ArgParser::getJobID() const {return job_ID;}
	std::string ArgParser::getOutputDirectory() const {return output_directory;}

	// mesh:
	unsigned int ArgParser::getGlobalRefinement() const {return global_refinement;}
	unsigned int ArgParser::getSphereRefinement() const {return sphere_refinement;}
	std::string ArgParser::getMeshFile() const {return mesh_file;}

	// chemicals:
	double ArgParser::getGoodDiffusionConstant() const {return good_diffusion_constant;}
	double ArgParser::getWasteDiffusionConstant() const {return waste_diffusion_constant;}
	double ArgParser::getGoodDecayConstant() const {return good_decay_constant;}
	double ArgParser::getWasteDecayConstant() const {return waste_decay_constant;}

	// bacteria:
	unsigned int ArgParser::getNumberBacteria() const {return number_bacteria;}
	unsigned int ArgParser::getNumberGroups() const {return number_groups;}
	double ArgParser::getBacteriaDiffusionConstant() const {return bacteria_diffusion_constant;}
	double ArgParser::getWasteSecretionRate() const {return waste_secretion_rate;}
	double ArgParser::getGoodSecretionRate() const {return good_secretion_rate;}
	double ArgParser::getMutationRate() const {return mutation_rate;}

	// fitness:
	double ArgParser::getAlphaGood() const {return alpha_good;}
	double ArgParser::getAlphaWaste() const {return alpha_waste;}
	double ArgParser::getSecretionCost() const {return secretion_cost;}
	double ArgParser::getGoodSaturation() const {return good_saturation;}
	double ArgParser::getWasteSaturation() const {return waste_saturation;}

	double ArgParser::getInefficiencyPenalty() const {return inefficiency_penalty;}

	//DEBUGGING:
	bool ArgParser::isDebuggingCode() const {return debug_code;}
	bool ArgParser::isPrintGrid() const {return print_grid;}
	bool ArgParser::isPrintVelocity() const {return print_velocity;}
	bool ArgParser::isCheckMass() const {return check_mass;}
	bool ArgParser::isPointSource() const {return point_source;}
	bool ArgParser::isInitialGaussian() const {return initial_gaussian;}
	bool ArgParser::isReproduceBacteria() const {return reproduce_bacteria;}
	double ArgParser::getFlowDelay() const {return flow_delay;}
	double ArgParser::getReproductionDelay() const {return reproduction_delay;}
	double ArgParser::getMutationDelay() const {return mutation_delay;}

// PARSERS:
// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------
	void ArgParser::parse(int argc, char** argv)
	{

		std::cout << "\n\n...Parsing parameters\n" << std::endl;
		bool odSet = false;
		std::string dirTemp;

		if(argc == 1)
		{
			std::cout << "...No input files given. Using *UNITIIALIZED* parameters. \n"; // won't work...
		}
		else if((argc % 2) == 1) // program + flags + values == odd
		{
			for(int i = 1; i < argc; i += 2)
			{
			  std::string flag = argv[i]; 

			  if(flag.compare("-d") == 0) { parseDirectory(argv[i+1]); }
			  else if(flag.compare("-fp") == 0) { assignParameters(argv[i+1]); }
			  else if(flag.compare("-fg") == 0) { geometry_file = argv[i+1]; }
			  else if(flag.compare("-fvx") == 0) { x_velocity_file = argv[i+1]; }
			  else if(flag.compare("-fvy") == 0) { y_velocity_file = argv[i+1]; }
			  else if(flag.compare("-o") == 0) { odSet = true; dirTemp = argv[i+1]; }
			  else if(flag.compare("-id") == 0 ) { job_ID = atoi(argv[i+1]); }
			  // over write parameters:
			  else if(flag.compare("-diff1") == 0) { good_diffusion_constant = atof(argv[i+1]); }
			  else if(flag.compare("-diff2") == 0) { waste_diffusion_constant = atof(argv[i+1]); }
			  else if(flag.compare("-vscale") == 0) { maximum_velocity = atof(argv[i+1]); }
			  else if(flag.compare("-mrate") == 0) { mutation_rate = atof(argv[i+1]); }
			  else if(flag.compare("-gref") == 0) { global_refinement = (unsigned int)atof(argv[i+1]); }
			  else if(flag.compare("-tmax") == 0) { run_time = atof(argv[i+1]); }
			  else
			  {
			    std::ostringstream message;
			    message << std::endl
			      << "***UNKNOWN FLAG*** \n"
			      << "Usage: ./program [-flags] (file or directory) \n"
			      << "\t -d \"directory\" \n"
			      << "\t -fp \"parameter file\" \n"
			      << "\t -fg \"geometry file\" \n"
			      << "\t -fvx \"x-velocity file\" \n"
			      << "\t -fvy \"y-velocity file\" \n" 
			      << "\t -o \"output directory\" \n"
			      << "\t -id \"job id\" \n" << std::endl;
			    throw std::invalid_argument(message.str());  // SHOULD THROW AN EXECPTION HERE
			  }

			} // for all inputs
		} // if even inputs (-flag value)
		else 
		{
			std::ostringstream message;
			message << std::endl
			  << "***INVALID NUMBER OF INPUT PARAMETERS*** \n"
			  << "Usage: ./program [-flags] (file or directory) \n"
			  << "\t -d \"directory\" \n"
			  << "\t -fp \"parameter file\" \n"
			  << "\t -fg \"geometry file\" \n"
			  << "\t -fvx \"x-velocity file\" \n"
			  << "\t -fvy \"y-velocity file\" \n" 
			  << "\t -o \"output directory\" \n"
			  << "\t -id \"job id\" \n" << std::endl;
			throw std::invalid_argument(message.str()); 
		} // if correct number of inputs


		// CHECK PARAMETERS:
		checkParameters();

		 // CREATE OUTPUT DIRECTORY:
		if(odSet)
			output_directory =  "./data_" + dirTemp + "_jobID_" +  std::to_string(job_ID) + "_" + currentDateTime();
		else
			output_directory =  "./data_jobID_" +  std::to_string(job_ID) + "_" + currentDateTime();

		int check;
		check = mkdir(output_directory.c_str(), 0777);

		if(check == 0)
			std::cout << "...Data output directory created: " << output_directory << std::endl;
		else
			throw std::runtime_error("\n*** FAILED TO CREATE OUTPUT DIRECTORY ***\n"); 
	}


	void ArgParser::parseDirectory(const char* directory)
	{
	  std::string direc(directory);
	  std::string::iterator it = direc.end();
	  --it;
	  if( (*it) != '/')
	    direc += "/"; // add the slash

	  std::cout << "...Configuration directory: " << direc << std::endl;

	  DIR *dir; // pointer to open directory
	  struct dirent *entry; // stuff in directory
	  // struct stat info; // information about each entry

	  //1 open
	  dir = opendir(directory);

	  if(!dir)
	  {
	    throw std::runtime_error("\n***INPUT PARAMETER DIRECTORY NOT FOUND***\n");
	  }

	  // 2 read:
	  while( (entry = readdir(dir)) != NULL )
	  {
	    if(entry->d_name[0] != '.')
	    {
	      std::string fileName = entry->d_name;

	      std::size_t found = fileName.find_last_of(".");
	      std::string extention = fileName.substr(found+1);

	      // std::cout << "filename: " << fileName << ", extention: " << extention << std::endl;

	      if(extention.compare("msh") == 0){ mesh_file = direc + fileName; }
	      if(extention.compare("dat") == 0)
	      {
	        std::size_t fnd = fileName.find_last_of("_");
	        std::string type = fileName.substr(fnd+1);

	        if(type.compare("velx.dat") == 0){ x_velocity_file = direc + fileName; }
	        if(type.compare("vely.dat") == 0){ y_velocity_file = direc + fileName; }
	        if(type.compare("parameters.dat") == 0 || type.compare("para.dat") == 0)
	        {
	          std::string pfile = direc + fileName;
	          assignParameters(pfile.c_str());
	        }
	        if(type.compare("geo.dat") == 0)
	        {
	          geometry_file = direc + fileName;
	        }

	      }

	      // ++i;
	      // std::cout <<"entry " << i <<  " : " << entry->d_name << std::endl;
	    } // if not current or parent directory -- should generalize to is not directory... 
	  } // while files in directory

	  // 3 close:
	  closedir(dir);

	} // parseDirectory()



	void ArgParser::assignParameters(const char* parameterFile)
	{
	  std::cout << "...Reading parameter file: " << parameterFile << std::endl;

	  std::ifstream infile(parameterFile);
	  std::string line;

	  // Input Format: variable value \n
	  while(std::getline(infile,line)){
	    std::istringstream stream(line);
	    std::string varName;
	    double value; 
	    stream >> varName >> value;

	    // CASES:
	    // common:
	    if(varName.compare("dimension") == 0){dimension = (unsigned int)value;}
	    if(varName.compare("run_time") == 0){run_time = value;}
	    if(varName.compare("time_step") == 0){time_step = value;}
	    if(varName.compare("save_period") == 0){save_period = (unsigned int) value;}       

	    //mesh:
	    if(varName.compare("global_refinement") == 0){global_refinement = value;}
	    if(varName.compare("sphere_refinement") == 0){sphere_refinement = value;}

	    //bacteria:
	    if(varName.compare("number_bacteria") == 0){number_bacteria = (unsigned int) value;} 
	    if(varName.compare("groups") == 0){number_groups = (unsigned int) value;} 
	    if(varName.compare("bacteria_diffusion") == 0){bacteria_diffusion_constant = value;}
	    if(varName.compare("good_secretion") == 0){good_secretion_rate = value;}
	    if(varName.compare("waste_secretion") == 0){waste_secretion_rate = value;}
	    if(varName.compare("mutation_rate") == 0){mutation_rate = value;}

	    //fitness:
		if(varName.compare("alpha_good") == 0){alpha_good = value;}
	    if(varName.compare("alpha_waste") == 0){alpha_waste = value;}
	    if(varName.compare("good_saturation") == 0){good_saturation = value;}
	    if(varName.compare("waste_saturation") == 0){waste_saturation = value;}
	    if(varName.compare("secretion_cost") == 0){secretion_cost = value;}
	    if(varName.compare("inefficiency_penalty") == 0){inefficiency_penalty = value;}

	    // chemicals:
	    if(varName.compare("good_diffusion") == 0){good_diffusion_constant = value;}
	    if(varName.compare("waste_diffusion") == 0){waste_diffusion_constant = value;}
	    if(varName.compare("good_decay") == 0){good_decay_constant = value;}
	    if(varName.compare("waste_decay") == 0){waste_decay_constant = value;}

	    // advection:
	    if(varName.compare("velocity_type") == 0 ){velocity_type = (VelocityType)value;} 
		if(varName.compare("maximum_velocity") == 0){maximum_velocity = value;}
		if(varName.compare("vortex_radius") == 0){vortex_radius = value;}
		if(varName.compare("vortex_rotation") == 0){vortex_rotation = value;}

		//DEBUGGING:
		if(varName.compare("debug_mode") == 0){debug_code = (bool)value;}
		if(varName.compare("print_grid") == 0){print_grid = (bool)value;}
		if(varName.compare("print_velocity") == 0){print_velocity = (bool)value;}
		if(varName.compare("check_mass") == 0){check_mass = (bool)value;}
		if(varName.compare("point_source") == 0){point_source = (bool)value;}
		if(varName.compare("initial_gaussian") == 0){initial_gaussian = (bool)value;}
		if(varName.compare("reproduce_bacteria") == 0){reproduce_bacteria = (bool)value;}
		if(varName.compare("flow_delay") == 0){flow_delay = value;}
		if(varName.compare("reproduction_delay") == 0){reproduction_delay = value;}
		if(varName.compare("mutation_delay") == 0){mutation_delay = value;}
	    
	  } // while -- read file and assign variables by case

	}// ArgParser::assignParameters()


	void ArgParser::checkParameters()
	{
	  if(bacteria_diffusion_constant < 0.0 || good_diffusion_constant < 0.0 || 
	  		waste_diffusion_constant < 0.0)
	    throw std::invalid_argument("ERROR: MUST CHOOSE POSITIVE DIFFUSION CONSTANTS.");
	  if(good_decay_constant < 0.0 || waste_decay_constant < 0.0)
	    throw std::invalid_argument("ERROR: MUST CHOOSE POSITIVE DECAY CONSTANTS");
	  if(good_secretion_rate < 0.0 || waste_secretion_rate < 0.0)
	    throw std::invalid_argument("ERROR: MUST CHOOSE POSITIVE SECRETION CONSTANTS");
	  if(alpha_good < 0.0 || alpha_waste < 0.0 || secretion_cost < 0.0)
	    throw std::invalid_argument("ERROR: MUST CHOOSE POSITIVE FITNESS CONSTANTS");
	  if(good_saturation < 0.0 || waste_saturation < 0.0)
	    throw std::invalid_argument("ERROR: MUST CHOOSE POSITIVE SATURATION CONSTANTS");
	  if(time_step < 0 || run_time < 0)
	    throw std::invalid_argument("ERROR: TIME CONSTANTS MUST BE POSITIVE"); // save rate is unsigned
	  // if(Nb < 0)
	  //   throw std::invalid_argument("ERROR: MUST CHOOSE POSITIVE NUMBER OF INITIAL BACTERIA"); // Nb is unsigned...
	} // checkParameters()


// OUTPUT:
	void ArgParser::print(std::ostream& out)
	{
		std::string vtypestr = getVelocityTypeString(velocity_type);

		std::ostringstream programData;
		programData<< "\n\n-----------------------------------------------------" << std::endl
			<< "\t\tSYSTEM PARAMETERS:" << std::endl
			<< "-----------------------------------------------------" << std::endl
		 	<< "\nDATE_TIME: " << currentDateTime() << std::endl
		 	<< "\nINPUT FILES: " << std::endl
		 	<< "\t Geometry File: " << geometry_file << std::endl
		 	<< "\t X-Velocity FIle: " << x_velocity_file << std::endl
		 	<< "\t Y-Velocity File: " << y_velocity_file << std::endl
		 	<< "\nVELOCITY:" << std::endl
		 	<< "\t Velocity Type: " << vtypestr << std::endl
		 	<< "\t Maximum velocity: " << maximum_velocity << std::endl
		 	<< "\t Vortex radius: " << vortex_radius << std::endl
		 	<< "\t Vortex rotation: " << vortex_rotation << std::endl
		 	<< "\nCHEMICALS: " << std::endl
		 	<< "\t Good diffusion: " << good_diffusion_constant << std::endl
		 	<< "\t Waste diffusion: " << waste_diffusion_constant << std::endl
		 	<< "\t Good decay rate: " << good_decay_constant << std::endl
		 	<< "\t Waste decay rate: " << waste_decay_constant << std::endl	
		 	<< "\nBACTERIA: " << std::endl
		 	<< "\t Number initial bacteria: " << number_bacteria << std::endl
		 	<< "\t Number initial groups: " << number_groups << std::endl
		 	<< "\t Bacteria diffusion: " << bacteria_diffusion_constant << std::endl
		 	<< "\t Good secretion rate: " << good_secretion_rate << std::endl
		 	<< "\t Waste secretion rate: " << waste_secretion_rate << std::endl
		 	<< "\t Mutation rate: " << mutation_rate << std::endl
		 	<< "\nFITNESS: " << std::endl
		 	<< "\t Alpha good: " << alpha_good << std::endl
		 	<< "\t Alpha waste: " << alpha_waste << std::endl
		 	<< "\t Secretion cost: " << secretion_cost << std::endl
		 	<< "\t Good saturation: " << good_saturation << std::endl
		 	<< "\t Waste saturation: " << waste_saturation << std::endl
		 	<< "\t Inefficiency penalty: " << inefficiency_penalty << std::endl
		 	<< "\nMESH REFINEMENT:" << std::endl
		 	<< "\t Global refinement: " << global_refinement << std::endl
		 	<< "\t Sphere refinement: " << sphere_refinement << std::endl
		 	<< "\t Mesh file: " << mesh_file << std::endl
		 	<< "\nRUNNING/SAVING: " << std::endl		//@todo print run mode and if debugging
		 	<< "\t Dimension: " << dimension << std::endl
		 	<< "\t Time step: " << time_step << std::endl
		 	<< "\t Run time: " << run_time << std::endl
		 	<< "\t Save period: " << save_period << std::endl
		 	<< "\t Job ID: " << job_ID << std::endl
		 	<< "\t Output directory: " << output_directory << std::endl;
		 	
		if(debug_code)
		{
			programData << "\n***RUNNING DEBUG PARAMETERS***" << std::endl
				<< "\t Print grid: " << print_grid << std::endl
				<< "\t Print velocity: " << print_velocity << std::endl
				<< "\t Check mass: " << check_mass << std::endl
				<< "\t Point source: " << point_source << std::endl
				<< "\t Initial gaussian: " <<  initial_gaussian << std::endl
				<< "\t Reproduce bacteria: " << reproduce_bacteria << std::endl
				<< "\t Flow delay: " << flow_delay << std::endl
				<< "\t Reproduction delay: " << reproduction_delay << std::endl
				<< "\t Mutation delay: " << mutation_delay << std::endl;
		}

		programData << "-----------------------------------------------------\n\n\n" << std::endl;


		out << programData.str(); 
	}

	void ArgParser::outputParameters()
	{
		  std::string outFileName = output_directory + "/programData.dat";
		  std::ofstream outFile(outFileName);
		  print(outFile);
	}


} // namespace MicrobeSimulator
#endif 

