#include "NEAT.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


#include "HCUBE_ExperimentRun.h"
#include "Experiments/HCUBE_Experiment.h"
#include "ImageExperiment/ImageExperiment.h"

//#include "Experiments/HCUBE_CheckersExperiment.h"

//#include "SgInit.h"
//#include "GoInit.h"

using namespace NEAT;

NEAT::GeneticPopulation* loadFromPopulation(string filename)
{
	string populationFilename = filename;
	cout << "Loading population file: " << populationFilename << endl;

	{
		TiXmlDocument doc(populationFilename);

		bool loadStatus;

		if (iends_with(populationFilename,".gz"))
		{
			loadStatus = doc.LoadFileGZ();
		}
		else
		{
			loadStatus = doc.LoadFile();
		}

		if (!loadStatus)
		{
			throw CREATE_LOCATEDEXCEPTION_INFO("Error trying to load the XML file!");
		}

		TiXmlElement *element = doc.FirstChildElement();

		NEAT::Globals* globals = NEAT::Globals::init(element);

		//Destroy the document
	}

	return new NEAT::GeneticPopulation(populationFilename);
}

void initializeHyperNEAT()
{
    /*
    char str[1024];
    initcake(str);
    SgInit();
    GoInit();
    */
}

void cleanupHyperNEAT()
{
	NEAT::Globals::deinit();
}

template<class T>
vector<T> convertListToVector(python::list *list)
{
    vector<T> vec;
    for(int a=0;a<python::len(*list);a++)
    {
        vec.push_back( 
            boost::python::extract<T>((*list)[a])
            );
    }
    return vec;
}

template<class T>
python::list std_vector_to_py_list(const std::vector<T>& v)
{
    python::object get_iter = python::iterator<std::vector<T> >();
    python::object iter = get_iter(v);
    python::list l(iter);
    return l;
}

void Py_setLayerInfo(
                     shared_ptr<NEAT::LayeredSubstrate<float> > substrate,
                     python::list _layerSizes,
                     python::list _layerNames,
                     python::list _layerAdjacencyList,
                     python::list _layerIsInput,
                     python::list _layerLocations,
					 bool normalize,
                     bool useOldOutputNames
                     )
{
    NEAT::LayeredSubstrateInfo layerInfo;

    for(int a=0;a<python::len(_layerSizes);a++)
    {
        layerInfo.layerSizes.push_back( 
            JGTL::Vector2<int>( 
            boost::python::extract<int>((_layerSizes)[a][0]), 
            boost::python::extract<int>((_layerSizes)[a][1])
            ) 
            );
    }

    for(int a=0;a<python::len(_layerNames);a++)
    {
        layerInfo.layerNames.push_back( 
            boost::python::extract<string>(_layerNames[a])
            );
    }

    for(int a=0;a<python::len(_layerAdjacencyList);a++)
    {
        layerInfo.layerAdjacencyList.push_back( 
            std::pair<string,string>( 
            boost::python::extract<string>((_layerAdjacencyList)[a][0]), 
            boost::python::extract<string>((_layerAdjacencyList)[a][1])
            ) 
            );
    }

    vector< bool > layerIsInput = convertListToVector<bool>(&_layerIsInput);
    for(int a=0;a<int(layerIsInput.size());a++)
    {
        layerInfo.layerIsInput.push_back(layerIsInput[a]);
    }

    for(int a=0;a<python::len(_layerLocations);a++)
    {
        layerInfo.layerLocations.push_back( 
            JGTL::Vector3<float>( 
            boost::python::extract<float>((_layerLocations)[a][0]), 
            boost::python::extract<float>((_layerLocations)[a][1]),
            boost::python::extract<float>((_layerLocations)[a][2])
            ) 
            );
    }

    layerInfo.normalize = normalize;
    layerInfo.useOldOutputNames = useOldOutputNames;

    substrate->setLayerInfo(layerInfo);
}

void Py_setLayerInfoFromCurrentExperiment(shared_ptr<NEAT::LayeredSubstrate<float> > substrate)
{
    int experimentType = int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);
    HCUBE::ExperimentRun experimentRun;
    experimentRun.setupExperiment(experimentType,"");

    substrate->setLayerInfo(
        experimentRun.getExperiment()->getLayerInfo()
		);
}

python::tuple Py_getLayerSize(shared_ptr<NEAT::LayeredSubstrate<float> > substrate,int index)
{
    return python::make_tuple(
        python::object(substrate->getLayerSize(index).x),
        python::object(substrate->getLayerSize(index).y)
        );
}

python::tuple Py_getLayerLocation(shared_ptr<NEAT::LayeredSubstrate<float> > substrate,int index)
{
    return python::make_tuple(
        python::object(substrate->getLayerLocation(index).x),
        python::object(substrate->getLayerLocation(index).y),
        python::object(substrate->getLayerLocation(index).z)
        );
}

Vector3<float> tupleToVector3Float(python::tuple t)
{
	return Vector3<float>(
		python::extract<float>(t[0]),
		python::extract<float>(t[1]),
		python::extract<float>(t[2])
	);
}

Vector3<int> tupleToVector3Int(python::tuple t)
{
	return Vector3<int>(
		python::extract<int>(t[0]),
		python::extract<int>(t[1]),
		python::extract<int>(t[2])
	);
}

shared_ptr<HCUBE::ExperimentRun> Py_setupExperiment(string file,string outputFile)
{
    cout << "LOADING GLOBALS FROM FILE: " << file << endl;
    cout << "OUTPUT FILE: " << outputFile << endl;
    NEAT::Globals::init(file);

    int experimentType = int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);

    cout << "Loading Experiment: " << experimentType << endl;

    shared_ptr<HCUBE::ExperimentRun> experimentRun(new HCUBE::ExperimentRun());

    experimentRun->setupExperiment(experimentType,outputFile);

    cout << "Experiment set up\n";

    experimentRun->createPopulation();

    experimentRun->setCleanup(true);

    cout << "Population Created\n";

    experimentRun->start();

    return experimentRun;
}

shared_ptr<HCUBE::ExperimentRun> Py_Experiment(string file,string outputFile)
{
    cout << "LOADING GLOBALS FROM FILE: " << file << endl;
    cout << "OUTPUT FILE: " << outputFile << endl;
    NEAT::Globals::init(file);

    int experimentType = int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);

    cout << "Loading Experiment: " << experimentType << endl;

    shared_ptr<HCUBE::ExperimentRun> experimentRun(new HCUBE::ExperimentRun());

    experimentRun->setupExperiment(experimentType,outputFile);

    cout << "Experiment set up\n";

    experimentRun->createPopulation();

    experimentRun->setCleanup(true);

    cout << "Population Created\n";

    cout << "Experiment Setup\n";

    return experimentRun;
}

int Py_getExperimentType()
{
    return int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);
}

int Py_getMaximumGenerations()
{
	return int(NEAT::Globals::getSingleton()->getParameterValue("MaxGenerations"));
}

// Configure some function references.

BOOST_PYTHON_MODULE(PyHyperNEAT)
{

	//To prevent instances being created from python, you add boost::python::no_init to the class_ constructor
    python::class_<HCUBE::ExperimentRun , shared_ptr<HCUBE::ExperimentRun>,boost::noncopyable >("ExperimentRun",python::no_init)
		.def("produceNextGeneration", &HCUBE::ExperimentRun::produceNextGeneration)
		.def("finishEvaluations", &HCUBE::ExperimentRun::finishEvaluations)
		.def("preprocessPopulation", &HCUBE::ExperimentRun::preprocessPopulation)
		.def("pythonEvaluationSet", &HCUBE::ExperimentRun::pythonEvaluationSet)
    ;

    python::class_<NEAT::GeneticPopulation , shared_ptr<NEAT::GeneticPopulation> >("GeneticPopulation",python::init<>())
		.def("getIndividual", &NEAT::GeneticPopulation::getIndividual)
		.def("getGenerationCount", &NEAT::GeneticPopulation::getGenerationCount)
		.def("getIndividualCount", &NEAT::GeneticPopulation::getIndividualCount)
	;

    python::class_<NEAT::GeneticGeneration , shared_ptr<NEAT::GeneticGeneration>,boost::noncopyable >("GeneticGeneration",python::no_init)
		.def("getIndividual", &NEAT::GeneticGeneration::getIndividual)
		.def("getIndividualCount", &NEAT::GeneticGeneration::getIndividualCount)
		.def("cleanup",&NEAT::GeneticGeneration::cleanup)
		.def("sortByFitness",&NEAT::GeneticGeneration::sortByFitness)
	;

	python::class_<NEAT::GeneticIndividual , shared_ptr<NEAT::GeneticIndividual>,boost::noncopyable >("GeneticIndividual", python::no_init)
	    .def("spawnFastPhenotypeStack", &NEAT::GeneticIndividual::spawnFastPhenotypeStack<float>)
        .def("getNodesCount", &NEAT::GeneticIndividual::getNodesCount)
        //.def("getNode", &NEAT::GeneticIndividual::getNode)
        .def("getLinksCount", &NEAT::GeneticIndividual::getLinksCount)
        //.def("getLink", &NEAT::GeneticIndividual::getLink)
        .def("linkExists", &NEAT::GeneticIndividual::linkExists)
        .def("getFitness", &NEAT::GeneticIndividual::getFitness)
        .def("getSpeciesID", &NEAT::GeneticIndividual::getSpeciesID)
        .def("isValid", &NEAT::GeneticIndividual::isValid)
        .def("printIndividual", &NEAT::GeneticIndividual::print)
        .def("reward",&NEAT::GeneticIndividual::reward)
    ;

	python::class_<std::vector<shared_ptr<NEAT::GeneticIndividual> > >("GeneticIndividualVector")
		.def(python::vector_indexing_suite<std::vector<shared_ptr<NEAT::GeneticIndividual> > >())
	;

    python::class_<NEAT::FastNetwork<float> , shared_ptr<NEAT::FastNetwork<float> > >("FastNetwork",python::init<>())
		.def("reinitialize", &NEAT::FastNetwork<float>::reinitialize)
		.def("update", &NEAT::FastNetwork<float>::update)
		.def("updateFixedIterations", &NEAT::FastNetwork<float>::updateFixedIterations)
		.def("getValue", &NEAT::FastNetwork<float>::getValue)
        .def("setValue", &NEAT::FastNetwork<float>::setValue)
		.def("hasLink", &NEAT::FastNetwork<float>::hasLink)
		.def("getLinkWeight", &NEAT::FastNetwork<float>::getLinkWeight)
    ;

	python::class_<NEAT::LayeredSubstrate<float> , shared_ptr<NEAT::LayeredSubstrate<float> > >("LayeredSubstrate",python::init<>())
		.def("populateSubstrate", &NEAT::LayeredSubstrate<float>::populateSubstrate)
	    .def("setLayerInfo", &Py_setLayerInfo)
	    .def("setLayerInfoFromCurrentExperiment", &Py_setLayerInfoFromCurrentExperiment)
		.def("getNetwork", &NEAT::LayeredSubstrate<float>::getNetwork, python::return_value_policy<python::reference_existing_object>())
		.def("getNumLayers", &NEAT::LayeredSubstrate<float>::getNumLayers)
		.def("setValue", &NEAT::LayeredSubstrate<float>::setValue)
		.def("getLayerSize", &Py_getLayerSize)
		.def("getLayerLocation", &Py_getLayerLocation)
		.def("getWeightRGB", &NEAT::LayeredSubstrate<float>::getWeightRGB)
		.def("getActivationRGB", &NEAT::LayeredSubstrate<float>::getActivationRGB)
		.def("dumpWeightsFrom", &NEAT::LayeredSubstrate<float>::dumpWeightsFrom)
		.def("dumpActivationLevels", &NEAT::LayeredSubstrate<float>::dumpActivationLevels)
	;

	python::class_<HCUBE::ImageExperiment , shared_ptr<HCUBE::ImageExperiment>,boost::noncopyable >("ImageExperiment",python::no_init)
		.def("setReward",&HCUBE::ImageExperiment::setReward)
	;

	/*python::class_<HCUBE::EvaluationSet , shared_ptr<HCUBE::EvaluationSet>,boost::noncopyable >("EvaluationSet",python::no_init)
		.def("runPython", &HCUBE::EvaluationSet::runPython)
		.def("getExperimentObject", &HCUBE::EvaluationSet::getExperimentObject)
	;*/

	python::class_<Vector3<int> >("NEAT_Vector3",python::init<>())
		.def(python::init<int,int,int>())
	;

	python::def("loadFromPopulation", loadFromPopulation, python::return_value_policy<python::manage_new_object>());
    python::def("initializeHyperNEAT", initializeHyperNEAT);
	python::def("cleanupHyperNEAT", cleanupHyperNEAT);
	python::def("tupleToVector3Int", tupleToVector3Int, python::return_value_policy<python::return_by_value>());
	python::def("Py_Experiment", Py_Experiment);
    python::def("setupExperiment", Py_setupExperiment);

    python::def("getExperimentType", Py_getExperimentType);
    python::def("getMaximumGenerations",Py_getMaximumGenerations);
}
