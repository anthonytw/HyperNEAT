#include "NEAT.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "HCUBE_ExperimentRun.h"
#include "Experiments/HCUBE_Experiment.h"
#include "ImageExperiment/ImageExperiment.h"

using namespace NEAT;

class PyHyperNEAT
{
    public:
        /**
         * Perform HyperNEAT initialization through Python.
         */
        static void initialize( void )
        {
            // No initialization necessary at this point.
        }

        /**
         * Perform HyperNEAT cleanup through Python.
         */
        static void cleanup( void )
        {
            NEAT::Globals::deinit();
        }

        /**
         * Returns the global parameters singleton.
         */
        static NEAT::Globals & getGlobalParameters( void )
        {
            return NEAT::Globals::getSingletonRef();
        }

        /**
         * Load a previously saved population from an XML file.
         *
         * @param filename The XML file.
         * @return The population.
         */
        static NEAT::GeneticPopulation * load( const string & filename )
        {
            cout << "Loading population file: " << filename << endl;
                
            TiXmlDocument doc(filename);
            bool loadStatus = false;

            if (iends_with(filename,".gz"))
                loadStatus = doc.LoadFileGZ();
            else
                loadStatus = doc.LoadFile();

            if (!loadStatus)
                throw CREATE_LOCATEDEXCEPTION_INFO("Error trying to load the XML file!");

            TiXmlElement *element = doc.FirstChildElement();
            NEAT::Globals* globals = NEAT::Globals::init(element);

            return new NEAT::GeneticPopulation(filename);
        }

        /**
         * Converts a Python list to a STL vector.
         *
         * @param list The list.
         * @return An STL vector.
         */
        template<class T>
        static vector<T> convertListToVector( python::list * list )
        {
            vector<T> vec;
            for ( int a=0; a<python::len(*list); ++a )
            {
                vec.push_back( 
                    boost::python::extract<T>((*list)[a]) );
            }
            return vec;
        }

        /**
         * Converts a STL vector to a Python list.
         *
         * @param vector The vector.
         * @return The Python list.
         */
        template<class T>
        static python::list convertVectorToList( const std::vector<T> & vector )
        {
            python::object get_iter = python::iterator<std::vector<T> >();
            python::object iter = get_iter(vector);
            python::list l(iter);
            return l;
        }

        /**
         * Sets a substrate layer information.
         */
        static void setLayerInfo(
             shared_ptr<NEAT::LayeredSubstrate<float> > substrate,
             python::list _layerSizes,
             python::list _layerNames,
             python::list _layerAdjacencyList,
             python::list _layerIsInput,
             python::list _layerLocations,
             bool normalize,
             bool useOldOutputNames )
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

        /**
         *
         */
        static void setLayerInfoFromCurrentExperiment(shared_ptr<NEAT::LayeredSubstrate<float> > substrate)
        {
            int experimentType = int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);
            HCUBE::ExperimentRun experimentRun;
            experimentRun.setupExperiment(experimentType,"");

            substrate->setLayerInfo(
                experimentRun.getExperiment()->getLayerInfo()
                );
        }

        /**
         * Returns the size of a substrate layer.
         */
        static python::tuple getLayerSize(shared_ptr<NEAT::LayeredSubstrate<float> > substrate,int index)
        {
            return python::make_tuple(
                python::object(substrate->getLayerSize(index).x),
                python::object(substrate->getLayerSize(index).y)
                );
        }

        /**
         * Returns the substrate layer position.
         */
        static python::tuple getLayerLocation(shared_ptr<NEAT::LayeredSubstrate<float> > substrate,int index)
        {
            return python::make_tuple(
                python::object(substrate->getLayerLocation(index).x),
                python::object(substrate->getLayerLocation(index).y),
                python::object(substrate->getLayerLocation(index).z)
                );
        }

        /**
         * Converts a tuple to an STL vector of floats.
         *
         * @param tuple The tuple.
         * @return A vector of floats.
         */
        static Vector3<float> tupleToVector3Float(python::tuple tuple)
        {
            return Vector3<float>(
                python::extract<float>(tuple[0]),
                python::extract<float>(tuple[1]),
                python::extract<float>(tuple[2])
            );
        }

        /**
         * Converts a tuple to an STL vector of ints.
         *
         * @param tuple The tuple.
         * @return A vector of ints.
         */
        static Vector3<int> tupleToVector3Int(python::tuple tuple)
        {
            return Vector3<int>(
                python::extract<int>(tuple[0]),
                python::extract<int>(tuple[1]),
                python::extract<int>(tuple[2])
            );
        }

        /**
         * Configure an experiment to run. Call when the experiment will be mainly
         * conducted in Python.
         *
         * @param experiment_filename The filename of the experiment initial
         *      conditions.
         * @param output_filename The filename of the output file.
         * @return The ExperimentRun object.
         */
        static shared_ptr<HCUBE::ExperimentRun> setupExperiment(
            const string & experiment_filename,
            const string & output_filename )
        {
            cout << "CONFIGURING EXPERIMENT:" << endl;
            cout << " : Globals : " << experiment_filename << endl;
            cout << " : Output  : " << output_filename << endl;
            
            NEAT::Globals::init(experiment_filename);
            int experimentType = int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);

            cout << "- Loading Experiment: " << experimentType << "..." << flush;
            shared_ptr<HCUBE::ExperimentRun> experimentRun(new HCUBE::ExperimentRun());
            experimentRun->setupExperiment(experimentType, output_filename);
            cout << "Done" << endl;
            
            cout << "- Creating population..." << flush;
            experimentRun->createPopulation();
            experimentRun->setCleanup(true);
            cout << "Done" << endl;

            cout << "- Setup complete" << endl;

            return experimentRun;
        }

        /**
         * Configure an experiment and then run it through the C++ interface.
         * In this case the experiment will run in C++ and the results will
         * be returned to Python.
         *
         * @param experiment_filename The filename of the experiment initial
         *      conditions.
         * @param output_filename The filename of the output file.
         * @return The ExperimentRun object.
         */
        static shared_ptr<HCUBE::ExperimentRun> setupAndRunExperiment(
                const string & experiment_filename,
                const string & output_filename )
        {
            // Setup experiment.
            shared_ptr<HCUBE::ExperimentRun> experimentRun = setupExperiment(experiment_filename, output_filename);

            // Run experiment.
            cout << "- Running experiment..." << endl;
            cout << "===========================================" << endl;
            experimentRun->start();
            cout << "===========================================" << endl;
            cout << "- Experiment finished" << endl;

            // Finished, now return.
            return experimentRun;
        }

        /**
         * Returns the type of experiment being run.
         */
        static int getExperimentType()
        {
            return int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);
        }

        /**
         * Returns the maximum number of generations for the experiment.
         */
        static int getMaximumGenerations()
        {
            return int(NEAT::Globals::getSingleton()->getParameterValue("MaxGenerations"));
        }
};

/**
 * The following code section will register the bindings from C++ to Python.
 * 
 * Specifying classes with python::no_init will prevent Python from creating
 * new objects of that type.
 *
 * Specifying classes with boost::noncopyable will prevent Python from copying
 * objects instead of passing references.
 *
 * Both of these should be used to avoid oddities.
 */
BOOST_PYTHON_MODULE(PyHyperNEAT)
{
	//To prevent instances being created from python, you add boost::python::no_init to the class_ constructor
    python::class_<HCUBE::ExperimentRun , shared_ptr<HCUBE::ExperimentRun>,boost::noncopyable >("ExperimentRun",python::no_init)
		.def("produceNextGeneration", &HCUBE::ExperimentRun::produceNextGeneration)
		.def("finishEvaluations", &HCUBE::ExperimentRun::finishEvaluations)
		.def("preprocessPopulation", &HCUBE::ExperimentRun::preprocessPopulation)
		.def("pythonEvaluationSet", &HCUBE::ExperimentRun::pythonEvaluationSet)
        .def("saveBest", &HCUBE::ExperimentRun::saveBest)
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
        .def("saveToFile", &NEAT::GeneticIndividual::saveToFile)
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
	    .def("setLayerInfo", &PyHyperNEAT::setLayerInfo)
	    .def("setLayerInfoFromCurrentExperiment", &PyHyperNEAT::setLayerInfoFromCurrentExperiment)
		.def("getNetwork", &NEAT::LayeredSubstrate<float>::getNetwork, python::return_value_policy<python::reference_existing_object>())
		.def("getNumLayers", &NEAT::LayeredSubstrate<float>::getNumLayers)
		.def("setValue", &NEAT::LayeredSubstrate<float>::setValue)
		.def("getLayerSize", &PyHyperNEAT::getLayerSize)
		.def("getLayerLocation", &PyHyperNEAT::getLayerLocation)
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

    python::class_<NEAT::Globals, boost::noncopyable >("Globals", python::no_init)
        .def("getParameterCount", &NEAT::Globals::getParameterCount)
        .def("getParameterName", &NEAT::Globals::getParameterName,
            python::return_value_policy<python::copy_const_reference>() )
        .def("getParameterValue", &NEAT::Globals::getParameterValue)
        .def("setParameterValue", &NEAT::Globals::setParameterValue)
    ;

	python::def("load", PyHyperNEAT::load, python::return_value_policy<python::manage_new_object>());
    python::def("initialize", PyHyperNEAT::initialize);
	python::def("cleanup", PyHyperNEAT::cleanup);
    python::def("getGlobalParameters",
        PyHyperNEAT::getGlobalParameters,
        python::return_value_policy<python::reference_existing_object>() );
	python::def("tupleToVector3Int", PyHyperNEAT::tupleToVector3Int, python::return_value_policy<python::return_by_value>());
	python::def("setupExperiment", PyHyperNEAT::setupExperiment);
    python::def("setupAndRunExperiment", PyHyperNEAT::setupAndRunExperiment);
    python::def("getExperimentType", PyHyperNEAT::getExperimentType);
    python::def("getMaximumGenerations", PyHyperNEAT::getMaximumGenerations);
}
