#include "HCUBE_Defines.h"

#include "HCUBE_ExperimentRun.h"

#include "Experiments/HCUBE_Experiment.h"

#include "Experiments/HCUBE_XorExperiment.h"


#ifdef EPLEX_INTERNAL
//#include "Experiments/HCUBE_XorCoExperiment.h"
//#include "Experiments/HCUBE_SimpleImageExperiment.h"
//#include "Experiments/HCUBE_RobotArmExperiment.h"
//#include "Experiments/HCUBE_RobotArmLatticeExperiment.h"
#endif
//#include "Experiments/HCUBE_TicTacToeExperiment.h"
//#include "Experiments/HCUBE_TicTacToeGameExperiment.h"
//#include "Experiments/HCUBE_TicTacToeGameNoGeomExperiment.h"
//#include "Experiments/HCUBE_FindPointExperiment.h"
//#include "Experiments/HCUBE_FindClusterExperiment.h"
//#include "Experiments/HCUBE_FindClusterNoGeomExperiment.h"
//#include "Experiments/HCUBE_CheckersExperiment.h"
#ifdef EPLEX_INTERNAL
//#include "Experiments/HCUBE_SpatialExperiment.h"
//#include "Experiments/HCUBE_CheckersScalingExperiment.h"
#endif
//#include "Experiments/HCUBE_CheckersExperimentNoGeom.h"
#ifdef EPLEX_INTERNAL
//#include "Experiments/HCUBE_CoCheckersExperiment.h"
//#include "Experiments/HCUBE_CheckersExperimentFogel.h"
#endif
//#include "Experiments/HCUBE_CheckersExperimentOriginalFogel.h"
#ifdef EPLEX_INTERNAL
//#include "Experiments/HCUBE_CheckersExperimentPruning.h"
//#include "Experiments/HCUBE_OthelloExperiment.h"
//#include "Experiments/HCUBE_OthelloCoExperiment.h"
//#include "Experiments/HCUBE_SpatialCasinoExperiment.h"
//#include "Experiments/HCUBE_ImageCompressionExperiment.h"
//#include "Experiments/HCUBE_BinaryCompressionExperiment.h"
//#include "Experiments/HCUBE_FindClusterBPExperiment.h"
//#include "Experiments/HCUBE_GoExperiment.h"
#endif
//#include "Experiments/HCUBE_CheckersExperimentSubstrateGeom.h"

#include "xor3par/HCUBE_Xor3ParExperiment.h"
#include "ImageExperiment/ImageExperiment.h"

#ifndef HCUBE_NOGUI
#include "HCUBE_MainFrame.h"
#include "HCUBE_UserEvaluationFrame.h"

#include "HCUBE_ExperimentPanel.h"
#endif

namespace HCUBE
{
    ExperimentRun::ExperimentRun()
        :
        running(false),
        started(false),
        cleanup(false),
        populationMutex(new mutex()),
        frame(NULL)
    {
    }

    ExperimentRun::~ExperimentRun()
    {
        delete populationMutex;
    }

    void ExperimentRun::setupExperiment(
        int _experimentType,
        string _outputFileName
        )
    {
        experimentType = _experimentType;
        outputFileName = _outputFileName;

        cout << "SETTING UP EXPERIMENT TYPE: " << experimentType << endl;

        for(int a=0;a<NUM_THREADS;a++)
        {
            switch (experimentType)
            {
                case EXPERIMENT_XOR:
                    experiments.push_back(shared_ptr<Experiment>(new XorExperiment("",a)));
                    break;
                case EXPERIMENT_XOR_3PARITY_ATW:
                    experiments.push_back(shared_ptr<Experiment>(new Xor3ParExperiment("Xor 3-Parity", a)));
                    break;
                case EXPERIMENT_IMAGE:
                    experiments.push_back(shared_ptr<Experiment>(new ImageExperiment("Image Experiment", a)));
                    break;
                default:
                    cout << string("ERROR: Unknown Experiment Type!\n");
                    throw CREATE_LOCATEDEXCEPTION_INFO("ERROR: Unknown Experiment Type!");
            }

        }

    }

    void ExperimentRun::createPopulation(string populationString)
    {
        if (iequals(populationString,""))
        {
            int popSize = (int)NEAT::Globals::getSingleton()->getParameterValue("PopulationSize");
            population = shared_ptr<NEAT::GeneticPopulation>(
                experiments[0]->createInitialPopulation(popSize)
                );
        }
        else
        {
#ifdef EPLEX_INTERNAL
            try
            {
                if (dynamic_cast<NEAT::CoEvoExperiment*>(experiments[0].get()))
                {
                    population = shared_ptr<NEAT::GeneticPopulation>(
                        new NEAT::GeneticPopulation(
                            populationString,
                            shared_ptr<NEAT::CoEvoExperiment>((NEAT::CoEvoExperiment*)experiments[0]->clone())
                            )
                        );
                    return;
                }
            }
            catch (const std::exception &ex)
            {
                throw CREATE_LOCATEDEXCEPTION_INFO(string("EXCEPTION ON DYNAMIC CAST: ")+string(ex.what()));
            }
#endif

            population = shared_ptr<NEAT::GeneticPopulation>(
                new NEAT::GeneticPopulation(populationString)
                );
        }
    }

    void ExperimentRun::setupExperimentInProgress(
        string populationFileName,
        string _outputFileName
        )
    {
        outputFileName = _outputFileName;

        {
            TiXmlDocument doc(populationFileName);

            bool loadStatus;

            if (iends_with(populationFileName,".gz"))
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

        int experimentType = int(NEAT::Globals::getSingleton()->getParameterValue("ExperimentType")+0.001);

        cout << "Loading Experiment: " << experimentType << endl;

        setupExperiment(experimentType,_outputFileName);

        cout << "Experiment set up.  Creating population...\n";

        createPopulation(populationFileName);

        cout << "Population Created\n";
    }

    void ExperimentRun::start()
    {
        cout << "Experiment started\n";
#ifndef _DEBUG
        try
        {
#endif
            int maxGenerations = int(NEAT::Globals::getSingleton()->getParameterValue("MaxGenerations"));

            started=running=true;

            int firstGen = (population->getGenerationCount()-1);
            for (int generations=firstGen;generations<maxGenerations;generations++)
            {
                if (generations>firstGen)
                {
                    //Even if we are loading an existing population,
                    //Re-evaluate all of the individuals
                    //as a sanity check
                    mutex::scoped_lock scoped_lock(*populationMutex);
                    cout << "PRODUCING NEXT GENERATION\n";
                    produceNextGeneration();
                    cout << "DONE PRODUCING\n";
                }

                if (experiments[0]->performUserEvaluations())
                {
#ifdef HCUBE_NOGUI
                    throw CREATE_LOCATEDEXCEPTION_INFO("ERROR: TRIED TO USE INTERACTIVE EVOLUTION WITH NO GUI!");
#else
                    frame->getUserEvaluationFrame()->updateEvaluationPanels();
                    running=false;
                    while (!running)
                    {
                        boost::xtime xt;
                        boost::xtime_get(&xt, boost::TIME_UTC_);
                        xt.sec += 1;
                        boost::thread::sleep(xt); // Sleep for 1/2 second
                        //cout << "Sleeping while user evaluates!\n";
                    }
#endif
                }
                else
                {
                    while (!running)
                    {
                        boost::xtime xt;
                        boost::xtime_get(&xt, boost::TIME_UTC_);
                        xt.sec += 1;
                        boost::thread::sleep(xt); // Sleep for 1/2 second
                    }
					preprocessPopulation();
                    evaluatePopulation();
                }

                cout << "Finishing evaluations\n";
                finishEvaluations();
                cout << "Evaluations Finished\n";
            }
            cout << "Experiment finished\n";

            // Save best individuals.
            saveBest();

#ifndef _DEBUG
        }
        catch (const std::exception &ex)
        {
            cout << "CAUGHT ERROR AT " << __FILE__ << " : " << __LINE__ << endl;
            CREATE_PAUSE(ex.what());
        }
        catch (...)
        {
            cout << "CAUGHT ERROR AT " << __FILE__ << " : " << __LINE__ << endl;
            CREATE_PAUSE("AN UNKNOWN EXCEPTION HAS BEEN THROWN!");
        }
#endif
    }

	void ExperimentRun::preprocessPopulation()
	{
		cout << "PREPROCESSING POPULATION\n";
        shared_ptr<NEAT::GeneticGeneration> generation = population->getGeneration();

		for(int a=0;a<generation->getIndividualCount();a++)
		{
			experiments[0]->preprocessIndividual(generation,generation->getIndividual(a));
		}

		cout << "PREPROCESSING COMPLETED\n";
	}

	shared_ptr<NEAT::GeneticGeneration> ExperimentRun::pythonEvaluationSet(){

		shared_ptr<NEAT::GeneticGeneration> generation = population->getGeneration();
		//Randomize population order for evaluation
		generation->randomizeIndividualOrder();

		//int populationSize = population->getIndividualCount();

		/*EvaluationSet* evalSet(
		                experiments[0],
		                generation,
		                population->getIndividualIterator(0),
		                populationSize
		                );*/

		/*shared_ptr<HCUBE::EvaluationSet>eval(new HCUBE::EvaluationSet(experiments[0],
                generation,
                population->getIndividualIterator(0),
                populationSize
                ));

		shared_ptr<NEAT::GeneticGeneration> individuals = eval->runPython();*/

		cout << "EXPORTED Individuals\n";

		return generation;
	}

    void ExperimentRun::evaluatePopulation()
    {
        shared_ptr<NEAT::GeneticGeneration> generation = population->getGeneration();
        //Randomize population order for evaluation
        generation->randomizeIndividualOrder();

        int populationSize = population->getIndividualCount();

        if(NUM_THREADS==1)
        {
            //Bypass the threading logic for a single thread

            EvaluationSet evalSet(
                experiments[0],
                generation,
                population->getIndividualIterator(0),
                populationSize
                );
            evalSet.run();
        }
        else
        {
            int populationPerProcess = populationSize/NUM_THREADS;

            boost::thread** threads = new boost::thread*[NUM_THREADS];
            EvaluationSet** evaluationSets = new EvaluationSet*[NUM_THREADS];

            for (int i = 0; i < NUM_THREADS; ++i)
            {
                if (i+1==NUM_THREADS)
                {
                    //Fix for uneven distribution
                    int populationIteratorSize =
                        populationSize
                        - populationPerProcess*(NUM_THREADS-1);
                    evaluationSets[i] =
                        new EvaluationSet(
                            experiments[i],
                            generation,
                            population->getIndividualIterator(populationPerProcess*i),
                            populationIteratorSize
                            );
                }
                else
                {

                    evaluationSets[i] =
                        new EvaluationSet(
                            experiments[i],
                            generation,
                            population->getIndividualIterator(populationPerProcess*i),
                            populationPerProcess
                            );
                }

                threads[i] =
                    new boost::thread(
                        boost::bind(
                            &EvaluationSet::run,
                            evaluationSets[i]
                            )
                        );

            }

            //loop through each thread, making sure it is finished before we move on
            for (int i=0;i<NUM_THREADS;++i)
            {
                /*if (!evaluationSets[i]->isFinished())
                  {
                  --i;
                  boost::xtime xt;
                  boost::xtime_get(&xt, boost::TIME_UTC);
                  xt.sec += 1;
                  boost::thread::sleep(xt); // Sleep for 1/2 second
                  }*/
                threads[i]->join();
            }

            for (int i = 0; i < NUM_THREADS; ++i)
            {
                delete threads[i];
                delete evaluationSets[i];
            }

            delete[] threads;
            delete[] evaluationSets;
        }
    }

    void ExperimentRun::finishEvaluations()
    {
        cout << "Adjusting fitness...\n";
        population->adjustFitness();
        cout << "Cleaning up...\n";

        //int generationDumpModulo = int(NEAT::Globals::getSingleton()->getParameterValue("GenerationDumpModulo"));
        //if (cleanup)
        //    population->cleanupOld(INT_MAX/2);
        //cout << "Dumping best individuals...\n";
        //population->dumpBest(outputFileName+string(".backup.xml"),true,true);
        //population->cleanupOld(25);
        //population->dumpBest("out/dumpBestWithGenes(backup).xml",true);

#ifndef HCUBE_NOGUI
        if (frame)
        {
            frame->updateNumGenerations(population->getGenerationCount());
        }
#endif

        cout << "Resetting generation data...\n";
        shared_ptr<NEAT::GeneticGeneration> generation = population->getGeneration();
        experiments[0]->resetGenerationData(generation);

        for (int a=0;a<population->getIndividualCount();a++)
        {
            //cout << __FILE__ << ":" << __LINE__ << endl;
            experiments[0]->addGenerationData(generation,population->getIndividual(a));
        }

        /*
          if(experimentType == EXPERIMENT_TIC_TAC_TOE_GAME)
          {
          //Take the best individual and run him a lot

          ((TicTacToeGameExperiment*)experiment)->setNumGames(20000);
          ((TicTacToeGameExperiment*)experiment)->setNumGames(100);
          }
        */
    }

    void ExperimentRun::produceNextGeneration()
    {
        cout << "Producing next generation.\n";
        try
        {
            population->produceNextGeneration();
        }
        catch (const std::exception &ex)
        {
            cout << "EXCEPTION DURING POPULATION REPRODUCTION: " << endl;
            CREATE_PAUSE(ex.what());
        }
        catch (...)
        {
            cout << "Unhandled Exception\n";
        }
    }

    void ExperimentRun::saveBest( void )
    {
        cout << "Saving best individuals...";
        string bestFileName = outputFileName.substr(0,outputFileName.length()-4)+string("_best.xml");
        population->dumpBest(bestFileName,true,false);
        cout << "Done!\n";
    }
}
