package ml.comet.examples

import com.beust.jcommander.{JCommander, Parameter, ParameterException}
import ml.comet.experiment.builder.OnlineExperimentBuilder
import ml.comet.experiment.{ExperimentBuilder, OnlineExperiment}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.util.Collections
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
/**
 * Train a simple, small MLP on MNIST data using Spark with Comet.
 *
 * <p>To run from command line execute the following at the root of this module:
 * <pre>
 * COMET_API_KEY=your_api_key \
 * COMET_WORKSPACE_NAME=your_workspace \
 * COMET_PROJECT_NAME=your_project_name \
 * mvn exec:java -Dexec.mainClass="ml.comet.examples.MNistExampleSpark"
 * </pre>
 */
object MNistExampleSpark {

  lazy val log: Logger = LoggerFactory.getLogger(classOf[MNistExampleSpark])

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    val experiment = ExperimentBuilder.OnlineExperiment.withApiKey("bla").withProjectName("the-project").build();

    try {
      new MNistExampleSpark().runMnistExperiment(args, experiment)
    } catch {
      case e: Exception => log.error("failed to run experiment", e)
    } finally {
      // make sure to close experiment at the end
      experiment.close()
    }
    sys.exit(0)
  }
}

class MNistExampleSpark {

  @Parameter(names = Array("--useSparkLocal"), description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
  var useSparkLocal: Boolean = true

  @Parameter(names = Array("--batchSizePerWorker"), description = "Number of examples to fit each worker with")
  var batchSizePerWorker: Int = 16

  @Parameter(names = Array("--numEpochs"), description = "Number of epochs for training")
  var numEpochs: Int = 10


  @throws[Exception]
  protected def runMnistExperiment(args: Array[String], experiment: OnlineExperiment): Unit = {
    experiment.setInterceptStdout()

    parseCommandLineArgs(args)

    experiment.logParameter("useSparkLocal", useSparkLocal)
    experiment.logParameter("batchSizePerWorker", batchSizePerWorker)
    experiment.logParameter("numEpochs", numEpochs)

    val rngSeed = 123 // random number seed for reproducibility

    // Create Spark context and convert MNIST data set into RDD
    //
    val sparkConf = new SparkConf
    if (useSparkLocal) {
      sparkConf.setMaster("local[*]")
    }
    sparkConf.setAppName("Comet MNIST Experiment with Spark (Scala)")
    sparkConf.getAll.foreach(x => experiment.logOther(String.valueOf(x._1), x._2))
    val sc = new JavaSparkContext(sparkConf)

    // Load the data into memory then parallelize
    // This isn't a good approach in general - but is simple to use for this example
    val iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, rngSeed)
    val iterTest = new MnistDataSetIterator(batchSizePerWorker, false, rngSeed)
    val trainDataList: ArrayBuffer[DataSet] = mutable.ArrayBuffer.empty[DataSet]
    val testDataList = mutable.ArrayBuffer.empty[DataSet]
    while (iterTrain.hasNext) {
      trainDataList += iterTrain.next
    }

    while (iterTest.hasNext) {
      testDataList += iterTest.next
    }

    val trainData: JavaRDD[DataSet] = sc.parallelize(trainDataList.toList)
    val testData: JavaRDD[DataSet] = sc.parallelize(testDataList.toList)

    // Create network model configuration
    //

    val numRows = 28
    val numColumns = 28
    val outputNum = 10 // number of output classes
    val batchSize = 128 // batch size for each epoch

    experiment.logParameter("numRows", numRows)
    experiment.logParameter("numColumns", numColumns)
    experiment.logParameter("outputNum", outputNum)
    experiment.logParameter("batchSize", batchSize)
    experiment.logParameter("rngSeed", rngSeed)
    experiment.logParameter("numEpochs", numEpochs)

    val lr = 0.006
    val nesterovsMomentum = 0.9
    val l2Regularization = 1e-4
    val inputActivation = Activation.RELU
    val hiddenActivation = Activation.SOFTMAX
    val weightInit = WeightInit.XAVIER
    val optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
    val lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD

    experiment.logParameter("learningRate", lr)
    experiment.logParameter("nesterovsMomentum", nesterovsMomentum)
    experiment.logParameter("l2Regularization", l2Regularization)
    experiment.logParameter("inputActivation", inputActivation)
    experiment.logParameter("hiddenActivation", hiddenActivation)
    experiment.logParameter("weightInit", weightInit)
    experiment.logParameter("optimizationAlgorithm", optimizationAlgorithm)
    experiment.logParameter("lossFunction", lossFunction)

    MNistExampleSpark.log.info("Building model....")

    val conf = new NeuralNetConfiguration.Builder()
      //include a random seed for reproducibility
      .seed(rngSeed)
      .updater(new Nesterovs(lr, nesterovsMomentum))
      // use stochastic gradient descent as an optimization algorithm
      .optimizationAlgo(optimizationAlgorithm)
      .l2(l2Regularization)
      .list
      .layer(new DenseLayer.Builder()
        .nIn(numRows * numColumns)
        .nOut(1000)
        .activation(inputActivation)
        .weightInit(weightInit)
        .build)
      .layer(new OutputLayer.Builder(lossFunction)
        .nIn(1000)
        .nOut(outputNum)
        .activation(hiddenActivation)
        .weightInit(weightInit)
        .build)
      .build

    experiment.logGraph(conf.toJson)

    // Create Spark model
    //

    // Configuration for Spark training, see:
    // https://deeplearning4j.konduit.ai/spark/tutorials/dl4j-on-spark-quickstart#parameter-averaging-implementation
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) // Each DataSet object: contains (by default) 32 examples
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2) // Async prefetching: 2 examples per worker
      .batchSizePerWorker(batchSizePerWorker)
      .build

    logSparkTrainConfig(tm, experiment)

    val sparkModel = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkModel.setCollectTrainingStats(true)
    val listener = new ScoreIterationListener(10)
    sparkModel.setListeners(
      Collections.singletonList[TrainingListener](listener))

    MNistExampleSpark.log.info("Training model....")

    for (i <- 0 until numEpochs) {
      sparkModel.fit(trainData)
      experiment.logMetric("score", sparkModel.getScore, i)
      MNistExampleSpark.log.info("Completed Epoch {}", i)
    }

    MNistExampleSpark.log.info("Evaluating model....")

    val evaluation: Evaluation = sparkModel.evaluate(testData)
    experiment.logHtml(evaluation.getConfusionMatrix.toHTML, false)

    MNistExampleSpark.log.info(evaluation.stats)

    // Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

    MNistExampleSpark.log.info("**************** MNIST Experiment Example finished ********************")
    MNistExampleSpark.log.info("Experiment data published at: {}", experiment.getExperimentLink.get())
  }

  private def logSparkTrainConfig(tm: ParameterAveragingTrainingMaster, experiment: OnlineExperiment): Unit = {
    import java.io.PrintWriter
    val fileName = "/tmp/sprkTrainConfig.json"
    new PrintWriter(fileName) {
      write(tm.toJson)
      close()
    }
    experiment.uploadAsset(new File(fileName), false)
  }

  private def parseCommandLineArgs(args: Array[String]): Unit = {
    val jcmdr = JCommander.newBuilder.addObject(this).build
    try {
      jcmdr.parse(args: _*)
    } catch {
      case e: ParameterException =>
        jcmdr.usage()
        try {
          Thread.sleep(500)
        } catch {
          case _: Exception => ()
        }
        throw e
    }
  }
}