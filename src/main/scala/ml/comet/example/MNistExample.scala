package ml.comet.example

import com.beust.jcommander.{JCommander, Parameter, ParameterException}
import ml.comet.experiment.{ExperimentBuilder, OnlineExperiment, OnlineExperimentImpl}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.{Model, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import scala.collection.mutable

/**
 * Train a simple, small MLP on MNIST data using Spark with Comet.
 */
object MNistExample {

  lazy val log: Logger = LoggerFactory.getLogger(classOf[MNistExample])
  val experiment: OnlineExperiment = ExperimentBuilder.OnlineExperiment.interceptStdout.build; //update application.conf
  experiment.setInterceptStdout()

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    new MNistExample().entryPoint(args)
  }

  case class StepScoreListener(experiment: OnlineExperiment, var printIterations: Int, log: Logger) extends BaseTrainingListener {
    override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
      if (printIterations <= 0) printIterations = 1
      // print score and log metric
      if (iteration % printIterations == 0) {
        val result = model.score
        log.info("Score at step/epoch {}/{}  is {} ", iteration, epoch, result)
        experiment.setEpoch(epoch)
        this.experiment.logMetric("score", model.score, iteration)
      }
    }
  }
}

class MNistExample {

  val experiment: OnlineExperiment = MNistExample.experiment
  print("experiment live at: " + experiment.getExperimentLink)

  @Parameter(names = Array("-useSparkLocal"), description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
  private val useSparkLocal: Boolean = true

  @Parameter(names = Array("-batchSizePerWorker"), description = "Number of examples to fit each worker with")
  private val batchSizePerWorker: Int = 16

  @Parameter(names = Array("-numEpochs"), description = "Number of epochs for training")
  private val numEpochs: Int = 2

  experiment.logParameter("useSparkLocal", useSparkLocal);
  experiment.logParameter("batchSizePerWorker", batchSizePerWorker);
  experiment.logParameter("numEpochs", numEpochs);

  @throws[Exception]
  protected def entryPoint(args: Array[String]): Unit = {

    parseCommandLineArgs(args)


    val numRows = 28
    val numColumns = 28
    val outputNum = 10 // number of output classes
    val batchSize = 128 // batch size for each epoch
    val rngSeed = 123 // random number seed for reproducibility

    experiment.logParameter("numRows", numRows)
    experiment.logParameter("numColumns", numColumns)
    experiment.logParameter("outputNum", outputNum)
    experiment.logParameter("batchSize", batchSize)
    experiment.logParameter("rngSeed", rngSeed)
    experiment.logParameter("numEpochs", numEpochs)

    val lr = 0.006
    val nesterovsMomentum = 0.9
    val l2Regularization = 1e-4
    val seed = 12345
    val inputActivation = Activation.RELU
    val hiddenActivation = Activation.SOFTMAX
    val weightInit = WeightInit.XAVIER
    val optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
    val lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD

    experiment.logParameter("learningRate", lr)
    experiment.logParameter("nesterovsMomentum", nesterovsMomentum)
    experiment.logParameter("l2Regularization", l2Regularization)
    experiment.logParameter("seed", seed)
    experiment.logParameter("inputActivation", inputActivation)
    experiment.logParameter("hiddenActivation", hiddenActivation)
    experiment.logParameter("weightInit", weightInit)
    experiment.logParameter("optimizationAlgorithm", optimizationAlgorithm)
    experiment.logParameter("lossFunction", lossFunction)


    MNistExample.log.info("Build model....")
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

    experiment.logGraph(conf.toJson);

    val model = new MultiLayerNetwork(conf)
    model.init()
    //print the score with every 1 iteration
    model.setListeners(MNistExample.StepScoreListener(experiment, 1, MNistExample.log))

    // Get the train dataset iterator
    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)

    MNistExample.log.info("Train model....")
    model.fit(mnistTrain, numEpochs)

    // Get the test dataset iterator
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    MNistExample.log.info("Evaluate model....")
    val eval = model.evaluate(mnistTest)
    MNistExample.log.info(eval.stats)

    experiment.logHtml(eval.getConfusionMatrix.toHTML, false)

    MNistExample.log.info("****************MNIST Experiment Example finished********************")


    val sparkConf = new SparkConf
    if (useSparkLocal) {
      sparkConf.setMaster("local[*]")
    }
    sparkConf.setAppName("DL4J Spark Example")
    sparkConf.getAll.foreach(x => experiment.logOther(String.valueOf(x._1), x._2))
    val sc = new JavaSparkContext(sparkConf)

    //Load the data into memory then parallelize
    //This isn't a good approach in general - but is simple to use for this example
    val iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, seed)
    val iterTest = new MnistDataSetIterator(batchSizePerWorker, false, seed)
    val trainDataList = mutable.ArrayBuffer.empty[DataSet]
    val testDataList = mutable.ArrayBuffer.empty[DataSet]
    while (iterTrain.hasNext) {
      trainDataList += iterTrain.next
    }

    while (iterTest.hasNext) {
      testDataList += iterTest.next
    }

    val trainData: JavaRDD[DataSet] = sc.parallelize(trainDataList)
    val testData: JavaRDD[DataSet] = sc.parallelize(testDataList)


    // Configuration for Spark training: see http://deeplearning4j.org/spark f
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) //Each DataSet object: contains (by default) 32 examples
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2) //Async prefetching: 2 examples per worker
      .batchSizePerWorker(batchSizePerWorker)
      .build

    logSparkTrainConfig(tm)

    val sparkModel = new SparkDl4jMultiLayer(sc, model, tm)
    sparkModel.setCollectTrainingStats(true)

    sparkModel.fit(trainData)

    //Perform evaluation (distributed)
    val evaluation: Evaluation = sparkModel.evaluate(testData)
    experiment.logHtml(evaluation.getConfusionMatrix.toHTML, false)
    MNistExample.log.info("***** Evaluation *****")
    MNistExample.log.info(evaluation.stats)

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

    MNistExample.log.info("****************MNIST Experiment Example finished********************")

    sys.exit(0)
  }

  private def logSparkTrainConfig(tm: ParameterAveragingTrainingMaster): Unit = {
    import java.io.PrintWriter
    val fileName = "/tmp/sprkTrainConfig.json"
    new PrintWriter(fileName) {
      write(tm.toJson);
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
          case e2: Exception => ()
        }
        throw e
    }
  }
}