package ml.comet

import com.beust.jcommander.{JCommander, Parameter, ParameterException}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.{Model, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.mutable
import ml.comet.experiment.{OnlineExperimentImpl}

import java.io.File

/**
 * Train a simple, small MLP on MNIST data using Spark with Comet.
 */
object MNistExample {

  lazy val log = LoggerFactory.getLogger(classOf[MNistExample])
  val experiment = OnlineExperimentImpl.builder().build(); //update defaults.conf
  experiment.setInterceptStdout()

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    new MNistExample().entryPoint(args)
  }

}

class MNistExample {

  val experiment = MNistExample.experiment
  print("experiment live at: " + experiment.getExperimentLink)

  @Parameter(names = Array("-useSparkLocal"), description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
  private val useSparkLocal: Boolean = true

  @Parameter(names = Array("-batchSizePerWorker"), description = "Number of examples to fit each worker with")
  private val batchSizePerWorker: Int = 16

  @Parameter(names = Array("-numEpochs"), description = "Number of epochs for training")
  private val numEpochs: Int = 15

  experiment.logParameter("useSparkLocal", useSparkLocal);
  experiment.logParameter("batchSizePerWorker", batchSizePerWorker);
  experiment.logParameter("numEpochs", numEpochs);

  @throws[Exception]
  protected def entryPoint(args: Array[String]) {

    parseCommandLineArgs(args)

    val sparkConf = new SparkConf
    if (useSparkLocal) {
      sparkConf.setMaster("local[*]")
    }
    sparkConf.setAppName("DL4J Spark Example")
    sparkConf.getAll.foreach(x => experiment.logOther(String.valueOf(x._1), x._2))
    val sc = new JavaSparkContext(sparkConf)

    //Load the data into memory then parallelize
    //This isn't a good approach in general - but is simple to use for this example
    val iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345)
    val iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345)
    val trainDataList = mutable.ArrayBuffer.empty[DataSet]
    val testDataList = mutable.ArrayBuffer.empty[DataSet]
    while (iterTrain.hasNext) {
      trainDataList += iterTrain.next
    }

    while (iterTest.hasNext) {
      testDataList += iterTest.next
    }

    val trainData = sc.parallelize(trainDataList)
    val testData = sc.parallelize(testDataList)


    val learningRate = 0.02
    val momentum = 0.9
    val l2 = 1e-4
    val seed = 12345
    val optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
    val activation = Activation.LEAKYRELU
    val weightInit = WeightInit.XAVIER
    val updater = Updater.NESTEROVS

    experiment.logParameter("learningRate", learningRate)
    experiment.logParameter("l2", l2)
    experiment.logParameter("seed", seed)
    experiment.logParameter("optimizationAlgo", optimizationAlgo)
    experiment.logParameter("activation", activation)
    experiment.logParameter("weightInit", weightInit)
    experiment.logParameter("updater", updater)


    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(optimizationAlgo)
      .iterations(1)
      .activation(activation)
      .weightInit(weightInit)
      .learningRate(learningRate)
      .updater(updater)
      .momentum(momentum)
      .regularization(true)
      .l2(l2)
      .list
      .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build)
      .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build)
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX).nIn(100).nOut(10).build)
      .pretrain(false)
      .backprop(true)
      .build

    experiment.logGraph(conf.toJson);

    //Configuration for Spark training: see http://deeplearning4j.org/spark f
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) //Each DataSet object: contains (by default) 32 examples
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2) //Async prefetching: 2 examples per worker
      .batchSizePerWorker(batchSizePerWorker)
      .build

    logSparkTrainConfig(tm)

    val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNet.setCollectTrainingStats(true)

    for (i <- 0 until numEpochs) {
      sparkNet.fit(trainData)
      experiment.logMetric("score", sparkNet.getScore, i)
      MNistExample.log.info("Completed Epoch {}", i)
    }

    //Perform evaluation (distributed)
    val evaluation: Evaluation = sparkNet.evaluate(testData)
    experiment.logHtml(evaluation.getConfusionMatrix.toHTML, false)
    MNistExample.log.info("***** Evaluation *****")
    MNistExample.log.info(evaluation.stats)

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

    MNistExample.log.info("***** Example Complete *****")
    sys.exit(0)
  }

  private def logSparkTrainConfig(tm: ParameterAveragingTrainingMaster) = {
    import java.io.PrintWriter
    val fileName = "/tmp/sprkTrainConfig.json"
    new PrintWriter(fileName) {
      write(tm.toJson);
      close
    }
    experiment.uploadAsset(new File(fileName), false)
  }

  private def parseCommandLineArgs(args: Array[String]) = {
    val jcmdr = new JCommander(this, Array[String](): _*)
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