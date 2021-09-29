package ml.comet;

import ml.comet.experiment.OnlineExperiment;
import ml.comet.experiment.OnlineExperimentImpl;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * https://github.com/deeplearning4j/oreilly-book-dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java
 * A Simple Multi Layered Perceptron (MLP) applied to digit classification for
 * the MNIST Dataset (http://yann.lecun.com/exdb/mnist/).
 * <p>
 * This file builds one input layer and one hidden layer.
 * <p>
 * The input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer will have 1000 output signals to the hidden layer.
 * <p>
 * The hidden layer has input dimensions of 1000. These are fed from the input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.
 */
public class MNistExampleJava {

    private static Logger log = LoggerFactory.getLogger(MNistExampleJava.class);

    public static void main(String[] args) throws Exception {
        OnlineExperiment experiment = OnlineExperimentImpl.builder().build(); //update defaults.conf
        experiment.setInterceptStdout();

        System.out.println("experiment live at: " + experiment.getExperimentLink());

        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 15; // number of epochs to perform

        experiment.logParameter("numRows", numRows);
        experiment.logParameter("numColumns", numColumns);
        experiment.logParameter("outputNum", outputNum);
        experiment.logParameter("batchSize", batchSize);
        experiment.logParameter("rngSeed", rngSeed);
        experiment.logParameter("numEpochs", numEpochs);

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        Double lr = 0.006;
        experiment.logParameter("lr", lr);

        OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        experiment.logParameter("optimizationAlgorithm", OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .optimizationAlgo(optimizationAlgorithm)
                .iterations(1)
                .learningRate(lr) //specify the learning rate
                .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();

        experiment.logGraph(conf.toJson());

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
            experiment.logMetric("score", model.score());
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        experiment.logHtml(eval.getConfusionMatrix().toHTML(), false);

        log.info("****************Example finished********************");
    }

}
