package ml.comet.examples;

import ml.comet.experiment.OnlineExperiment;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;

/**
 * The listener to be invoked at each iteration of model training.
 */
public class ExperimentScoreIterationListener extends BaseTrainingListener {
    private final OnlineExperiment experiment;
    private int printIterations;
    private final Logger log;

    ExperimentScoreIterationListener(OnlineExperiment experiment, int printIterations, Logger log) {
        this.experiment = experiment;
        this.printIterations = printIterations;
        this.log = log;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (printIterations <= 0) {
            printIterations = 1;
        }
        // print score and log metric
        if (iteration % printIterations == 0) {
            double result = model.score();
            log.info("Score at step/epoch {}/{}  is {} ", iteration, epoch, result);
            experiment.setEpoch(epoch);
            experiment.logMetric("score", model.score(), iteration);
        }
    }
}
