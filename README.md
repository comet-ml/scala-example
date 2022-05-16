# Comet Java SDK Scala Example
Collection of examples demonstrating how to use [Comet Java SDK](https://github.com/comet-ml/comet-java-sdk) 
with [Apache Spark](https://spark.apache.org) using [Deeplearning4j on Spark](https://deeplearning4j.konduit.ai/spark/tutorials/dl4j-on-spark-quickstart)
and [Scala](https://scala-lang.org) programming language.

# System Requirements

The Scala experiments defined in this repository use **Scala 2.11.12** and intended to be run
using **Java 8**.

You need to have **Java 8** installed and configured as default on your system in order to
run Scala examples from the command line as described later. If you are using IDE, make 
sure to select Java 8 SDK for building and running.

If you have higher Java version in your system, please make sure to change Scala version
defined in the project's [POM](./pom.xml) to meet your JDK version. For version compatibility please
refer to [Scala version compatibility table](https://docs.scala-lang.org/overviews/jdk-compatibility/overview.html).

# Configuration and Setup

Please make sure that you have account at [Comet.ml](https://www.comet.ml) and got your Comet API key.
For more details please refer to [Quick Start Guide](https://www.comet.ml/docs/quick-start/).

## Providing configuration for Comet experiment

There are a few ways to properly configure your Comet experiment:

* Using environment variables
* Using configuration file
* Using Experiment builder API

Next, we describe all of them.

### Environment variables

The configuration can be done using the following environment variables that you should set
before running the Comet experiment:

* **COMET_API_KEY** - the Comet API key you got from your Comet cabinet
* **COMET_WORKSPACE_NAME** - the name of workspace where your experiments will be saved at Comet.ml
* **COMET_PROJECT_NAME** - the name of project within workspace for your experiments

You can set mentioned environment variables via command line when running experiment. 
We discuss this method later.

### Configuration file

Additionally, you can use experiment configuration file to provide necessary configuration options.
This can be useful if you plan to run your experiment multiple times. The configuration file you can
find at [application.conf](./src/main/resources/application.conf).

It has the following records which you should update by uncommenting and providing valid values:

```text
comet {
    # baseUrl = "https://www.comet.ml"
    # apiKey = "XXXX"
    # project = "java-sdk"
    # workspace = "my-team"
}
```

For extra options please refer to [CometConfig.java](https://github.com/comet-ml/comet-java-sdk/blob/master/comet-java-client/src/main/java/ml/comet/experiment/impl/config/CometConfig.java)

### Using Experiment builder API

Also, it is possible to configure Comet Experiment using experiment builder API.
The following code snippet demonstrate how to do this:

```text
val experiment = ExperimentBuilder.OnlineExperiment
      .interceptStdout
      .withWorkspace("my-team")
      .withProjectName("java-sdk")
      .withApiKey("XXXX")
      .build
```

or 

```text
val experiment = ExperimentBuilder.OnlineExperiment
      .interceptStdout
      .withApiKey("XXXX").asInstanceOf[OnlineExperimentBuilder] // need to have explicit type casting here
      .withWorkspace("my-team")
      .withProjectName("java-sdk")
      .build
```

For extra configuration options please refer to [OnlineExperimentBuilder](https://github.com/comet-ml/comet-java-sdk/blob/master/comet-java-client/src/main/java/ml/comet/experiment/builder/OnlineExperimentBuilder.java).

# Running Comet Experiment example

We provide two examples of running the classic MNIST experiment:

* Using Deeplearning4j on Spark and Scala
* Using Deeplearning4j and Java

Next, we discuss how to ru each of them.

## Scala with Deeplearning4j on Spark

The source code of this experiment available at [MNistExampleSpark](./src/main/scala/ml/comet/examples/MNistExampleSpark.scala)

You can run the experiment with the following command using your favourite terminal application.

```text
COMET_API_KEY=your_api_key \
COMET_WORKSPACE_NAME=your_workspace \
COMET_PROJECT_NAME=your_project_name \
mvn package exec:java -Dexec.mainClass="ml.comet.examples.MNistExampleSpark" -Dexec.args="--numEpochs 2"
```

Where:
* **numEpochs** - allows you to define number of training epochs [**default: 10**]
* **useSparkLocal** - allows to use spark local (helper for testing/running without spark submit) [**default: true**]

Make sure to provide correct values of the environment variables.

## Java with Deeplearning4j

As reference, we also provide Java implementation of the MNIST experiment with source code available 
at [MNistExampleJava](./src/main/scala/ml/comet/examples/MNistExampleJava.java)

You can run the experiment with the following command.

```text
COMET_API_KEY=your_api_key \
COMET_WORKSPACE_NAME=your_workspace \
COMET_PROJECT_NAME=your_project_name \
mvn exec:java -Dexec.mainClass="ml.comet.examples.MNistExampleJava" -Dexec.args="--epochs 2"
```

Where:
* **epochs** - allows you to define the number of training epochs [default: 10]

Make sure to provide correct values of the environment variables.

# References

1. [Comet Java SDK](https://github.com/comet-ml/comet-java-sdk)
2. [Deeplearning4j on Spark](https://deeplearning4j.konduit.ai/spark/tutorials/dl4j-on-spark-quickstart)
3. [Apache Spark](https://spark.apache.org)
4. [Comet Quick Start Guide](https://www.comet.ml/docs/quick-start/)