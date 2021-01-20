package ai.certifai.mockexam;

/* ===================================================================
 * We will solve a task of classifying vehicles.
 * The dataset contains 5 classes, each with approximately 48 to 60 images
 * Images are of 100x100 to 240x240 RGB
 * Put the dataset into your resource folder
 *
 * Source: https://www.kaggle.com/rishabkoul1/vechicle-dataset
 * ===================================================================
 * TO-DO
 *
 * 1. Get the file from resource
 * 2. Define image augmentation
 * 3. Put the defined augmentation into a pipeline
 * 4. Set up the dataset iterator
 * 5. Return the train iterator and test iterator from the instantiated setup
 * 6. Define configuration for model
 * 7. Define configuration for early stopping
 * 8. Perform some hyperparameter tuning
 *
 *
 * ====================================================================
 ** NOTE: Only make changes at sections specified. Do not shift any of the code.
*/

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class VehicleClassifier {

    //Do not change the parameters here
    private static int height = 100;
    private static int width = 100;
    private static int nClass = 5;
    private static int nChannels = 3;
    private static int seed = 1234;

    //Tunable parameters
    private static int batchSize = 12;
    private static double trainPerc = 0.8;
    private  static double lr = 1e-3;

    //16 marks in classifier, 7 marks in datasetiterator
    //10 marks for config, 7 marks for able to run

    public static void main(String[] args) throws IOException {

        /*
         * Step 1: Get the image dataset file
         *
         * */

        /*
         * Step 2: Define image augmentation
         * 2-1: Perform random cropping
         * 2-2: Perform flipping
         * 2-3: Perform rotation
         * */
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform verticalFlip = new FlipImageTransform(0);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage15 = new RotateImageTransform(15);
        ImageTransform showImage = new ShowImageTransform("Image",100);

        /*
         * Step 3: Put the defined augmentation into a pipeline
         * Set all probabilities to 0.3
         * Set shuffle to false
         * */
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.3),
                new Pair<>(verticalFlip, 0.3),
                new Pair<>(cropImage,0.3),
                new Pair<>(rotateImage15,0.3),
                new Pair<>(showImage,1.0));

        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        /*
         * Step 4: Setting up the DataSetIterator
         * 4-1: Instantiate an instance of the VehicleDataSetIterator class
         * 4-2: Complete the code in .setup() method
         * 4-3: Setup the DataSetIterator using the inputs provided
         *
         * */


        /*
         * Step 5: Return the train iterator and test iterator from the instantiated setup
         * 5-1: Complete the code in .makeIterator() method, .trainIterator() method and .testIterator() method
         * 5-2: Return the train iterator and test iterator
         * */
        VehicleDataSetIterator.setup(batchSize,trainPerc,transform);
        DataSetIterator trainIter = VehicleDataSetIterator.trainIterator(transform,batchSize);
        DataSetIterator testIter = VehicleDataSetIterator.testIterator(1);

        System.out.println("Number of train examples: " + VehicleDataSetIterator.train.length());
        System.out.println("Number of test examples: " + VehicleDataSetIterator.test.length());

        /*
         * Step 6: Define configuration for model
         * */
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(0.001))
                .list()
                .layer(0,new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).activation(Activation.RELU).nIn(nChannels).nOut(32).build())
                .layer(1,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2).padding(1,1).build())
                .layer(2, new DenseLayer.Builder().nOut(32).activation(Activation.RELU).build())
                .layer(3, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(nClass).build())
                .setInputType(InputType.convolutional(height,width,nChannels))
                .build();

        /*
         * Step 7: Define configuration for early stopping
         * 7-1: use ROC as score calculator and use AUC as the calculator metric
         * 7-2: terminate the training if no score improvement in 2 epochs
         * 7-3: terminate the training also if there is an invalid iteration score like NaNs
         * */
        EarlyStoppingConfiguration esConfig = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(2))
                .scoreCalculator(new DataSetLossCalculator(testIter,true))
                .evaluateEveryNEpochs(1)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig,config,trainIter);
        EarlyStoppingResult result = trainer.fit();

        MultiLayerNetwork model = (MultiLayerNetwork) result.getBestModel();
        model.init();

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);

        System.out.println("Train stats:\n"+evalTrain);
        System.out.println("Test stats:\n"+evalTest);


    }

}
