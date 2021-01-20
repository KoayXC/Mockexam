package ai.certifai.mockexam;

import javafx.util.Pair;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/* ===================================================================
 * We will solve a task of classifying horse breeds.
 * The dataset contains 4 classes, each with just over 100 images
 * Images are of 256x256 RGB
 *
 * Source: https://www.kaggle.com/olgabelitskaya/horse-breeds
 * ===================================================================
 * TO-DO
 *
 * 1. In HorseBreedIterator complete both methods (i) setup and (ii) makeIterator
 * 2. Complete ImageTransform Pipeline
 * 3. Complete your network configuration
 * 4. Train your model and set listeners
 * 5. Perform evaluation on both train and test set
 * 6. [OPTIONAL] Mitigate the underfitting problem
 *
 * ====================================================================
 * Assessment will be based on
 *
 * 1. Correct and complete configuration details
 * 2. HorseBreedClassifier is executable
 * 3. Convergence of the network
 *
 * ====================================================================
 ** NOTE: Only make changes at sections with the following. Replace accordingly.
 *
 *   /*
 *    *
 *    * WRITE YOUR CODES HERE
 *    *
 *    *
 */

public class HorseBreedClassifier {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(HorseBreedClassifier.class);
    private static final int height = 64;
    private static final int width = 64;
    private static final int nChannel = 3;
    private static final int nOutput = 4;
    private static final int seed = 141;
    private static Random rng = new Random(seed);
    private static double lr = 1e-4;
    private static final int nEpoch = 20;
    private static final int batchSize = 3;

    public static void main(String[] args) throws IOException {

        HorseBreedIterator.setup();

        // Build an Image Transform pipeline consisting of
        // a horizontal flip, crop, rotation, and random cropping

        FlipImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(5);
        ImageTransform rotate30 = new RotateImageTransform(30);
        ImageTransform randomCrop = new RandomCropTransform(seed,height,width);
        boolean shuffle = false;

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,1.0),
                new Pair<>(cropImage,1.0),
                new Pair<>(rotate30,1.0),
                new Pair<>(randomCrop,1.0)
        );

        ImageTransform transform = new PipelineImageTransform();

        DataSetIterator trainIter = HorseBreedIterator.getTrain(transform,batchSize);
        DataSetIterator testIter = HorseBreedIterator.getTest(1);

        // Build your model configuration

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nIn(nChannel)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(1, new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build()
                )
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(nOutput)
                        .build()
                )
                .setInputType(InputType.convolutional(height,width,nChannel))

                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("**************************************** MODEL SUMMARY ****************************************");
        System.out.println(model.summary());

        // Train your model and set listeners

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

        for (int i=0; i<100; i++){
            trainIter.reset();
            model.fit(trainIter);
        }
        log.info("**************************************** MODEL EVALUATION ****************************************");

        // Perform evaluation on both train and test set

        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

    }

}