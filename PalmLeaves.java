//package ai.certifai.mockexam;
//
///* ===================================================================
// * We will solve a task of classifying palm leaves' conditions.
// * The dataset contains 3 classes (brown spots, healthy and white scale), each with varying number of samples
// * Images are either 350x192 or 192x350, and all are coloured
// *
// * Source: https://www.kaggle.com/hadjerhamaidi/date-palm-data
// * ===================================================================
// * TO-DO
// *
// * 1. Complete the flow of building a dataset iterator
// * 2. Perform feature scaling
// * 3. Complete model configurations according to specifications
// * 4. Train your model
// * 5. Perform model evaluation
// * 6. Save your model
// *
// *
// * ====================================================================
// * Assessment will be based on
// *
// * 1. Correct and complete codes
// * 2. Executable program
// * 3. Convergence of network
// * TOTAL MARKS = 30
// * ====================================================================
// ** NOTE: Only make changes at sections with the following. Replace accordingly.
// *
// *   /*
// *    *
// *    * WRITE YOUR CODES HERE
// *    *
// *    *
// *    */
//
//import org.datavec.api.io.labels.ParentPathLabelGenerator;
//import org.datavec.image.loader.BaseImageLoader;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.nd4j.linalg.io.ClassPathResource;
//
//import java.io.File;
//import java.util.Random;
//
//public class PalmLeaves {
//
//    private static int height = 80;
//    private static int width = 80;
//    private static int channel = 3;
//    private static int nClasses = 3;
//    private static int nEpoch = 5;
//    private static int batchSize = 32;
//    private static double learningRate = 1e-2;
//    private static final String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
//    private static final ParentPathLabelGenerator myLabels = new ParentPathLabelGenerator();
//    private static final int seed = 122;
//    private static final int trainFraction = 70; //Percentage of train split
//    private static final Random rng = new Random(seed);
//
//    //Note: All variables must be used. Variables with keyword final cannot be changed.
//    //      You are free to make changes to non-final variables to your liking.
//
//    public static void main(String[] args) {
//
//        // 1. Parse your data and split into train and test set by completing the following
//
//        File myFile = new ClassPathResource("PalmLeaves").getFile();
//
//        /*
//         *
//         * WRITE YOUR CODES HERE
//         *
//         *
//         */
//
//        ImageRecordReader trainRR = new ImageRecordReader(height,width,channel,myLabels);
//        ImageRecordReader testRR = new ImageRecordReader(height,width,channel,myLabels);
//
//        // 2. Initialize your record reader, create your iterator and perform feature scaling
//
//        /*
//         *
//         * WRITE YOUR CODES HERE
//         *
//         *
//         */
//
//
//        /* 3. Build your model configuration which must include:
//         * - Use Nesterovs updater with an Exponential Learning Rate schedule with Gamma = 0.75, changing by Epoch
//         * - At least 3 Convolution Layers
//         * - At least 2 Pooling Layers
//         * - At least 2 fully connected layers
//         */
//
//        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
//                .list()
//                .build();
//
//        // 4. Initialize your model by completing the following
//
//        MultiLayerNetwork model = /*WRITE YOUR CODE HERE*/;
//        /*
//         *
//         * WRITE YOUR CODES HERE
//         *
//         *
//         */
//
//        System.out.println(model.summary());
//
//        // 5. Do model training and set score listeners
//
//        /*
//         *
//         * WRITE YOUR CODES HERE
//         *
//         *
//         */
//
//        // 6.  Complete the codes to evaluate your model
//
//        /*
//         *
//         * WRITE YOUR CODES HERE
//         *
//         *
//         */
//
//        System.out.println("Train Evaluation: "+ );
//        System.out.println("Test Evaluation: "+ );
//
//        // 7. Save your model to a new file called MockExamModel.zip
//        //    which is to be located in your working directory in the generated-models folder
//
//        /*
//         *
//         * WRITE YOUR CODES HERE
//         *
//         *
//         */
//
//    }
//}