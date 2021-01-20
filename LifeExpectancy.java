//package ai.certifai.mockexam;
//
///*
// * Before you start, you should:
// * 1. Put the CSV file into your resources folder
//
// * Here, the hints of what you should do is given below:
//
// * Step 1: Read the CSV file in a RecordReader.
// * Step 2: Specify the transform process.
// * Step 3: Put each instances into the List<List<Writable>>.
// * Step 4: Execute the transform process.
// * Step 5: Perform shuffling of dataset with seed number provided.
// * Step 6: Split the dataset using train fraction provided and assign the large fraction as trainAndVal
// *         while the another as test split.
// * Step 7: Define feature scaling. Try to experiment with both feature scaling methods.
// * Step 8: Perform scaling only on trainAndVal
// * Step 9: Perform K-fold cross validation. Take k=5.
// * Step 10: Write the model config in the static method and return it in the main method.
// * Step 11: Perform training
// * Step 12: Perform evaluation on trainAndVal and test set
//
// */
//
//
//import org.datavec.api.transform.TransformProcess;
//import org.datavec.api.transform.filter.FilterInvalidValues;
//import org.datavec.api.transform.schema.Schema;
//import org.datavec.api.writable.Writable;
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.deeplearning4j.ui.api.UIServer;
//import org.deeplearning4j.ui.stats.StatsListener;
//import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
//import org.nd4j.linalg.dataset.ViewIterator;
//
//public class LifeExpectancy {
//
//    private static int seed = 12345;
//    private static double lr = 0.001;
//    private static int epoch = 10;
//    private static double trainPerc = 0.7;
//
//    public static void main(String[] args) {
//
//        /*
//         * Step 1: Read the CSV file in a RecordReader
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        //define schema
//        Schema schema = new Schema.Builder()
//                .addColumnsDouble("country")
//                .addColumnInteger("year")
//                .addColumnString("status")
//                .addColumnDouble("life-exp")
//                .addColumnsInteger("adult-mortality","infant-death")
//                .addColumnsDouble("alcohol","perc-expenditure")
//                .addColumnsInteger("hepa-b","measles")
//                .addColumnDouble("bmi")
//                .addColumnsInteger("under-5-deaths","polio")
//                .addColumnDouble("total-expenditure")
//                .addColumnInteger("diphtheria")
//                .addColumnsDouble("hiv-aids","gdp")
//                .addColumnInteger("population")
//                .addColumnsDouble("thinness-1_19","thinness-5_9","income-composition","schooling")
//                .build();
//
//        /*
//         * Step 2: Specify the transform process
//         * 2-1: convert "status" into categorical - "developing" and "developed"
//         * 2-2: convert "status" to one-hot
//         *
//         */
//        TransformProcess tp = new TransformProcess.Builder(schema)
//                .removeColumns("country")
//                //====== Your code block starts here===========
//
//                //====== Your code block ends here===========
//                .filter(new FilterInvalidValues())
//                .build();
//
//        List<List<Writable>> original = new ArrayList<>();
//
//        /*
//         * Step 3: Put each instances into the List<List<Writable>>
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 4: Execute transform process
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        System.out.println("Schema before transformed: \n"+tp.getInitialSchema());
//        System.out.println("Schema after transformed: \n"+tp.getFinalSchema());
//
//        System.out.println("Size before transformed: "+original.size());
//        System.out.println("Size after transformed: "+transformed.size());
//        System.out.println("Columns before transformed: "+tp.getInitialSchema().numColumns());
//        System.out.println("Columns after transformed: "+tp.getFinalSchema().numColumns());
//
//        CollectionRecordReader crr = new CollectionRecordReader(transformed);
//
//        /*
//         * Step 5: Perform shuffling of dataset with seed number provided
//         * Keep in mind that there is only 1 output
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 6: Split the dataset using train fraction provided and
//         *         assign the large fraction as trainAndVal while the another as test split
//         * Keep in mind that there is only 1 output
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 7: Define feature scaling
//         * hint: experiment with both types of scaling method
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        //perform scaling only on train and validation set
//        /*
//         * Step 8: Perform scaling only on trainAndVal
//         *
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        ViewIterator testIter = new ViewIterator(test,test.numExamples());
//        testIter.setPreProcessor(scaler);
//
//        /*
//         * Step 9: Perform K-fold cross validation
//         * Take K=5
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 10: Write the model config in the static method and return it in the main method
//         * Keep in mind to use the following input parameters provided: seed and learning rate
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        MultiLayerNetwork model = new MultiLayerNetwork(config);
//
//        InMemoryStatsStorage storage = new InMemoryStatsStorage();
//        UIServer server = UIServer.getInstance();
//        server.attach(storage);
//        model.setListeners(new StatsListener(storage,1),new ScoreIterationListener(10));
//
//        /*
//         * Step 11: Perform training
//         * */
//        model.init();
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//        System.out.println(model.summary());
//
//        /*
//         * Step 12: Perform evaluation on trainAndVal and test set
//         * */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//        System.out.println("Train & Validation evaluation\n"+evalTrain.stats());
//        System.out.println("Test evaluation\n"+evalTest.stats());
//
//    }
//
//    private static MultiLayerConfiguration getConfig(int seedNum, double learningRate, int nFeatures, int nClass){
//
//        /*
//          Step 10-1: write the config using the input variables and complete the static method
//          Specifications:
//          1. use Adam
//          2. use Xavier
//          3. use 1 hidden layer with 100 neurons
//          4. use MSE as loss function
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//    }
//
//
//}