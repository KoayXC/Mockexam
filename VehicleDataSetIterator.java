package ai.certifai.mockexam;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class VehicleDataSetIterator {

    private static int imgHeight;
    private static int imgWidth;
    private static int imgChannels;
    private static int nClass;
    private static int batchsize;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static PathFilter pathFilter = new BalancedPathFilter(new Random(1234), BaseImageLoader.ALLOWED_FORMATS,labelMaker);
    static InputSplit train,test;


    private static ImageTransform imgTransform;

    public VehicleDataSetIterator() {

    }

    public static void setup(int batchSize, double trainPerc, ImageTransform transform) {
    }

    public void setup(File file, int height, int width, int channels, int numClass, ImageTransform imageTransform,
                      int batchSize, double trainPerc) throws IOException {

        imgHeight = height;
        imgWidth = width;
        imgChannels = channels;
        nClass = numClass;
        imgTransform = imageTransform;
        batchsize = batchSize;

        /*
         * Step 4-2: Get the file and split it into train and test dataset
         * */
        File input = new ClassPathResource("vehicle_with_train").getFile();

        FileSplit fileSplit = new FileSplit(input, BaseImageLoader.ALLOWED_FORMATS);
        PathFilter balanceFilter = new BalancedPathFilter(new Random(1234),BaseImageLoader.ALLOWED_FORMATS,new ParentPathLabelGenerator());

        InputSplit[] sample = fileSplit.sample(balanceFilter, 90,10);

        train = sample[0];
        test = sample[1];
    }

    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {

        ImageRecordReader imgRR = new ImageRecordReader(imgHeight, imgWidth, imgChannels, labelMaker);

        if (training && imgTransform!=null){
            imgRR.initialize(split,imgTransform);
        } else {
            imgRR.initialize(split);
        }

        /*
         * Step 5-1: Read the record reader and define a feature scaling method
         * Keep in mind that there is only 1 output
         * */
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imgRR, batchsize, 1,nClass);

        return dataSetIterator;
    }


    public static DataSetIterator trainIterator(ImageTransform transform, int batchSize) throws IOException {
        /*
         * Step 5-1: Return trainIterator
         * */
        return makeIterator(train,true);
    }

    public static DataSetIterator testIterator(int i) throws IOException {
        /*
         * Step 5-1: Return testIterator
         * */
        return makeIterator(test,false);
    }
}
