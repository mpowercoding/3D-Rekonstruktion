package pixelschubser.codesamples;

import org.opencv.core.*;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class MatchingKeypoints {
    public void run() {
        //Loading the OpenCV core library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        //Reading the source images

        String filename1 = getClass().getClassLoader().getResource("IMG_0525.JPG").getPath();
        String filename2 = getClass().getClassLoader().getResource("IMG_0526.JPG").getPath();

        String file1 = filename1;
        Mat src1 = Imgcodecs.imread(file1);
        String file2 = filename2;
        Mat src2 = Imgcodecs.imread(file2);
        //Creating an empty matrix to store the destination image
        Mat dst = new Mat();
        FastFeatureDetector detector = FastFeatureDetector.create();
        //Detecting the key points in both images
        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        detector.detect(src1, keyPoints1);
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        detector.detect(src2, keyPoints2);
        MatOfDMatch matof1to2 = new MatOfDMatch();
        Features2d.drawMatches(src1, keyPoints1, src2, keyPoints2, matof1to2, dst);

        Size sz = new Size(1920,1080);
        Imgproc.resize( dst, dst, sz );

        HighGui.imshow("Feature Matching", dst);
        HighGui.waitKey();
    }
}