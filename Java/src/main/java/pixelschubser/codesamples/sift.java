package pixelschubser.codesamples;

import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.bytedeco.javacpp.opencv_features2d.drawKeypoints;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class sift {
    public void run2() {
        int nFeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.03;
        double edgeThreshold = 10;
        double sigma = 1.6;
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();

        String filename1 = getClass().getClassLoader().getResource("IMG_0525.JPG").getPath();
        Mat image = Imgcodecs.imread(filename1, Imgcodecs.IMREAD_GRAYSCALE);


        SIFT sift = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        // public void detect(Mat image, MatOfKeyPoint keypoints, Mat mask)
        sift.detect(image, keypoints1, descriptors1);
        List<KeyPoint> punkte = keypoints1.toList();

        // draw keypoints on image
        Mat outputImage = new Mat();
        // Your image, keypoints, and output image
        Features2d.drawKeypoints(image, keypoints1, outputImage);
//        String filename = "keypoints.jpg";
//        System.out.println(String.format("Writing %s...", filename));
//        imwrite(filename, outputImage);

        Size sz = new Size(2000, 1500);
        Imgproc.resize(outputImage, outputImage, sz);

        HighGui.imshow("Good Matches", outputImage);
        HighGui.waitKey(0);
        System.exit(0);
    }
}
