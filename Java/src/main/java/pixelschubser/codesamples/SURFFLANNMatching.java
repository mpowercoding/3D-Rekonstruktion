package pixelschubser.codesamples;

import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
//import org.opencv.xfeatures2d.SURF;

import java.util.ArrayList;
import java.util.List;

class SURFFLANNMatching {
    public void run(String[] args) {
//        String filename1 = args.length > 1 ? args[0] : "../../resources/IMG_0525.JPG";
//        String filename2 = args.length > 1 ? args[1] : "../../resources/IMG_0526.JPG";
//        image = ImageIO.read(getClass().getClassLoader().getResource("resources/icon.gif"));
        String filename1 = getClass().getClassLoader().getResource("IMG_0525.JPG").getPath();
        String filename2 = getClass().getClassLoader().getResource("IMG_0526.JPG").getPath();
        Mat img1 = Imgcodecs.imread(filename1, Imgcodecs.IMREAD_GRAYSCALE);
        Mat img2 = Imgcodecs.imread(filename2, Imgcodecs.IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            System.err.println("Cannot read images!");
            System.exit(0);
        }
        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
//        double hessianThreshold = 400;
//        int nOctaves = 4, nOctaveLayers = 3;
//        boolean extended = false, upright = false;
//        SURF detector = SURF.create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

        int hessianThreshold = 400;
        int nOctaves = 4, nOctaveLayers = 3;
        boolean extended = false, upright = false;
        double edgeThreshold = 2.0;
        double sigma = 1.0;
        SIFT detector = SIFT.create(hessianThreshold, nOctaveLayers, edgeThreshold, sigma);


        MatOfKeyPoint keypoints1 = new MatOfKeyPoint(), keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();
        detector.detectAndCompute(img1, new Mat(), keypoints1, descriptors1);
        detector.detectAndCompute(img2, new Mat(), keypoints2, descriptors2);
        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
        //-- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.7f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);
        //-- Draw matches
        Mat imgMatches = new Mat();
//        Features2d.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1), Scalar.all(-1), new MatOfByte(), Features2d.NOT_DRAW_SINGLE_POINTS);
//        Features2d.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1), Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);
        Features2d.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1), Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
        //-- Show detected matches

        Size sz = new Size(1920, 1080);
        Imgproc.resize(imgMatches, imgMatches, sz);

        HighGui.imshow("Good Matches", imgMatches);
        HighGui.waitKey(0);
        System.exit(0);
    }
}
