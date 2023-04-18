package pixelschubser;

import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import pixelschubser.exceptions.ImageObjetEmptyException;

import java.util.List;

import static pixelschubser.Homography.getHomographyMatrix;

public class ORB_Example {
    public void run() throws ImageObjetEmptyException {
        /**
         * Keine Sonderzeichen oder Leerzeichen im Namen!!
         */
//        Image a = new Image("IMG_0525.JPG", Imgcodecs.IMREAD_COLOR);
//        Image b = new Image("IMG_0526.JPG", Imgcodecs.IMREAD_COLOR);
//        Image b = new Image("unsplashMathiasKonrath.jpg", Imgcodecs.IMREAD_COLOR);
        Image2 a = new Image2("IMG_0525.JPG", Imgcodecs.IMREAD_GRAYSCALE);
        Image2 b = new Image2("IMG_0526.JPG", Imgcodecs.IMREAD_GRAYSCALE);
//        Image2 b = new Image2("unsplashMathiasKonrath.jpg", Imgcodecs.IMREAD_GRAYSCALE);

        /**
         * get objects for code below
         */
        Mat imgObject = a.getImgObject();
        Mat imgScene = b.getImgObject();
        MatOfKeyPoint keypointsObject = a.getMatOfKeyPoint();
        MatOfKeyPoint keypointsScene = b.getMatOfKeyPoint();
        Mat descriptorsObject = a.getKeypointDescriptionObject();
        Mat descriptorsScene = b.getKeypointDescriptionObject();

        List<DMatch> listOfGoodMatches = FeatureMatching.bfMatching(descriptorsObject, descriptorsScene);
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);

        /**
         * get homography
         */
        Mat homography = getHomographyMatrix(keypointsObject, keypointsScene, listOfGoodMatches);


        /**
         * Draw keypoints
         */
        Mat imgMatches = new Mat();
        Features2d.drawMatches(imgObject, keypointsObject, imgScene, keypointsScene, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        MatchDrawer.draw_matches(keypointsObject, keypointsScene, listOfGoodMatches, imgObject, imgScene, imgMatches);
        System.exit(0);
    }
}
