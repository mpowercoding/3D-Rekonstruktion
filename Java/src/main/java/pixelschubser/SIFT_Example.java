package pixelschubser;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import pixelschubser.exceptions.ImageObjetEmptyException;

import java.util.ArrayList;
import java.util.List;

public class SIFT_Example {
    public void run() throws ImageObjetEmptyException {
        /**
         * Keine Sonderzeichen oder Leerzeichen im Namen!!
         */
//        Image a = new Image("IMG_0525.JPG", Imgcodecs.IMREAD_COLOR);
//        Image b = new Image("IMG_0526.JPG", Imgcodecs.IMREAD_COLOR);
//        Image b = new Image("unsplashMathiasKonrath.jpg", Imgcodecs.IMREAD_COLOR);
//        Image a = new Image("IMG_0525.JPG", Imgcodecs.IMREAD_GRAYSCALE);
        Image a = new Image("Milch/IMG_0675.JPG", Imgcodecs.IMREAD_GRAYSCALE);
//        Image b = new Image("IMG_0526.JPG", Imgcodecs.IMREAD_GRAYSCALE);
        Image b = new Image("Milch/IMG_0677.JPG", Imgcodecs.IMREAD_GRAYSCALE);
//        Image b = new Image("unsplashMathiasKonrath.jpg", Imgcodecs.IMREAD_GRAYSCALE);

        /**
         * get objects for code below
         */
        Mat imgObject = a.getImgObject();
        Mat imgScene = b.getImgObject();
        MatOfKeyPoint keypointsObject = a.getMatOfKeyPoint();
        MatOfKeyPoint keypointsScene = b.getMatOfKeyPoint();
        Mat descriptorsObject = a.getKeypointDescriptionObject();
        Mat descriptorsScene = b.getKeypointDescriptionObject();

        List<DMatch> listOfGoodMatches = FeatureMatching.flannMatching(descriptorsObject, descriptorsScene);
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);

        /**
         * custom filter
         */
        //-- Localize the object
        List<Point> obj = new ArrayList<>();
        List<Point> scene = new ArrayList<>();
        List<KeyPoint> listOfKeypointsObject = keypointsObject.toList();
        List<KeyPoint> listOfKeypointsScene = keypointsScene.toList();

        if (false) {
            listOfGoodMatches = custom_filter_matches(keypointsObject, keypointsScene, listOfGoodMatches, 0.3);
        }

        /**
         * Draw keypoints
         */
        Mat imgMatches = new Mat();
        Features2d.drawMatches(imgObject, keypointsObject, imgScene, keypointsScene, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        /**
         * Draw matches
         */
        for (int i = 0; i < listOfGoodMatches.size(); i++) {
            obj.add(listOfKeypointsObject.get(listOfGoodMatches.get(i).queryIdx).pt);
            scene.add(listOfKeypointsScene.get(listOfGoodMatches.get(i).trainIdx).pt);
        }
        MatOfPoint2f objMat = new MatOfPoint2f(), sceneMat = new MatOfPoint2f();
        objMat.fromList(obj);
        sceneMat.fromList(scene);
        double ransacReprojThreshold = 3.0;
        Mat H = Calib3d.findHomography(objMat, sceneMat, Calib3d.RANSAC, ransacReprojThreshold);
        for(int i = 0; i < H.cols(); i++) {
            for(int j = 0; j < H.rows(); j++) {
                System.out.print(H.get(j, i) + " - ");
            }
            System.out.println();
        }


        //-- Get the corners from the image_1 ( the object to be "detected" )
        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2), sceneCorners = new Mat();
        float[] objCornersData = new float[(int) (objCorners.total() * objCorners.channels())];
        objCorners.get(0, 0, objCornersData);
        objCornersData[0] = 0;
        objCornersData[1] = 0;
        objCornersData[2] = imgObject.cols();
        objCornersData[3] = 0;
        objCornersData[4] = imgObject.cols();
        objCornersData[5] = imgObject.rows();
        objCornersData[6] = 0;
        objCornersData[7] = imgObject.rows();
        objCorners.put(0, 0, objCornersData);
        Core.perspectiveTransform(objCorners, sceneCorners, H);
        float[] sceneCornersData = new float[(int) (sceneCorners.total() * sceneCorners.channels())];
        sceneCorners.get(0, 0, sceneCornersData);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        Imgproc.line(imgMatches, new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]),
                new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]),
                new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]),
                new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]),
                new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]), new Scalar(0, 255, 0), 4);
        //-- Show detected matches

        Size sz = new Size(2000, 1500);
        Imgproc.resize(imgMatches, imgMatches, sz);

        HighGui.imshow("Good Matches & Object detection", imgMatches);
        HighGui.waitKey(0);
        System.exit(0);
    }

    private List<DMatch> custom_filter_matches(MatOfKeyPoint keypointsObject, MatOfKeyPoint keypointsScene, List<DMatch> listOfGoodMatches, double tolerance) {
        listOfGoodMatches = this.calculate_mean_line(listOfGoodMatches, keypointsObject.toList(), keypointsScene.toList(), tolerance);

        System.out.println("Step 3 matches: " + listOfGoodMatches.size());
        return listOfGoodMatches;
    }

    public List<DMatch> calculate_mean_line(List<DMatch> listOfGoodMatches, List<KeyPoint> listOfKeypointsObject, List<KeyPoint> listOfKeypointsScene, double tolerance) {
        int matches = 0;
        double slope = 0;
        List<Double> slopes = new ArrayList<>();
        for (int i = matches; matches < listOfGoodMatches.size(); matches++) {
            Point pta = listOfKeypointsObject.get(listOfGoodMatches.get(matches).queryIdx).pt;
            Point ptb = listOfKeypointsScene.get(listOfGoodMatches.get(matches).trainIdx).pt;

            double match_slope = (ptb.y - pta.y) / (ptb.x - pta.x);
            slopes.add(match_slope);
            slope += match_slope;
        }

        double mean_slope = slope / matches;
        List<DMatch> result = new ArrayList<>();
        double[] bounds = {mean_slope * (1 - tolerance), mean_slope * (1 + tolerance)};

        for (int i = 0; i < listOfGoodMatches.size(); i++) {
            double match_slope = slopes.get(i);
            if (match_slope >= bounds[0] && match_slope <= bounds[1]) {
                result.add(listOfGoodMatches.get(i));
            }
        }

        return result;
    }
}
