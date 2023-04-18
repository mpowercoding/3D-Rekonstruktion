package pixelschubser.codesamples;

import org.opencv.calib3d.StereoBM;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import pixelschubser.Image;
import pixelschubser.MatchDrawer;
import pixelschubser.exceptions.ImageObjetEmptyException;

import java.util.ArrayList;

import static org.opencv.core.CvType.CV_64FC1;
import static pixelschubser.Homography.getHomographyMatrix;

public class test20220204 {
    public void run() throws ImageObjetEmptyException {
        /**
         * import numpy as np
         * import cv2 as cv
         * from matplotlib import pyplot as plt
         * imgL = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
         * imgR = cv.imread('tsukuba_r.png', cv.IMREAD_GRAYSCALE)
         * stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
         * disparity = stereo.compute(imgL,imgR)
         * plt.imshow(disparity,'gray')
         * plt.show()
         */

        Image a = new Image("Milch/IMG_0675.JPG", Imgcodecs.IMREAD_GRAYSCALE);
        Mat a_mat = a.getImgObject();
        Image b = new Image("Milch/IMG_0677.JPG", Imgcodecs.IMREAD_GRAYSCALE);
        Mat b_mat = b.getImgObject();

        StereoBM stereo = StereoBM.create(16, 15);

        Mat disparity = new Mat(3, 3, CV_64FC1);
        stereo.compute(a_mat, b_mat, disparity);


        //draw_matches(MatOfKeyPoint keypointsObject, MatOfKeyPoint keypointsScene, List<DMatch> listOfGoodMatches,
        //                                    Mat imgObject, Mat imgScene, Mat imgMatches)
        Mat empty_obj = new Mat(3, 3, CV_64FC1);
        MatOfKeyPoint empty_matofkeypoint = new MatOfKeyPoint();
        MatchDrawer.draw_matches(empty_matofkeypoint, empty_matofkeypoint, new ArrayList<DMatch>(), a_mat, b_mat, empty_obj);
    }
}
