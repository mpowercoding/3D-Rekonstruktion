package pixelschubser;

import org.jetbrains.annotations.NotNull;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.norm;

public class Homography {
    @NotNull
    public static Mat getHomographyMatrix(MatOfKeyPoint keypointsObject, MatOfKeyPoint keypointsScene, List<DMatch> listOfGoodMatches) {
        return prepareHomography(keypointsObject, keypointsScene, listOfGoodMatches);
    }

    @NotNull
    private static Mat calculateHomographyMatrix(List<Point> obj, List<Point> scene) {
        MatOfPoint2f objMat = new MatOfPoint2f(), sceneMat = new MatOfPoint2f();
        objMat.fromList(obj);
        sceneMat.fromList(scene);
        double ransacReprojThreshold = 3.0;
        return Calib3d.findHomography(objMat, sceneMat, Calib3d.RANSAC, ransacReprojThreshold);
    }

    @NotNull
    private static Mat prepareHomography(MatOfKeyPoint keypointsObject, MatOfKeyPoint keypointsScene, List<DMatch> listOfGoodMatches) {
        List<Point> obj = new ArrayList<>();
        List<Point> scene = new ArrayList<>();
        List<KeyPoint> listOfKeypointsObject = keypointsObject.toList();
        List<KeyPoint> listOfKeypointsScene = keypointsScene.toList();

        for (DMatch listOfGoodMatch : listOfGoodMatches) {
            obj.add(listOfKeypointsObject.get(listOfGoodMatch.queryIdx).pt);
            scene.add(listOfKeypointsScene.get(listOfGoodMatch.trainIdx).pt);
        }
        return calculateHomographyMatrix(obj, scene);
    }

    //https://stackoverflow.com/questions/8927771/computing-camera-pose-with-homography-matrix-based-on-4-coplanar-points/10781165#10781165
//    public static void cameraPoseFromHomography(Mat H, Mat pose)
//    {
//        pose = eye(3, 4, cv2.CV_32FC1);      // 3x4 matrix, the camera pose
//        float norm1 = (float)norm(H.col(0));
//        float norm2 = (float)norm(H.col(1));
//        float tnorm = (norm1 + norm2) / 2.0f; // Normalization value
//
//        Mat p1 = H.col(0);       // Pointer to first column of H
//        Mat p2 = pose.col(0);    // Pointer to first column of pose (empty)
//
//        cv::normalize(p1, p2);   // Normalize the rotation, and copies the column to pose
//
//        p1 = H.col(1);           // Pointer to second column of H
//        p2 = pose.col(1);        // Pointer to second column of pose (empty)
//
//        cv::normalize(p1, p2);   // Normalize the rotation and copies the column to pose
//
//        p1 = pose.col(0);
//        p2 = pose.col(1);
//
//        Mat p3 = p1.cross(p2);   // Computes the cross-product of p1 and p2
//        Mat c2 = pose.col(2);    // Pointer to third column of pose
//        p3.copyTo(c2);       // Third column is the crossproduct of columns one and two
//
//        pose.col(3) = H.col(2) / tnorm;  //vector t [R|t] is the last column of pose
//    }
}
