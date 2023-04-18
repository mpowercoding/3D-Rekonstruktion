package pixelschubser;

import nu.pattern.OpenCV;
import org.opencv.core.Core;
import pixelschubser.codesamples.test20220204;
import pixelschubser.exceptions.ImageObjetEmptyException;

public class main {
    public static void main(String[] args) throws ImageObjetEmptyException {
//        new MatchingKeypoints().run();

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        OpenCV.loadLocally();
//        System.load("/home/schneidermi/Dokumente/Services/3drek/opencv-4.5.5/build2/bin/");

//        new SIFT_Example().run();
//        new ORB_Example().run();
        new test20220204().run();
    }
}
