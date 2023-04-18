package pixelschubser;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import pixelschubser.exceptions.ImageObjetEmptyException;

import java.util.Optional;

public class Image2 {
    private String path;
    private Mat imgObject;
    private MatOfKeyPoint keypointsObject;
    private Mat descriptorsObject;
    private int mode;


    public Image2(String resource_name) {
        this.path = getClass().getClassLoader().getResource(resource_name).getPath();
        this.mode = 0;
    }

    public Image2(String resource_name, int mode) {
        this.path = getClass().getClassLoader().getResource(resource_name).getPath();
        this.mode = (mode >= 0 && mode <= 255) ? mode : 0;
    }

    public Mat getImgObject() throws ImageObjetEmptyException {
        Optional<Mat> opt = Optional.ofNullable(imgObject);
        if (!opt.isPresent()) {
            this.imgObject = Imgcodecs.imread(path, this.mode);
        }
        if (imgObject.empty()) {
            throw new ImageObjetEmptyException();
        }
        return imgObject;
    }

    private void extract_features() {
        //http://www.java2s.com/example/java-api/org/opencv/features2d/orb/compute-3-0.html
        ORB orb = ORB.create();

        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        orb.detect(imgObject, keypoints);

        Mat descriptors = new Mat();
        orb.compute(imgObject, keypoints, descriptors);

        this.keypointsObject = keypoints;
        this.descriptorsObject = descriptors;
    }


    public MatOfKeyPoint getMatOfKeyPoint() {
        Optional<MatOfKeyPoint> opt = Optional.ofNullable(keypointsObject);
        if (!opt.isPresent()) {
            this.extract_features();
        }
        return keypointsObject;
    }

    public Mat getKeypointDescriptionObject() {
        Optional<Mat> opt = Optional.ofNullable(descriptorsObject);
        if (!opt.isPresent()) {
            this.extract_features();
        }
        return descriptorsObject;
    }
}
