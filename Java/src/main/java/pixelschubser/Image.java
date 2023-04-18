package pixelschubser;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import pixelschubser.exceptions.ImageObjetEmptyException;

import java.util.Optional;

public class Image {
    private String path;
    private Mat imgObject;
    private MatOfKeyPoint keypointsObject;
    private Mat descriptorsObject;
    private int mode;


    public Image(String resource_name) {
        this.path = getClass().getClassLoader().getResource(resource_name).getPath();
        this.mode = 0;
    }

    public Image(String resource_name, int mode) {
        this.path = getClass().getClassLoader().getResource(resource_name).getPath();
        this.mode = (mode >= 0 && mode <= 255)?mode:0;
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
        int nFeatures = 0;
        int nOctaveLayers = 10;
        double contrastThreshold = 0.05;
        double edgeThreshold = 100;
        double sigma = 1.6;
        SIFT detector = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        MatOfKeyPoint keypointsObject = new MatOfKeyPoint();
        Mat descriptorsObject = new Mat();
        detector.detectAndCompute(imgObject, new Mat(), keypointsObject, descriptorsObject);

        this.keypointsObject = keypointsObject;
        this.descriptorsObject = descriptorsObject;
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
