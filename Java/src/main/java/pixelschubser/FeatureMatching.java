package pixelschubser;

import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.features2d.DescriptorMatcher;

import java.util.ArrayList;
import java.util.List;

public class FeatureMatching {
    public static List<DMatch> flannMatching(Mat descriptor_a, Mat descriptor_b) {
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptor_a, descriptor_b, knnMatches, 2);
        System.out.println("Step 1 matches: " + knnMatches.size());

        /**
         * Filter matches using the Lowe's ratio test
         */
//            float ratioThresh = 0.75f;
        float ratioThresh = 0.6f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }

        return listOfGoodMatches;
    }

    public static List<DMatch> bfMatching(Mat descriptor_a, Mat descriptor_b) {
        MatOfDMatch matchMatrix = new MatOfDMatch();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        matcher.match(descriptor_a, descriptor_b, matchMatrix);
        DMatch[] matches = matchMatrix.toArray();

        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (DMatch match : matches) {
            if (match.distance <= 50) {
                listOfGoodMatches.add(match);
            }
        }

        return listOfGoodMatches;
    }
}
