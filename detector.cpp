#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat filterColors(const cv::Mat& image);
cv::Mat grayscale(const cv::Mat& img);
cv::Mat gaussianBlur(const cv::Mat& img, int kernel_size);
cv::Mat canny(const cv::Mat& img, int low_threshold, int high_threshold);
cv::Mat regionOfInterest(const cv::Mat& img, const std::vector<std::vector<cv::Point>>& vertices);
void drawLines(cv::Mat& img, const std::vector<cv::Vec4i>& lines, cv::Scalar color, int thickness);
cv::Mat houghLines(const cv::Mat& img, double rho, double theta, int threshold, double min_line_len, double max_line_gap);
cv::Mat weightedImg(const cv::Mat& img, const cv::Mat& initial_img, double alpha, double beta, double lambda);
cv::Mat annotateImageArray(const cv::Mat& image_in);


int main() {
    // Load an image
    std::string img_path = "japan_road.png";
    

    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    try {
        cv::Mat annotated_image = annotateImageArray(image);

        // Display the result
        cv::imshow("Annotated Image", annotated_image);
        cv::waitKey(0);  // Wait for a key press to close the window
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}   



cv::Mat filterColors(const cv::Mat& image) {
    // Filter the image to include only yellow and white pixels

    // Filter white pixels
    int white_threshold = 200; //130
    cv::Scalar lower_white(white_threshold, white_threshold, white_threshold);
    cv::Scalar upper_white(255, 255, 255);
    cv::Mat white_mask;
    cv::inRange(image, lower_white, upper_white, white_mask);
    cv::Mat white_image;
    cv::bitwise_and(image, image, white_image, white_mask);

    // Filter yellow pixels
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar lower_yellow(90, 100, 100);
    cv::Scalar upper_yellow(110, 255, 255);
    cv::Mat yellow_mask;
    cv::inRange(hsv, lower_yellow, upper_yellow, yellow_mask);
    cv::Mat yellow_image;
    cv::bitwise_and(image, image, yellow_image, yellow_mask);

    // Combine the two above images
    cv::Mat image2;
    cv::addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, image2);

    return image2;
}


cv::Mat grayscale(const cv::Mat& img) {
    // Applies the Grayscale transform
    // This will return an image with only one color channel
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat gaussianBlur(const cv::Mat& img, int kernel_size) {
    // Applies a Gaussian Noise kernel
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), 0);
    return blurred;
}

cv::Mat canny(const cv::Mat& img, int low_threshold, int high_threshold) {
    // Applies the Canny transform
    cv::Mat edges;
    cv::Canny(img, edges, low_threshold, high_threshold);
    return edges;
}

cv::Mat regionOfInterest(const cv::Mat& img, const std::vector<std::vector<cv::Point>>& vertices) {
    // Applies an image mask.
    // Only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black.

    // defining a blank mask to start with
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

    // defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    cv::Scalar ignore_mask_color;
    if (img.channels() > 1) {
        ignore_mask_color = cv::Scalar(255, 255, 255);  // Assuming 3 channels (BGR)
    } else {
        ignore_mask_color = cv::Scalar(255);
    }

    // filling pixels inside the polygon defined by "vertices" with the fill color
    cv::fillPoly(mask, vertices, ignore_mask_color);

    // returning the image only where mask pixels are nonzero
    cv::Mat masked_image;
    cv::bitwise_and(img, mask, masked_image);
    return masked_image;
}


void drawLines(cv::Mat& img, const std::vector<cv::Vec4i>& lines, cv::Scalar color = cv::Scalar(0, 255, 255), int thickness = 5) {
    if (lines.empty()) {
        return;
    }

    bool draw_right = true;
    bool draw_left = true;

    double slope_threshold = 0.5;
    std::vector<double> slopes;
    std::vector<cv::Vec4i> new_lines;

    // Calculate slopes and filter lines based on the slope threshold
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        
        double slope = (x2 - x1 == 0) ? 999.0 : static_cast<double>(y2 - y1) / (x2 - x1);
        
        if (std::abs(slope) > slope_threshold) {
            slopes.push_back(slope);
            new_lines.push_back(line);
        }
    }

    std::vector<cv::Vec4i> right_lines, left_lines;
    int img_x_center = img.cols / 2;

    // Separate lines into right and left based on their slope and position
    for (size_t i = 0; i < new_lines.size(); ++i) {
        const auto& line = new_lines[i];
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        
        if (slopes[i] > 0 && x1 > img_x_center && x2 > img_x_center) {
            right_lines.push_back(line);
        } else if (slopes[i] < 0 && x1 < img_x_center && x2 < img_x_center) {
            left_lines.push_back(line);
        }
    }

    auto linearRegression = [](const std::vector<cv::Vec4i>& lines, bool& draw_flag) -> std::pair<double, double> {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        int n = 0;

        for (const auto& line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];

            sum_x += x1 + x2;
            sum_y += y1 + y2;
            sum_xy += x1 * y1 + x2 * y2;
            sum_xx += x1 * x1 + x2 * x2;
            n += 2;
        }

        if (n > 0) {
            double denominator = n * sum_xx - sum_x * sum_x;
            if (denominator != 0) {
                double m = (n * sum_xy - sum_x * sum_y) / denominator;
                double b = (sum_y * sum_xx - sum_x * sum_xy) / denominator;
                return {m, b};
            }
        }
        
        draw_flag = false;
        return {1.0, 1.0};
    };

    double right_m, right_b, left_m, left_b;
    std::tie(right_m, right_b) = linearRegression(right_lines, draw_right);
    std::tie(left_m, left_b) = linearRegression(left_lines, draw_left);

    int y1 = img.rows;
    int y2 = static_cast<int>(img.rows * (1 - 0.4)); // Adjust the trap_height value as per your requirement

    int right_x1 = static_cast<int>((y1 - right_b) / right_m);
    int right_x2 = static_cast<int>((y2 - right_b) / right_m);
    
    int left_x1 = static_cast<int>((y1 - left_b) / left_m);
    int left_x2 = static_cast<int>((y2 - left_b) / left_m);

    if (draw_right) {
        cv::line(img, cv::Point(right_x1, y1), cv::Point(right_x2, y2), color, thickness);
    }
    if (draw_left) {
        cv::line(img, cv::Point(left_x1, y1), cv::Point(left_x2, y2), color, thickness);
    }
}

cv::Mat houghLines(const cv::Mat& img, double rho, double theta, int threshold, double min_line_len, double max_line_gap) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(img, lines, rho, theta, threshold, min_line_len, max_line_gap);

    cv::Mat line_img(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    drawLines(line_img, lines);

    return line_img;
}

cv::Mat weightedImg(const cv::Mat& img, const cv::Mat& initial_img, double alpha = 0.8, double beta = 1.0, double lambda = 0.0) {
    // Ensure the images have the same size and type
    if (img.size() != initial_img.size() || img.type() != initial_img.type()) {
        throw std::invalid_argument("Input images must have the same size and type.");
    }

    cv::Mat result;
    cv::addWeighted(initial_img, alpha, img, beta, lambda, result);
    return result;
}

cv::Mat annotateImageArray(const cv::Mat& image_in) {
    // Constants 
    int kernel_size = 3;
    int low_threshold = 50;
    int high_threshold = 150;
    double trap_bottom_width = 0.85;
    double trap_top_width = 0.07;
    double trap_height = 0.4;
    double rho = 2;
    double theta = 3.14159 / 180;
    int threshold = 15;
    int min_line_length = 10;
    int max_line_gap = 20;

    // Only keep white and yellow pixels in the image, all other pixels become black
    cv::Mat image = filterColors(image_in);

    // Read in and grayscale the image
    cv::Mat gray = grayscale(image);

    // Apply Gaussian smoothing
    cv::Mat blur_gray = gaussianBlur(gray, kernel_size);

    // Apply Canny Edge Detector
    cv::Mat edges = canny(blur_gray, low_threshold, high_threshold);

    // Create masked edges using trapezoid-shaped region-of-interest
    cv::Size imshape = image.size();
    std::vector<std::vector<cv::Point>> vertices = { {
        cv::Point((imshape.width * (1 - trap_bottom_width)) / 2, imshape.height),
        cv::Point((imshape.width * (1 - trap_top_width)) / 2, imshape.height - static_cast<int>(imshape.height * trap_height)),
        cv::Point(imshape.width - (imshape.width * (1 - trap_top_width)) / 2, imshape.height - static_cast<int>(imshape.height * trap_height)),
        cv::Point(imshape.width - (imshape.width * (1 - trap_bottom_width)) / 2, imshape.height)
    } };

    cv::Mat masked_edges = regionOfInterest(edges, vertices);

    // Run Hough on edge detected image
    cv::Mat line_image = houghLines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap);

    // Draw lane lines on the original image
    cv::Mat initial_image;
    image_in.convertTo(initial_image, CV_8U);
    
    cv::Mat annotated_image = weightedImg(line_image, initial_image);

    return annotated_image;
}

