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
    // Initialize the output image with the same size and type as the input
    cv::Mat output = cv::Mat::zeros(image.size(), image.type());

    // Iterate over each pixel
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Get the BGR values of the current pixel
            cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);

            // Check if the pixel is white
            if (bgr[0] >= 200 && bgr[1] >= 200 && bgr[2] >= 200) {
                output.at<cv::Vec3b>(y, x) = bgr;
                continue;
            }

            // Convert BGR to HSV
            float b = bgr[0] / 255.0f;
            float g = bgr[1] / 255.0f;
            float r = bgr[2] / 255.0f;

            float max_value = std::max({r, g, b});
            float min_value = std::min({r, g, b});
            float delta = max_value - min_value;

            float hue = 0;

            if (delta != 0) {
                if (max_value == r) {
                    hue = 60 * (fmod(((g - b) / delta), 6));
                } else if (max_value == g) {
                    hue = 60 * (((b - r) / delta) + 2);
                } else if (max_value == b) {
                    hue = 60 * (((r - g) / delta) + 4);
                }
                if (hue < 0) {
                    hue += 360;
                }
            }

            float saturation = max_value == 0 ? 0 : (delta / max_value);
            float value = max_value;

            // Check if the pixel is yellow
            if ((hue >= 90 && hue <= 110) && saturation >= (100 / 255.0f) && value >= (100 / 255.0f)) {
                output.at<cv::Vec3b>(y, x) = bgr;
            }
        }
    }

    return output;
}


cv::Mat grayscale(const cv::Mat& img) {
    // Create a single-channel output image with the same size as the input
    cv::Mat gray(img.rows, img.cols, CV_8UC1);

    // Iterate over each pixel
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            // Get the BGR values of the current pixel
            cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);

            // Calculate the grayscale value using the luminance formula
            // Common formula: 0.299*R + 0.587*G + 0.114*B
            uchar gray_value = static_cast<uchar>(0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]);

            // Set the grayscale value in the output image
            gray.at<uchar>(y, x) = gray_value;
        }
    }

    return gray;
    // // Applies the Grayscale transform
    // // This will return an image with only one color channel
    // cv::Mat gray;
    // cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // return gray;
}

std::vector<std::vector<float>> generateGaussianKernel(int kernel_size, float sigma) {
    std::vector<std::vector<float>> kernel(kernel_size, std::vector<float>(kernel_size));
    float sum = 0.0; // For normalization
    int half_size = kernel_size / 2;

    // Generate the Gaussian kernel
    for (int x = -half_size; x <= half_size; x++) {
        for (int y = -half_size; y <= half_size; y++) {
            float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[x + half_size][y + half_size] = value;
            sum += value;
        }
    }

    // Normalize the kernel so that the sum is 1
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

cv::Mat gaussianBlur(const cv::Mat& img, int kernel_size) {
    int half_size = kernel_size / 2;
    float sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8; // Automatic sigma calculation

    // Generate the Gaussian kernel
    std::vector<std::vector<float>> kernel = generateGaussianKernel(kernel_size, sigma);

    // Create output image (same size as input, and same type)
    cv::Mat blurred = cv::Mat::zeros(img.size(), img.type());

    // Iterate over each pixel in the input image
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            float sum = 0.0;
            float weight_sum = 0.0;

            // Apply convolution with Gaussian kernel
            for (int ky = -half_size; ky <= half_size; ky++) {
                for (int kx = -half_size; kx <= half_size; kx++) {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;

                    // Boundary check (replicate edge pixels)
                    neighbor_x = std::min(std::max(neighbor_x, 0), img.cols - 1);
                    neighbor_y = std::min(std::max(neighbor_y, 0), img.rows - 1);

                    // Access the intensity of the neighbor pixel (grayscale image assumed)
                    float pixel_value = static_cast<float>(img.at<uchar>(neighbor_y, neighbor_x));

                    // Gaussian weight
                    float weight = kernel[ky + half_size][kx + half_size];

                    sum += pixel_value * weight;
                    weight_sum += weight;
                }
            }

            // Assign the blurred value to the output image
            blurred.at<uchar>(y, x) = static_cast<uchar>(sum / weight_sum);
        }
    }

    return blurred;
}

// Step 3: Sobel Operator to compute gradients
void sobelOperator(const cv::Mat& img, cv::Mat& magnitude, cv::Mat& direction) {
    cv::Mat grad_x = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat grad_y = cv::Mat::zeros(img.size(), CV_32F);

    float Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
            float sum_x = 0.0, sum_y = 0.0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = img.at<uchar>(y + ky, x + kx);
                    sum_x += pixel * Gx[ky + 1][kx + 1];
                    sum_y += pixel * Gy[ky + 1][kx + 1];
                }
            }

            grad_x.at<float>(y, x) = sum_x;
            grad_y.at<float>(y, x) = sum_y;

            magnitude.at<float>(y, x) = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            direction.at<float>(y, x) = std::atan2(sum_y, sum_x);
        }
    }
}

// Step 4: Non-Maximum Suppression
cv::Mat nonMaximumSuppression(const cv::Mat& magnitude, const cv::Mat& direction) {
    cv::Mat suppressed = cv::Mat::zeros(magnitude.size(), CV_8U);

    for (int y = 1; y < magnitude.rows - 1; y++) {
        for (int x = 1; x < magnitude.cols - 1; x++) {
            float angle = direction.at<float>(y, x) * (180.0 / M_PI);
            angle = std::fmod(angle + 180, 180); // Normalize angle to [0, 180)

            float mag = magnitude.at<float>(y, x);
            float neighbor1 = 0, neighbor2 = 0;

            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                neighbor1 = magnitude.at<float>(y, x - 1);
                neighbor2 = magnitude.at<float>(y, x + 1);
            } else if (22.5 <= angle && angle < 67.5) {
                neighbor1 = magnitude.at<float>(y - 1, x + 1);
                neighbor2 = magnitude.at<float>(y + 1, x - 1);
            } else if (67.5 <= angle && angle < 112.5) {
                neighbor1 = magnitude.at<float>(y - 1, x);
                neighbor2 = magnitude.at<float>(y + 1, x);
            } else if (112.5 <= angle && angle < 157.5) {
                neighbor1 = magnitude.at<float>(y - 1, x - 1);
                neighbor2 = magnitude.at<float>(y + 1, x + 1);
            }

            if (mag >= neighbor1 && mag >= neighbor2) {
                suppressed.at<uchar>(y, x) = static_cast<uchar>(mag);
            }
        }
    }

    return suppressed;
}

// Step 5: Double Thresholding and Edge Tracking by Hysteresis
cv::Mat doubleThresholdAndHysteresis(const cv::Mat& img, int low_thresh, int high_thresh) {
    cv::Mat edges = cv::Mat::zeros(img.size(), CV_8U);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            int pixel = img.at<uchar>(y, x);
            if (pixel >= high_thresh) {
                edges.at<uchar>(y, x) = 255; // Strong edge
            } else if (pixel >= low_thresh) {
                edges.at<uchar>(y, x) = 128; // Weak edge
            }
        }
    }

    return edges;
}

cv::Mat canny(const cv::Mat& img, int low_threshold, int high_threshold) {
    // Input: already Gaussian-blurred image
    cv::Mat magnitude = cv::Mat::zeros(img.size(), CV_32F);
    cv::Mat direction = cv::Mat::zeros(img.size(), CV_32F);

    // Step 1: Compute gradients using Sobel operator
    sobelOperator(img, magnitude, direction);

    // Step 2: Non-Maximum Suppression to thin edges
    cv::Mat suppressed = nonMaximumSuppression(magnitude, direction);

    // Step 3: Double Thresholding and Edge Tracking by Hysteresis
    cv::Mat edges = doubleThresholdAndHysteresis(suppressed, low_threshold, high_threshold);

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

