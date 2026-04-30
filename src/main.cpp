#include "detector.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

void saveResults(const cv::Mat& image, const std::vector<EllipseParams>& ellipses, const std::string& filename) {
    cv::Mat result = image.clone();

    for (auto& e : ellipses) {
        cv::ellipse(result, cv::RotatedRect(e.center, e.size, e.angle), cv::Scalar(0, 255, 0), 2);
        cv::circle(result, e.center, 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imwrite(filename, result, {cv::IMWRITE_JPEG_QUALITY, 100});
    std::cout << "Saved to " << filename << " (Total: " << ellipses.size() << ")" << std::endl;
}

void saveCsv(const std::vector<EllipseParams>& ellipses, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cant open CSV: " << filename << std::endl;
        return;
    }

    f << "id,x,y,a,b,angle\n";
    f << std::fixed << std::setprecision(2);

    for (const auto& e : ellipses) {
        double major = std::max(e.size.width, e.size.height);
        double minor = std::min(e.size.width, e.size.height);

        f << e.id << "," << e.center.x << "," << e.center.y << ","
          << major << "," << minor << "," << e.angle << std::endl;
    }

    std::cout << "CSV saved to " << filename << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "TO USE: " << argv[0] << " path/to/image.jpg" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Cannot load: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image: " << image.cols << "x" << image.rows << std::endl;

    Detector detector;
    auto t0 = cv::getTickCount();
    auto ellipses = detector.detectEllipses(image);
    double elapsed = (cv::getTickCount() - t0) / cv::getTickFrequency();

    std::cout << "Time: " << elapsed << " s" << std::endl;
    std::cout << "Detected: " << ellipses.size() << " ellipses" << std::endl;

    saveResults(image, ellipses, "result.jpg");
    saveCsv(ellipses, "result.csv");

    return 0;
}