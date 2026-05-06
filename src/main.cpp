#include "detector.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <string>

namespace fs = std::filesystem;

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

void printUsage() {
    std::cout << "Usage:" << std::endl << std::endl;
    std::cout << "Detection:" << std::endl;
    std::cout << "main --mode detect --input <image_path> --out <output_dir>" << std::endl << std::endl;
    std::cout << "Save binary:" << std::endl;
    std::cout << "main --mode binary --input <image_path> --out <output_dir>" << std::endl;
}

std::string getArgValue(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == key) {
            return argv[i + 1];
        }
    }
    return "";
}

bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

bool fileExists(const fs::path& path) {
    return fs::exists(path) && fs::is_regular_file(path);
}

void saveVisualResults(const fs::path& outDir, const cv::Mat& image, const std::vector<EllipseParams>& ellipses) {
    fs::path outPath = outDir / "result.jpg";
    cv::Mat result = image.clone();

    for (const auto& e : ellipses) {
        cv::ellipse(result, cv::RotatedRect(e.center, e.size, e.angle), cv::Scalar(0, 255, 0), 2);
        cv::circle(result, e.center, 2, cv::Scalar(0, 0, 255), -1);
    }

    std::cout << "Saving visual result to: " << outPath.generic_string() << std::endl;
    if (!cv::imwrite(outPath.string(), result)) {
        throw std::runtime_error("Failed to save image: " + outPath.string());
    }
}

void saveCsvResults(const fs::path& outDir, const std::vector<EllipseParams>& ellipses) {
    fs::path outPath = outDir / "result.csv";
    std::ofstream f(outPath);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open CSV for writing: " + outPath.generic_string());
    }

    f << "id,x,y,major,minor,angle\n";
    f << std::fixed << std::setprecision(2);

    for (const auto& e : ellipses) {
        double major = std::max(e.size.width, e.size.height);
        double minor = std::min(e.size.width, e.size.height);

        f << e.id << "," << e.center.x << "," << e.center.y << ","
          << major << "," << minor << "," << e.angle << std::endl;
    }
    std::cout << "CSV data saved to: " << outPath.generic_string() << std::endl;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || hasFlag(argc, argv, "--help")) {
            printUsage();
            return 0;
        }

        const std::string mode = getArgValue(argc, argv, "--mode");
        const std::string inputPath = getArgValue(argc, argv, "--input");
        const std::string outDirPath = getArgValue(argc, argv, "--out");

        if (mode.empty() || inputPath.empty() || outDirPath.empty()) {
            std::cerr << "Error: --mode, --input and --out are required\n";
            printUsage();
            return 1;
        }

        fs::path inputImagePath(inputPath);
        fs::path outputDir(outDirPath);

        if (!fileExists(inputImagePath)) {
            std::cerr << "Error: input image not found: " << inputImagePath.generic_string() << "\n";
            return 1;
        }

        fs::create_directories(outputDir);

        cv::Mat image = cv::imread(inputImagePath.string());
        if (image.empty()) {
            std::cerr << "Error: can't read image: " << inputImagePath.generic_string() << "\n";
            return 1;
        }

        Detector detector;

        if (mode == "binary") {
            std::cout << "Running in binary mode...\n";
            cv::Mat binary = detector.toBinary(image);
            fs::path binPath = outputDir / "binary.jpg";
            
            if (cv::imwrite(binPath.generic_string(), binary)) {
                std::cout << "Binary mask saved to: " << binPath.generic_string() << "\n";
            } else {
                std::cerr << "Error: Failed to save binary mask\n";
                return 1;
            }
            return 0; 
        }

        if (mode == "detect") {
            std::cout << "Processing image: " << inputImagePath.filename().generic_string() 
                      << " (" << image.cols << "x" << image.rows << ")\n";

            auto t0 = cv::getTickCount();
            std::vector<EllipseParams> ellipses = detector.detectEllipses(image);
            double elapsed = (cv::getTickCount() - t0) / cv::getTickFrequency();

            std::cout << "Detection finished in " << elapsed << " s.\n";
            std::cout << "Detected: " << ellipses.size() << " ellipses.\n";

            saveVisualResults(outputDir, image, ellipses);
            saveCsvResults(outputDir, ellipses);

            std::cout << "All results saved to: " << outputDir.generic_string() << "\n";
        } else {
            std::cerr << "Error: Unknown mode '" << mode << "'\n";
            printUsage();
            return 1;
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 1;
    }
}