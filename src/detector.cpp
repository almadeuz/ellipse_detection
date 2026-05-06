#include "detector.hpp"
#include <iostream>
#include <algorithm>

Detector::Detector() : segmenter(config), filter_stage(config) {};

cv::Mat Detector::toBinary(const cv::Mat& image) {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    
    cv::Mat cL;
    cv::extractChannel(lab, cL, 0);

    cv::Mat bin_L;
    cv::threshold(cL, bin_L, 235, 255, cv::THRESH_BINARY);
    cv::bitwise_not(bin_L, bin_L);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_L, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(bin_L, contours, -1, cv::Scalar(255), cv::FILLED);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(bin_L, bin_L, cv::MORPH_OPEN, kernel);

    return bin_L;
}

std::vector<EllipseParams> Detector::detectEllipses(const cv::Mat& image) {
    std::vector<EllipseParams> results;
    
    try {
        cv::Mat binary = toBinary(image);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        int cnt = 0;
        for (const auto& contour : contours) {
            if (contour.size() < 20 || contour.size() > image.size().width * 2 + image.size().height * 2 - 5) continue;
            cnt++;

            segmenter.resetSegmentIdCounter();
            cv::Rect bbox = cv::boundingRect(contour);
            auto res = segmenter.analyzeContour(contour, binary, cnt, bbox);
            auto filtered = filter_stage.filterFitness(res, binary, bbox);
            auto refined_set1 = filter_stage.refineResults(filtered, binary);
            auto unique_candidates = filter_stage.filterNMS(refined_set1, binary, 0.8, 0.5);
            auto optimal_set = filter_stage.applyGreedy(unique_candidates, binary, bbox);
            static int ellipse_internal_id = 0;
            std::vector<EllipseParams> ellipses;
            ellipses.reserve(optimal_set.size());
            for (const auto& i : optimal_set) {
                EllipseParams p = i.ellipse;
                p.id = ellipse_internal_id++;
                p.contour_id = cnt;
                ellipses.push_back(p);
            }

            results.insert(results.end(), ellipses.begin(), ellipses.end());
        }
    } catch (const cv::Exception& e) {
        std::cerr << "err: " << e.what() << std::endl;
    }
    
    return results;
}