#pragma once
#include "types.hpp"
#include "contour_segmenter.hpp"
#include "ellipse_filter.hpp"

// Класс з
class Detector {
public:
    Detector();
    // Производит бинаризацию входного изображения.
    cv::Mat toBinary(const cv::Mat& image); 
    // Извлекает контуры и запускает цикл обработки
    std::vector<EllipseParams> detectEllipses(const cv::Mat& image);

private:

    Config config;
    Segmenter segmenter;
    Filter filter_stage;
};