#pragma once
#include "types.hpp"
#include "contour_segmenter.hpp"
#include "ellipse_filter.hpp"

// Класс з
class Detector {
public:
    Detector();

    // Извлекает контуры и запускает цикл обработки
    std::vector<EllipseParams> detectEllipses(const cv::Mat& image);

private:
    // Производит бинаризацию входного изображения.
    cv::Mat toBinary(const cv::Mat& image);

    Config config;
    Segmenter segmenter;
    Filter filter_stage;
};