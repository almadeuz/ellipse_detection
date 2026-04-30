#pragma once
#include "types.hpp"

class Filter {
public:
    Filter(const Config& config);

    // Удаляет эллипсы сильно вылезающие за контур или слишком тонкие
    std::vector<DetectedEllipse> filterFitness(const std::vector<DetectedEllipse>& ellipses_data, const cv::Mat& binary, const cv::Rect& bbox);

    // Подавляет эллипсы-дубликаты на основе IoU
    std::vector<DetectedEllipse> filterNMS(const std::vector<DetectedEllipse>& candidates, const cv::Mat& binary, double iou, double beta);

    // Обновляет границы эллипса относительно его центра
    std::vector<DetectedEllipse> refineResults(const std::vector<DetectedEllipse>& accepted, const cv::Mat& binary);

    // Использует жадный алгоритм с переоценкой эллипсов для составления финального набора
    std::vector<DetectedEllipse> applyGreedy(const std::vector<DetectedEllipse>& candidates, const cv::Mat& binary, const cv::Rect& contour_bbox);

private:
    struct Cand {
        DetectedEllipse data;
        cv::Mat mask;
        int area = 0;
        double combined = 0.0;
        double iou = 0.0;
        bool used = false;
    };

    std::vector<cv::Point2f> getEllipsePolygon(const EllipseParams& e, double& poly_area);

    double getEllipsesIntersectionArea(const EllipseParams& e1, const EllipseParams& e2,double& area1, double& area2);

    double computeEllipseIntersection(const EllipseParams& e1, const EllipseParams& e2);

    double computeSurfaceRatio(const EllipseParams& ellipse, const cv::Mat& binary);

    cv::Mat fillLargestContour(const cv::Mat& roi, int W, int H);

    cv::Mat buildEllipseMask(const DetectedEllipse& cand, const cv::Rect& roi_offset, int W, int H);

    double computeBoundaryIoU(const cv::Mat& predicted, const cv::Mat& ground_truth, int bandwidth);

    double computeBoundaryPrecision(const cv::Mat& predicted, const cv::Mat& ground_truth, int bandwidth);

    std::vector<DetectedEllipse> filterByMinAxis(const std::vector<DetectedEllipse>& src);
    std::vector<DetectedEllipse> filterByAspectRatio(const std::vector<DetectedEllipse>& src);
    std::vector<DetectedEllipse> filterBySurfaceRatio(const std::vector<DetectedEllipse>& src, const cv::Mat& binary);
    std::vector<DetectedEllipse> filterByAreaRatio(const std::vector<DetectedEllipse>& src);

    std::vector<DetectedEllipse> removeOverlappedEllipses(const std::vector<Cand>& selected, const cv::Mat& target_blob, int W, int H, double overlap_threshold);

    void refineSingleEllipse(DetectedEllipse& det, const cv::Mat& binary);

    const Config& config;
};