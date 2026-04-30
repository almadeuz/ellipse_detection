#pragma once
#include "types.hpp"

class Segmenter {
public:
    Segmenter(const Config& config);
    
    // Сбрасывает счетчик id сегментов
    void resetSegmentIdCounter() { segment_id_counter = 0; }

    
    std::vector<DetectedEllipse> analyzeContour(const std::vector<cv::Point>& contour, const cv::Mat& binary, int contour_id, const cv::Rect& contour_bbox);

private:
    SegmentGroup createGroupFromIndices(const std::vector<size_t>& indices,  const std::vector<SegmentGroup>& source_sets);
    
    bool isIndexInRange(size_t idx, size_t start, size_t end, size_t total);

    double getPointSide(const cv::Point2f& point, const cv::Point2f& P, const cv::Point2f& d);
    
    bool pointInSearchRegion(const cv::Point2f& point, const cv::Point& P1, const cv::Point& P2, const cv::Point& Pmid, const cv::Point2f& tangent1, const cv::Point2f& tangent2);

    bool groupInSearchRegion(const SegmentGroup& group_to_check, const SegmentGroup& reference_group, const cv::Mat& binary, const cv::Rect& contour_bbox);

    std::vector<SegmentGroup> mergeBySearchRegion(std::vector<SegmentGroup>& ellipse_sets, const cv::Mat& binary, int contour_id, const cv::Rect& contour_bbox);

    size_t computeStep(size_t contour_size);

    std::vector<double> computeTurn(const std::vector<cv::Point>& contour, size_t step);

    std::vector<ContourSegment> segmentContour(const std::vector<cv::Point>& contour, const std::vector<double>& turns);

    std::vector<double> computeAzimuths(const std::vector<double>& turns, int step);

    JoinDecision checkConvexSegmentsJoin(const ContourSegment& seg1,  const ContourSegment& seg2, const std::vector<double>& azimuths, const std::vector<cv::Point>& contour);

    std::vector<SegmentGroup> groupConvexSegments(const std::vector<ContourSegment>& all_segments,const std::vector<double>& azimuths, const std::vector<cv::Point>& contour);

    const Config& config;
    size_t segment_id_counter = 0;
};