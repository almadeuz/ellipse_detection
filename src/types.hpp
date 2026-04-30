#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <string>
#include <cmath>
#include <algorithm>

static int cont = 0;

// Параметры фильтрации и присоединения
struct Config {
    int min_axis_length = 6;
    double surface_threshold = 0.875;
    double area_ratio_threshold = 0.03;
    double convexity_tolerance = CV_PI / 18.0;
    double join_distance_factor = 0.25;
    double azimuth_diff = CV_PI / 18.0;
    double max_aspect_ratio = 4.0;

};

struct EllipseParams {
    cv::Point2f center;
    cv::Size2f size;
    float angle;
    int id = 0;  
    int contour_id = 0;
    
    EllipseParams() {}
    EllipseParams(cv::RotatedRect rect) : center(rect.center), size(rect.size), angle(rect.angle) {}

    double area() const { return CV_PI * size.width * size.height / 4.0; }

    cv::Rect getBoundingBox() const {
        return cv::RotatedRect(center, size, angle).boundingRect();
    }

    bool isValid() const { return size.width > 0 && size.height > 0; }
};

struct ContourSegment {
    std::vector<cv::Point> points;
    std::vector<size_t> contour_indices;
    size_t start_idx = 0;
    size_t end_idx = 0;
    bool is_convex = true;
    size_t segment_id = SIZE_MAX;
    
    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }

    size_t getContourIndex(size_t segment_index) const {
        if (contour_indices.empty() || segment_index >= contour_indices.size()) {
            return start_idx + segment_index;
        }
        return contour_indices[segment_index];
    }
};

struct SegmentGroup {
    std::vector<ContourSegment> segments;
    std::set<size_t> segment_ids;
    size_t total_points_count = 0;

    cv::Point search_start;
    cv::Point search_end;
    cv::Point search_pmid;
    cv::Point2f search_tangent_start;
    cv::Point2f search_tangent_end;

    void computeFullGeometry() {
        if (segments.empty()) return;
        search_start = segments.front().points.front();
        search_end = segments.back().points.back();
        search_tangent_start = computeTangent(true);
        search_tangent_end = computeTangent(false);
        search_pmid = getPointAt(total_points_count / 2);
    }
    
    void addSegment(const ContourSegment& seg) {
        segments.push_back(seg);
        total_points_count += seg.points.size();
        if (seg.segment_id != SIZE_MAX) segment_ids.insert(seg.segment_id);
    }

    void addSegments(const std::vector<ContourSegment>& segs) {
        for (const auto& seg : segs) {
            segments.push_back(seg);
            total_points_count += seg.points.size();
            if (seg.segment_id != SIZE_MAX) segment_ids.insert(seg.segment_id);
        }
        //computeFullGeometry();
    }

    void popSegment() {
        if (segments.empty()) return;
        total_points_count -= segments.back().points.size();
        if (segments.back().segment_id != SIZE_MAX) {
            segment_ids.erase(segments.back().segment_id);
        }
        segments.pop_back();
    }

    size_t pointCount() { return total_points_count; }

    std::vector<cv::Point> flattenPoints() {
        std::vector<cv::Point> all_pts;
        all_pts.reserve(total_points_count);
        for (auto& seg : segments) {
            all_pts.insert(all_pts.end(), seg.points.begin(), seg.points.end());
        }
        return all_pts;
    }

    cv::Point getPointAt(size_t global_idx) {
        size_t acc = 0;
        for (auto& seg : segments) {
            if (global_idx < acc + seg.points.size()) {
                return seg.points[global_idx - acc];
            }
            acc += seg.points.size();
        }
        return (!segments.empty()) ? segments.back().points.back() : cv::Point(0,0);
    }

private:
    cv::Point2f computeTangent(bool is_start) {
        if (segments.empty()) return cv::Point2f(1, 0); 

        auto& target_seg = is_start ? segments.front() : segments.back();
        auto& pts = target_seg.points;
        size_t n = pts.size();
        int window = static_cast<int>(n / 2);
        if (n < 3) return cv::Point2f(1, 0);

        cv::Point2f sum(0, 0);
        if (is_start) {
            cv::Point2f origin(pts.front());
            for (int i = 1; i <= window; ++i) {
                sum += cv::Point2f(pts[i]) - origin;
            }
        } else {
            cv::Point2f origin(pts.back());
            for (int i = 1; i <= window; ++i) {
                sum += cv::Point2f(pts[n - 1 - i]) - origin;
            }
        }

        double mag = static_cast<double>(cv::norm(sum));
        if (mag > 1e-5f) {
            return sum / mag;
        }
        return cv::Point2f(1, 0);
    }
};

struct DetectedEllipse {
    EllipseParams ellipse;
    std::vector<cv::Point> points; 
    std::set<size_t> segment_ids;
    cv::Rect bbox;
    double area = 0.0;
    double aspect_ratio = 0.0;
    double min_axis = 0.0;
    bool is_valid = false;
    bool marked_for_removal = false;

    void updateMetrics() {
        bbox = ellipse.getBoundingBox();
        area = ellipse.area();
        is_valid = ellipse.isValid();
        min_axis = std::min(ellipse.size.width, ellipse.size.height);
        aspect_ratio = (min_axis > 0) ? (std::max(ellipse.size.width, ellipse.size.height) / min_axis) : 0.0;
    }
};

struct JoinDecision {
    bool should_join;
    size_t best_end_idx;
    size_t best_start_idx;
    double best_az_diff;
};