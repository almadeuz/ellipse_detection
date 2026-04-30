#include "ellipse_filter.hpp"
#include <algorithm>

Filter::Filter(const Config& config) : config(config) {}

std::vector<cv::Point2f> Filter::getEllipsePolygon(const EllipseParams& e, double& poly_area) {
    std::vector<cv::Point> pts_int;
    cv::ellipse2Poly(e.center, cv::Size(e.size.width/2, e.size.height/2), cvRound(e.angle), 0, 360, 1, pts_int);
    std::vector<cv::Point2f> pts(pts_int.begin(), pts_int.end());
    poly_area = cv::contourArea(pts);
    return pts;
}

double Filter::getEllipsesIntersectionArea(const EllipseParams& e1, const EllipseParams& e2, double& area1, double& area2) {
    cv::Rect r1 = e1.getBoundingBox();
    cv::Rect r2 = e2.getBoundingBox();
    if ((r1 & r2).empty()) {
        area1 = e1.area();
        area2 = e2.area();
        return 0.0;
    }
    std::vector<cv::Point2f> p1 = getEllipsePolygon(e1, area1);
    std::vector<cv::Point2f> p2 = getEllipsePolygon(e2, area2);
    std::vector<cv::Point2f> inter_poly;
    return cv::intersectConvexConvex(p1, p2, inter_poly);
}

double Filter::computeEllipseIntersection(const EllipseParams& e1, const EllipseParams& e2) {
    double a1, a2;
    double inter_area = getEllipsesIntersectionArea(e1, e2, a1, a2);
    if (inter_area == 0.0) return 0.0;
    double mn = std::min(a1, a2);
    return (mn > 0) ? (inter_area / mn) : 0.0;
}

double Filter::computeSurfaceRatio(const EllipseParams& ellipse, const cv::Mat& binary) {
    cv::Rect bbox = ellipse.getBoundingBox() & cv::Rect(0, 0, binary.cols, binary.rows);
    if (bbox.empty()) return 0.0;

    cv::Mat mask = cv::Mat::zeros(bbox.size(), CV_8UC1);
    cv::Point2f cs(ellipse.center.x - bbox.x, ellipse.center.y - bbox.y);
    cv::ellipse(mask, cv::RotatedRect(cs, ellipse.size, ellipse.angle), cv::Scalar(255), cv::FILLED);

    double ratio = cv::mean(binary(bbox), mask).val[0] / 255.0;
    return ratio;
}

cv::Mat Filter::fillLargestContour(const cv::Mat& roi, int W, int H) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(roi, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat filled = cv::Mat::zeros(H, W, CV_8UC1);
    if (cs.empty()) return filled;

    size_t bi = 0;
    for (size_t i = 1; i < cs.size(); ++i)
        if (cs[i].size() > cs[bi].size()) bi = i;
    cv::drawContours(filled, cs, (int)bi, cv::Scalar(255), cv::FILLED);
    return filled;
}

double Filter::computeBoundaryIoU(const cv::Mat& predicted, const cv::Mat& ground_truth, int bandwidth) {
    if (predicted.empty() || ground_truth.empty()) return 0.0;

    cv::Mat pred_bin = predicted > 0;
    cv::Mat gt_bin = ground_truth > 0;

    cv::Size ksize(2 * bandwidth + 1, 2 * bandwidth + 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, ksize);

    cv::Mat pred_boundary, gt_boundary;
    cv::morphologyEx(pred_bin, pred_boundary, cv::MORPH_GRADIENT, kernel);
    cv::morphologyEx(gt_bin,   gt_boundary,   cv::MORPH_GRADIENT, kernel);
    pred_boundary = pred_boundary > 0;
    gt_boundary = gt_boundary   > 0;

    cv::Mat tp_mask, fp_outside_mask, fn_mask;
    cv::bitwise_and(pred_boundary, gt_boundary,  tp_mask);
    cv::bitwise_and(pred_boundary, ~gt_bin,       fp_outside_mask);
    cv::bitwise_and(gt_boundary,   ~pred_boundary, fn_mask);

    double tp = (double)cv::countNonZero(tp_mask);
    double fp_out = (double)cv::countNonZero(fp_outside_mask);
    double fn = (double)cv::countNonZero(fn_mask);

    return (tp + fp_out + fn > 0) ? tp / (tp + fp_out + fn) : 0.0;
}

double Filter::computeBoundaryPrecision(const cv::Mat& predicted, const cv::Mat& ground_truth, int bandwidth) {
    if (predicted.empty() || ground_truth.empty()) return 0.0;

    cv::Mat pred_bin = predicted > 0;
    cv::Mat gt_bin = ground_truth > 0;

    cv::Size ksize(2 * bandwidth + 1, 2 * bandwidth + 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, ksize);

    cv::Mat pred_boundary;
    cv::morphologyEx(pred_bin, pred_boundary, cv::MORPH_GRADIENT, kernel);
    pred_boundary = pred_boundary > 0;

    cv::Mat gt_boundary_dilated;
    cv::morphologyEx(gt_bin, gt_boundary_dilated, cv::MORPH_GRADIENT, kernel);
    gt_boundary_dilated = gt_boundary_dilated > 0;

    cv::Mat matched;
    cv::bitwise_and(pred_boundary, gt_boundary_dilated, matched);

    double ellipse_perimeter = (double)cv::countNonZero(pred_boundary);
    double matched_px = (double)cv::countNonZero(matched);

    return (ellipse_perimeter > 0) ? matched_px / ellipse_perimeter : 0.0;
}

cv::Mat Filter::buildEllipseMask(const DetectedEllipse& cand, const cv::Rect& roi_offset, int W, int H) {
    cv::Mat m = cv::Mat::zeros(H, W, CV_8UC1);
    cv::Point2f local_c(cand.ellipse.center.x - roi_offset.x, cand.ellipse.center.y - roi_offset.y);
    cv::ellipse(m, cv::RotatedRect(local_c, cand.ellipse.size, cand.ellipse.angle), cv::Scalar(255), cv::FILLED);

    return m;
}

std::vector<DetectedEllipse> Filter::filterByMinAxis(const std::vector<DetectedEllipse>& src) {
    std::vector<DetectedEllipse> out;
    out.reserve(src.size());
    for (const auto& it : src)
        if (it.is_valid && it.min_axis >= config.min_axis_length)
            out.push_back(it);

    return out;
}

std::vector<DetectedEllipse> Filter::filterByAspectRatio(const std::vector<DetectedEllipse>& src) {
    std::vector<DetectedEllipse> out;
    out.reserve(src.size());
    for (const auto& it : src)
        if (it.aspect_ratio <= config.max_aspect_ratio)
            out.push_back(it);

    return out;
}

std::vector<DetectedEllipse> Filter::filterBySurfaceRatio(const std::vector<DetectedEllipse>& src,
                                                          const cv::Mat& binary) {
    std::vector<DetectedEllipse> out;
    out.reserve(src.size());
    for (const auto& it : src)
        if (computeSurfaceRatio(it.ellipse, binary) >= config.surface_threshold)
            out.push_back(it);

    return out;
}

std::vector<DetectedEllipse> Filter::filterByAreaRatio(const std::vector<DetectedEllipse>& src) {
    double max_area = 0;
    for (const auto& it : src)
        if (it.area > max_area) max_area = it.area;

    std::vector<DetectedEllipse> out;
    out.reserve(src.size());
    for (const auto& it : src)
        if (max_area > 0 && it.area >= config.area_ratio_threshold * max_area)
            out.push_back(it);
    return out;
}

std::vector<DetectedEllipse> Filter::filterFitness(const std::vector<DetectedEllipse>& ellipses_data,
                                                   const cv::Mat& binary, const cv::Rect& bbox) {
    if (ellipses_data.empty()) return {};

    auto step0 = filterByMinAxis(ellipses_data);
    if (step0.empty()) return {};

    auto step1 = filterByAspectRatio(step0);
    if (step1.empty()) return {};

    auto step2 = filterBySurfaceRatio(step1, binary);
    if (step2.empty()) return {};

    return filterByAreaRatio(step2);
}

std::vector<DetectedEllipse> Filter::filterNMS(const std::vector<DetectedEllipse>& candidates, const cv::Mat& binary, double iou_threshold, double beta) {
    if (candidates.size() <= 1) return candidates;

    struct Scored {
        DetectedEllipse data;
        double f_score;
        cv::Mat mask;
    };

    cv::Rect global_bbox = candidates[0].bbox;
    for (const auto& c : candidates) global_bbox |= c.bbox;
    global_bbox &= cv::Rect(0, 0, binary.cols, binary.rows);

    cv::Mat roi_binary = binary(global_bbox);
    double total_blood = std::max(1.0, (double)cv::countNonZero(roi_binary));
    double beta_sq = beta * beta;

    std::vector<Scored> scored_list;
    for (const auto& cand : candidates) {
        cv::Mat e_mask = buildEllipseMask(cand, global_bbox, global_bbox.width, global_bbox.height);

        cv::Mat b_mask;
        cv::bitwise_and(e_mask, roi_binary, b_mask);
        double tp = cv::countNonZero(b_mask);
        double area_px = cv::countNonZero(e_mask);
        double fp = area_px - tp;
        double precision = tp / (tp + fp + 1e-6);
        double recall = tp / total_blood;
        double f05 = (1 + beta_sq) * (precision * recall) /
                     ((beta_sq * precision) + recall + 1e-6);

        scored_list.push_back({cand, f05, e_mask});
    }

    std::sort(scored_list.begin(), scored_list.end(), [](const Scored& a, const Scored& b) {
        return a.f_score > b.f_score;
    });

    std::vector<bool> removed(scored_list.size(), false);
    std::vector<DetectedEllipse> winners;

    for (size_t i = 0; i < scored_list.size(); ++i) {
        if (removed[i]) continue;
        winners.push_back(scored_list[i].data);

        for (size_t j = i + 1; j < scored_list.size(); ++j) {
            if (removed[j]) continue;
            cv::Mat intersection, union_mask;
            cv::bitwise_and(scored_list[i].mask, scored_list[j].mask, intersection);
            cv::bitwise_or(scored_list[i].mask,  scored_list[j].mask, union_mask);
            double iou = (double)cv::countNonZero(intersection) /
                         (cv::countNonZero(union_mask) + 1e-6);
            if (iou > iou_threshold) removed[j] = true;
        }
    }
    return winners;
}

std::vector<DetectedEllipse> Filter::removeOverlappedEllipses(const std::vector<Cand>& selected, const cv::Mat& target_blob, int W, int H, double overlap_threshold) {
    std::vector<bool> keep(selected.size(), true);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < selected.size(); ++i) {
            if (!keep[i]) continue;
            cv::Mat si_blob;
            cv::bitwise_and(selected[i].mask, target_blob, si_blob);
            int si_cov_area = cv::countNonZero(si_blob);

            cv::Mat others_mask = cv::Mat::zeros(H, W, CV_8UC1);
            for (size_t j = 0; j < selected.size(); ++j) {
                if (i == j || !keep[j]) continue;
                cv::Mat tmp;
                cv::bitwise_and(selected[j].mask, target_blob, tmp);
                cv::bitwise_or(others_mask, tmp, others_mask);
            }
            cv::Mat overlap;
            cv::bitwise_and(si_blob, others_mask, overlap);
            if (si_cov_area > 0 && (double)cv::countNonZero(overlap) / si_cov_area > overlap_threshold) {
                keep[i] = false;
                changed = true;
            }
        }
    }

    std::vector<DetectedEllipse> result;
    for (size_t i = 0; i < selected.size(); ++i)
        if (keep[i]) result.push_back(selected[i].data);
    return result;
}

std::vector<DetectedEllipse> Filter::applyGreedy(const std::vector<DetectedEllipse>& candidates, const cv::Mat& binary, const cv::Rect& contour_bbox) {
    if (candidates.empty()) return {};

    cv::Rect safe_roi = contour_bbox & cv::Rect(0, 0, binary.cols, binary.rows);
    if (safe_roi.empty()) return {};

    int W = safe_roi.width;
    int H = safe_roi.height;

    cv::Mat target_blob = fillLargestContour(binary(safe_roi), W, H);
    int total_blob_area = cv::countNonZero(target_blob);
    if (total_blob_area <= 0) return {};

    struct Scored {
        DetectedEllipse data;
        cv::Mat mask;
        double b_iou;
        double combined;
    };
    std::vector<Scored> scored_list;

    for (const auto& cand : candidates) {
        cv::Mat m = buildEllipseMask(cand, safe_roi, W, H);
        double b_iou = computeBoundaryIoU(m, target_blob, 2);
        double bp_score = computeBoundaryPrecision(m, target_blob, 2);
        double combined = b_iou * bp_score;
        scored_list.push_back({cand, m, b_iou, combined});
    }

    std::sort(scored_list.begin(), scored_list.end(), [](const Scored& a, const Scored& b) {
        return a.combined > b.combined;
    });

    std::vector<Cand> pool;
    for (const auto& s : scored_list) {
        Cand pc;
        pc.data = s.data;
        pc.mask = s.mask;
        pc.area = cv::countNonZero(pc.mask);
        pc.combined = s.combined;
        pool.push_back(std::move(pc));
    }

    cv::Mat covered_mask = cv::Mat::zeros(H, W, CV_8UC1);
    std::vector<Cand> selected_it;
    const double beta = 0.2;
    const double beta_sq = beta * beta;

    for (size_t iter = 0; iter < pool.size(); ++iter) {
        cv::Mat residual;
        cv::bitwise_and(target_blob, ~covered_mask, residual);
        int residual_area = cv::countNonZero(residual);
        if (residual_area < total_blob_area * 0.01) break;

        int best_idx = -1;
        double best_score = -1.0;

        for (size_t i = 0; i < pool.size(); ++i) {
            if (pool[i].used) continue;

            cv::Mat intersection;
            cv::bitwise_and(pool[i].mask, residual, intersection);
            double tp_dyn = (double)cv::countNonZero(intersection);

            double fp_dyn = (double)pool[i].area - tp_dyn;
            double precision = tp_dyn / (tp_dyn + fp_dyn + 1e-6);
            double recall = tp_dyn / (double)total_blob_area;
            double f_score = (1.0 + beta_sq) * (precision * recall) /
                               ((beta_sq * precision) + recall + 1e-10);
            if (f_score < 0.20) continue;

            double final_score = f_score * pool[i].combined;
            if (final_score > best_score) { best_score = final_score; best_idx = (int)i; }
        }

        if (best_idx == -1) break;
        pool[best_idx].used = true;
        selected_it.push_back(pool[best_idx]);

        cv::Mat just_added;
        cv::bitwise_and(pool[best_idx].mask, target_blob, just_added);
        cv::bitwise_or(covered_mask, just_added, covered_mask);
    }

    return removeOverlappedEllipses(selected_it, target_blob, W, H, 0.8);
}

void Filter::refineSingleEllipse(DetectedEllipse& det, const cv::Mat& binary) {
    if (!det.is_valid) return;

    float xc = det.ellipse.center.x;
    float yc = det.ellipse.center.y;
    float a = det.ellipse.size.width  / 2.0f;
    float b = det.ellipse.size.height / 2.0f;
    float angle_rad = det.ellipse.angle * (float)CV_PI / 180.0f;
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);

    std::vector<cv::Point2f> edge_points;
    const int samples = 72;

    for (int i = 0; i < samples; ++i) {
        float t = 2.0f * (float)CV_PI * i / samples;
        float cos_t = std::cos(t);
        float sin_t = std::sin(t);
        float ex = a * cos_t * cos_a - b * sin_t * sin_a + xc;
        float ey = a * cos_t * sin_a + b * sin_t * cos_a + yc;
        float vx = ex - xc;
        float vy = ey - yc;
        float R = std::sqrt(vx*vx + vy*vy);
        if (R < 1e-6f) continue;

        float nx = vx / R;
        float ny = vy / R;

        float max_search_d = R * 1.01f;
        cv::Point2f found_edge(-1, -1);

        for (float d = 1.0f; d <= max_search_d; d += 1.0f) {
            int px = cvRound(xc + nx * d);
            int py = cvRound(yc + ny * d);
            if (px < 0 || py < 0 || px >= binary.cols || py >= binary.rows) break;

            if (binary.at<uchar>(py, px) > 0) {

                int npx = cvRound(xc + nx * (d + 2));
                int npy = cvRound(yc + ny * (d + 2));
                bool is_edge = (npx < 0 || npy < 0 ||
                                npx >= binary.cols || npy >= binary.rows ||
                                binary.at<uchar>(npy, npx) == 0);
                if (is_edge)
                    found_edge = cv::Point2f((float)px, (float)py);
            } else if (found_edge.x != -1) {
                break;
            }
        }

        if (found_edge.x != -1)
            edge_points.push_back(found_edge);
    }

    if (edge_points.size() >= 20) {
        try {
            cv::RotatedRect new_rect = cv::fitEllipseAMS(edge_points);
            float dist_moved = (float)cv::norm(new_rect.center - det.ellipse.center);
            if (dist_moved < a) {
                det.ellipse = EllipseParams(new_rect);
                det.updateMetrics();
                det.points.clear();
                for (const auto& p : edge_points)
                    det.points.push_back(cv::Point((int)p.x, (int)p.y));
            }
        } catch (...) {}
    }
}

std::vector<DetectedEllipse> Filter::refineResults(const std::vector<DetectedEllipse>& accepted, const cv::Mat& binary) {
    std::vector<DetectedEllipse> refined_list = accepted;
    for (auto& det : refined_list)
        refineSingleEllipse(det, binary);
    return refined_list;
}