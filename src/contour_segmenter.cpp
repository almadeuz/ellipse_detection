#include "contour_segmenter.hpp"
#include <queue>
#include <map>
#include <algorithm>

Segmenter::Segmenter(const Config& config) : config(config) {};

double Segmenter::getPointSide(const cv::Point2f& point, const cv::Point2f& P, const cv::Point2f& d) {
    cv::Point2f v = point - P;
    return d.x * v.y - d.y * v.x;
}

bool Segmenter::pointInSearchRegion(const cv::Point2f& point, const cv::Point& P1, const cv::Point& P2, const cv::Point& Pmid, const cv::Point2f& tangent1, const cv::Point2f& tangent2) {
    cv::Point2f chord_dir = cv::Point2f(P2) - cv::Point2f(P1);
    double side_pmid = getPointSide(cv::Point2f(Pmid), cv::Point2f(P1), chord_dir);
    double side_pt = getPointSide(point, cv::Point2f(P1), chord_dir);
    if (side_pmid * side_pt > 0) return false;
    
    double side_pt_l1 = getPointSide(point, cv::Point2f(P1), tangent1);
    if (side_pt_l1 > 0) return false;
    
    double side_pt_l2 = getPointSide(point, cv::Point2f(P2), tangent2);
    if (side_pt_l2 < 0) return false;
    
    return true;
}

bool Segmenter::groupInSearchRegion(const SegmentGroup& group_to_check, const SegmentGroup& reference_group, const cv::Mat& binary, const cv::Rect& contour_bbox) {
    if (reference_group.segments.empty() || group_to_check.segments.empty()) return false;

    cv::Point P1 = reference_group.search_start;
    cv::Point P2 = reference_group.search_end;
    cv::Point Pmid = reference_group.search_pmid;
    cv::Point2f t1 = reference_group.search_tangent_start;
    cv::Point2f t2 = reference_group.search_tangent_end;

    bool success = true;
    for (auto& seg : group_to_check.segments) {
        for (auto& pt : seg.points) {
            if (!pointInSearchRegion(cv::Point2f(pt), P1, P2, Pmid, t1, t2)) {
                success = false;
                break;
            }
        }
        if (!success) break;
    }

    return success;
}

SegmentGroup Segmenter::createGroupFromIndices(const std::vector<size_t>& indices, const std::vector<SegmentGroup>& source_sets) {
    SegmentGroup res = source_sets[indices[0]];
    for (size_t idx = 1; idx < indices.size(); ++idx) {
        res.addSegments(source_sets[indices[idx]].segments);
    }
    return res;
}

bool Segmenter::isIndexInRange(size_t idx, size_t start, size_t end, size_t total) {
    if (start <= end) return (idx >= start && idx <= end);
    return (idx >= start || idx <= end);
}

std::vector<SegmentGroup> Segmenter::mergeBySearchRegion(std::vector<SegmentGroup>& ellipse_sets, const cv::Mat& binary, int contour_id, const cv::Rect& contour_bbox) {
    size_t n = ellipse_sets.size();
    if (n == 0) return {};
    if (n == 1) return ellipse_sets;

    for (auto& group : ellipse_sets) group.computeFullGeometry();

    std::vector<std::vector<bool>> adj(n, std::vector<bool>(n, false));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (groupInSearchRegion(ellipse_sets[j], ellipse_sets[i], binary, contour_bbox) && 
    groupInSearchRegion(ellipse_sets[i], ellipse_sets[j], binary, contour_bbox)) {
    adj[i][j] = adj[j][i] = true;
}
        }
    }

    std::vector<SegmentGroup> final_results;

    for (size_t i = 0; i < n; ++i) {
        final_results.push_back(createGroupFromIndices({i}, ellipse_sets));

        for (size_t j = i + 1; j < n; ++j) {
            if (!adj[i][j]) continue;
            final_results.push_back(createGroupFromIndices({i, j}, ellipse_sets));

            for (size_t k = j + 1; k < n; ++k) {
                if (!adj[i][k] || !adj[j][k]) continue;
                final_results.push_back(createGroupFromIndices({i, j, k}, ellipse_sets));

                for (size_t l = k + 1; l < n; ++l) {
                    if (!adj[i][l] || !adj[j][l] || !adj[k][l]) continue;
                    final_results.push_back(createGroupFromIndices({i, j, k, l}, ellipse_sets));
                }
            }
        }
    }

    return final_results;
}

size_t Segmenter::computeStep(size_t contour_size) {
    return static_cast<size_t>(std::sqrt(contour_size)) / 2;
}

std::vector<double> Segmenter::computeTurn(const std::vector<cv::Point>& contour, size_t step) {
    size_t n = contour.size();
    std::vector<double> turns(n, 0.0);

    for (int i = 0; i < n; i++) {
        cv::Point2f A(contour[(i + step) % n]);
        cv::Point2f B(contour[i]);
        cv::Point2f C(contour[(i + n - step) % n]);
        cv::Point2f AB = B - A, BC = C - B;

        double na = cv::norm(AB);
        double nb = cv::norm(BC);

        if (na < 1e-9 || nb < 1e-9) continue;

        double dot = AB.x * BC.x + AB.y * BC.y;
        double cross = AB.x * BC.y - AB.y * BC.x;
        turns[i] = std::atan2(cross, dot);
    }

    return turns;
}

std::vector<ContourSegment> Segmenter::segmentContour(const std::vector<cv::Point>& contour, const std::vector<double>& turns) {
    std::vector<ContourSegment> segments;
    if (turns.empty()) return segments;
    
    ContourSegment cur;
    bool cur_convex = (turns[0] >= -config.convexity_tolerance);
    cur.start_idx = 0;
    cur.is_convex = cur_convex;
    
    for (size_t i = 0; i < contour.size(); i++) {
        bool is_convex = (turns[i] >= -config.convexity_tolerance);
        if (is_convex == cur_convex) {
            cur.points.push_back(contour[i]);
            cur.contour_indices.push_back(i);
            cur.end_idx = i;
        } else {
            if (cur.size() < 5) cur.is_convex = false;
            segments.push_back(cur);
            cur = ContourSegment();
            cur.points.push_back(contour[i]);
            cur.contour_indices.push_back(i);
            cur.start_idx = i;
            cur.end_idx = i;
            cur.is_convex = is_convex;
            cur_convex = is_convex;
        }
    }
    segments.push_back(cur);
    
    if (segments.size() >= 2 && segments.front().is_convex == segments.back().is_convex && segments.front().start_idx == 0) {
        ContourSegment merged = segments.back();
        merged.points.insert(merged.points.end(), segments.front().points.begin(), segments.front().points.end());
        merged.contour_indices.insert(merged.contour_indices.end(), segments.front().contour_indices.begin(), segments.front().contour_indices.end());
        merged.end_idx = segments.front().end_idx;
        segments.back() = merged;
        segments.erase(segments.begin());
    }
    
    return segments;
}

std::vector<double> Segmenter::computeAzimuths(const std::vector<double>& turns, int step) {
    std::vector<double> azimuths(turns.size(), 0.0);
    if (turns.empty()) return azimuths;
    
    double cum = 0.0;
    for (size_t i = 0; i < turns.size(); i++) {
        azimuths[i] = cum / step;
        cum += turns[i];
    }
    return azimuths;
}

JoinDecision Segmenter::checkConvexSegmentsJoin(const ContourSegment& seg1, const ContourSegment& seg2, const std::vector<double>& azimuths, const std::vector<cv::Point>& contour) {
    JoinDecision decision;
    decision.should_join = false;
    decision.best_az_diff = CV_PI;
    
    if (seg1.points.empty() || seg2.points.empty() || !seg1.is_convex || !seg2.is_convex)
        return decision;
    
    double max_dist = config.join_distance_factor * std::sqrt(contour.size());
    int s1 = seg1.points.size();
    int s2 = seg2.points.size();
    bool brk = false;
    
    for (int i = s1 - 1; i >= 0 && !brk; i--) {
        for (int j = 0; j < s2 && !brk; j++) {
            size_t idx1 = seg1.getContourIndex(i);
            size_t idx2 = seg2.getContourIndex(j);
            double diff = std::abs(azimuths[idx1] - azimuths[idx2]);
            double dist = cv::norm(contour[idx1] - contour[idx2]);
            if (dist > max_dist) continue;
            if (diff < config.azimuth_diff) {
                decision.best_az_diff  = diff;
                decision.best_end_idx   = idx1;
                decision.best_start_idx = idx2;
                brk = true;
            }
        }
    }
    
    decision.should_join = (decision.best_az_diff < config.azimuth_diff);
    return decision;
}

std::vector<SegmentGroup> Segmenter::groupConvexSegments(const std::vector<ContourSegment>& all_segments, const std::vector<double>& azimuths, const std::vector<cv::Point>& contour) {
    std::vector<ContourSegment> convex_segs;
    for (auto& seg : all_segments) if (seg.is_convex) convex_segs.push_back(seg);
    
    size_t n = convex_segs.size();
    if (n == 0) return {};
    std::vector<JoinDecision> next_joins(n);
    std::vector<bool> can_join_next(n, false);
    for (size_t i = 0; i < n; ++i) {
        next_joins[i] = checkConvexSegmentsJoin(convex_segs[i], convex_segs[(i + 1) % n], azimuths, contour);
        can_join_next[i] = next_joins[i].should_join;
    }

    size_t start_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!can_join_next[(i + n - 1) % n]) {
            start_idx = i;
            break;
        }
    }

    std::vector<SegmentGroup> results;
    std::vector<bool> visited(n, false);

    for (size_t i = 0; i < n; ++i) {
        size_t curr = (start_idx + i) % n;
        if (visited[curr]) continue;
        std::vector<size_t> chain = {curr};
        visited[curr] = true;

        while (can_join_next[chain.back()]) {
            size_t next = (chain.back() + 1) % n;
            if (visited[next]) break;
            chain.push_back(next);
            visited[next] = true;
        }

        ContourSegment merged;
        merged.segment_id = segment_id_counter++;
        merged.is_convex = true;

        for (size_t k = 0; k < chain.size(); ++k) {
            size_t idx = chain[k];
            const auto& s = convex_segs[idx];
            size_t limit_start = s.contour_indices.front();
            size_t limit_end   = s.contour_indices.back();

            if (k > 0) {
                limit_start = next_joins[chain[k-1]].best_start_idx;
            }
            if (can_join_next[idx] && (k + 1 < chain.size() || n == chain.size())) { 
                limit_end = next_joins[idx].best_end_idx;
            }

            for (size_t p = 0; p < s.points.size(); ++p) {
                size_t c_idx = s.contour_indices[p];
                if (isIndexInRange(c_idx, limit_start, limit_end, contour.size())) {
                    merged.points.push_back(s.points[p]);
                    merged.contour_indices.push_back(c_idx);
                }
            }
        }

        if (merged.points.size() >= 5) {
            merged.start_idx = merged.contour_indices.front();
            merged.end_idx = merged.contour_indices.back();
            SegmentGroup group;
            group.addSegment(merged);
            results.push_back(group);
        }
    }

    return results;
}

std::vector<DetectedEllipse> Segmenter::analyzeContour(const std::vector<cv::Point>& contour, const cv::Mat& binary, int contour_id, const cv::Rect& contour_bbox) {
    std::vector<DetectedEllipse> result;

    size_t step = computeStep(contour.size());
    std::vector<double> turns = computeTurn(contour, step);
    if (turns.empty()) return result;

    std::vector<ContourSegment> segments = segmentContour(contour, turns);
    if (segments.empty()) return result;
    
    std::vector<double> azimuths = computeAzimuths(turns, step);
    
    std::vector<SegmentGroup> ellipse_sets = groupConvexSegments(segments, azimuths, contour);
    
    std::vector<SegmentGroup> merged_sets = mergeBySearchRegion(ellipse_sets, binary, contour_id, contour_bbox);

    std::set<std::pair<size_t, size_t>> seen_segs;
    for (auto& eset : merged_sets) {
        for (auto& seg : eset.segments) {
            auto key = std::make_pair(seg.start_idx, seg.end_idx);
            if (!seen_segs.count(key)) {
                // result.segments_with_ids.push_back(seg);
                seen_segs.insert(key);
            }
        }
    }
    
    std::set<std::set<size_t>> existing_sigs;
    for (auto& eset : merged_sets) existing_sigs.insert(eset.segment_ids);
    
    for (auto& orig : ellipse_sets) {
        if (!existing_sigs.count(orig.segment_ids) && orig.pointCount() >= 5) {
            merged_sets.push_back(orig);
            existing_sigs.insert(orig.segment_ids);
            for (auto& seg : orig.segments) {
                auto key = std::make_pair(seg.start_idx, seg.end_idx);
                if (!seen_segs.count(key)) {
                    // result.segments_with_ids.push_back(seg);
                    seen_segs.insert(key);
                }
            }
        }
    }


    cv::RotatedRect ellipse = cv::fitEllipseAMS(contour);

    double max_allowed_size = static_cast<double>(std::max(binary.cols, binary.rows) * 2);
    
    if (ellipse.size.width > 0 && ellipse.size.height > 0 && ellipse.size.width < max_allowed_size && ellipse.size.height < max_allowed_size) {
        DetectedEllipse ell;
        ell.ellipse = EllipseParams(ellipse);
        ell.points = std::move(contour);
        
        ell.updateMetrics();
        result.push_back(ell);
    }

    for (auto& eset : merged_sets) {
        if (eset.pointCount() < 5) continue;
        try {
            std::vector<cv::Point> flat_pts = eset.flattenPoints();
            cv::RotatedRect ellipse = cv::fitEllipseAMS(flat_pts);

            double max_allowed_size = static_cast<double>(std::max(binary.cols, binary.rows) * 2);
            
            if (ellipse.size.width > 0 && ellipse.size.height > 0 && ellipse.size.width < max_allowed_size && ellipse.size.height < max_allowed_size) {
                DetectedEllipse ell;
                ell.ellipse = EllipseParams(ellipse);
                ell.points = std::move(flat_pts);
                ell.segment_ids = eset.segment_ids;
                
                ell.updateMetrics();
                result.push_back(ell);
            }
        } catch (cv::Exception& e) {
        }
    }


    return result;
}