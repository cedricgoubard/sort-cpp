#pragma once

#include <map>
#include <opencv2/core.hpp>

#include <sort_obj_track/track.h>
#include <sort_obj_track/munkres.h>
#include <sort_obj_track/utils.h>

struct  RectWithClass {
    cv::Rect rect;
    int class_id;
    float score;
};

class Tracker {
public:
    Tracker();
    ~Tracker() = default;

    static float CalculateIou(const cv::Rect& det, const Track& track);

    static void HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,
                           size_t nrows, size_t ncols,
                           std::vector<std::vector<float>>& association);

/**
 * Assigns detections to tracked object (both represented as bounding boxes)
 * Returns 2 lists of matches, unmatched_detections
 * @param detection
 * @param tracks
 * @param matched
 * @param unmatched_det
 * @param iou_threshold
 */
    static void AssociateDetectionsToTrackers(const std::vector<RectWithClass>& detection,
                                       std::map<int, Track>& tracks,
                                       std::map<int, RectWithClass>& matched,
                                       std::vector<RectWithClass>& unmatched_det,
                                       float iou_threshold = 0.3);

    void Run(const std::vector<RectWithClass>& detections);

    std::map<int, Track> GetTracks();

private:
    // Hash-map between ID and corresponding tracker
    std::map<int, Track> tracks_;

    // Assigned ID for each bounding box
    int id_;
};