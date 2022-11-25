#pragma once

#include <vision_msgs/Detection2D.h>
#include <opencv2/core.hpp>
#include <sort_obj_track/kalman_filter.h>

class Track {
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void Init(const cv::Rect& bbox);
    void Predict();
    void Update(const cv::Rect& bbox);
    cv::Rect GetStateAsBbox() const;
    vision_msgs::Detection2D GetStateAsROS2DDetection() const;
    float GetNIS() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

private:
    Eigen::VectorXd ConvertBboxToObservation(const cv::Rect& bbox) const;
    cv::Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;
    vision_msgs::Detection2D ConvertStateToROS2DDetection(const Eigen::VectorXd &state) const;

    KalmanFilter kf_;
};
