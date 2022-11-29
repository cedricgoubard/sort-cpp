#pragma once

#include <vector>

#include <ros/ros.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <opencv2/core.hpp>
#include <sort_obj_track/kalman_filter.h>

struct  RectWithClass {
    cv::Rect rect;
    int class_id;
    float score;
};

class ClassObsMemory{

    public:
        int class_id_;
        float score_;
        ClassObsMemory(int n_classes, int memory_length_);
        void AddObs(const int class_id, const float score);
        void UpdateClass();

    private: 
        std::vector<std::vector<float>> obs;
        std::vector<float> obs_sums;
        float total_sum;
        std::vector<int> obs_idx;
        int memory_length;
};

class Track {
public:
    int class_id_;
    float score_;
    std::vector<std::vector<float>> class_distribution_;
    // Constructor
    // Track();
    Track(int id_, float score_, int n_classes, int mem_size);

    // Destructor
    ~Track() = default;

    void Init(const cv::Rect& bbox);
    void Predict();
    void Update(const RectWithClass det);
    cv::Rect GetStateAsBbox() const;
    vision_msgs::Detection2D GetStateAsROS2DDetection() const;
    float GetNIS() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

private:
    Eigen::VectorXd ConvertBboxToObservation(const cv::Rect& bbox) const;
    cv::Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;
    vision_msgs::Detection2D ConvertStateToROS2DDetection(const Eigen::VectorXd &state) const;

    KalmanFilter kf_;
    ClassObsMemory mem;
    
};
