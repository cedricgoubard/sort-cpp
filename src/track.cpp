#include <sort_obj_track/track.h>

Track::Track(int id_, float score_, int n_classes, int mem_size) : kf_(8, 4), class_id_(id_), score_(score_), mem(ClassObsMemory(n_classes, mem_size))
{

    /*** Define constant velocity model ***/
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    kf_.F_ << 1, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1;

    // Give high uncertainty to the unobservable initial velocities
    kf_.P_ << 10, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 0, 0, 0, 0,
        0, 0, 0, 0, 10000, 0, 0, 0,
        0, 0, 0, 0, 0, 10000, 0, 0,
        0, 0, 0, 0, 0, 0, 10000, 0,
        0, 0, 0, 0, 0, 0, 0, 10000;

    kf_.H_ << 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0;

    kf_.Q_ << 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0.01, 0, 0, 0,
        0, 0, 0, 0, 0, 0.01, 0, 0,
        0, 0, 0, 0, 0, 0, 0.0001, 0,
        0, 0, 0, 0, 0, 0, 0, 0.0001;

    kf_.R_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 10, 0,
        0, 0, 0, 10;

}

// Get predicted locations from existing trackers
// dt is time elapsed between the current and previous measurements
void Track::Predict()
{
    kf_.Predict();

    // hit streak count will be reset
    if (coast_cycles_ > 0)
    {
        hit_streak_ = 0;
    }
    // accumulate coast cycle count
    coast_cycles_++;
}

// Update matched trackers with assigned detections
void Track::Update(RectWithClass det)
{
    // get measurement update, reset coast cycle count
    coast_cycles_ = 0;
    // accumulate hit streak count
    hit_streak_++;

    // observation - center_x, center_y, area, ratio
    Eigen::VectorXd observation = ConvertBboxToObservation(det.rect);
    kf_.Update(observation);
    this->mem.AddObs(det.class_id, det.score);
    this->class_id_ = this->mem.class_id_;
    this->score_ = this->mem.score_;
}

// Create and initialize new trackers for unmatched detections, with initial bounding box
void Track::Init(const cv::Rect &bbox)
{
    kf_.x_.head(4) << ConvertBboxToObservation(bbox);
    hit_streak_++;
}

/**
 * Returns the current bounding box estimate
 * @return
 */
cv::Rect Track::GetStateAsBbox() const
{
    return ConvertStateToBbox(kf_.x_);
}

/**
 * Returns the current bounding box estimate
 * @return
 */
vision_msgs::Detection2D Track::GetStateAsROS2DDetection() const
{
    return ConvertStateToROS2DDetection(kf_.x_);
}

float Track::GetNIS() const
{
    return kf_.NIS_;
}

/**
 * Takes a bounding box in the form [x, y, width, height] and returns z in the form
 * [x, y, s, r] where x,y is the centre of the box and s is the scale/area and r is
 * the aspect ratio
 *
 * @param bbox
 * @return
 */
Eigen::VectorXd Track::ConvertBboxToObservation(const cv::Rect &bbox) const
{
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(4);
    auto width = static_cast<float>(bbox.width);
    auto height = static_cast<float>(bbox.height);
    float center_x = bbox.x + width / 2;
    float center_y = bbox.y + height / 2;
    observation << center_x, center_y, width, height;
    return observation;
}

/**
 * Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
 * [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
 *
 * @param state
 * @return
 */
cv::Rect Track::ConvertStateToBbox(const Eigen::VectorXd &state) const
{
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    auto width = std::max(0, static_cast<int>(state[2]));
    auto height = std::max(0, static_cast<int>(state[3]));
    auto tl_x = static_cast<int>(state[0] - width / 2.0);
    auto tl_y = static_cast<int>(state[1] - height / 2.0);
    cv::Rect rect(cv::Point(tl_x, tl_y), cv::Size(width, height));
    return rect;
}

vision_msgs::Detection2D Track::ConvertStateToROS2DDetection(const Eigen::VectorXd &state) const
{
    vision_msgs::Detection2D detection;
    detection.bbox.center.x = state[0];
    detection.bbox.center.y = state[1];
    detection.bbox.size_x = state[2];
    detection.bbox.size_y = state[3];
    vision_msgs::ObjectHypothesisWithPose ob = vision_msgs::ObjectHypothesisWithPose();
    ob.id = class_id_;
    ob.score = score_;
    detection.results.push_back(ob);
    return detection;
}

ClassObsMemory::ClassObsMemory(int n_classes, int memory_length_) : memory_length(memory_length_), total_sum(0)
{
    obs = std::vector<std::vector<float>>(n_classes, std::vector<float>(memory_length, 0));
    obs_idx = std::vector<int>(n_classes, 0);
    obs_sums = std::vector<float>(n_classes, 0);
}

void ClassObsMemory::AddObs(const int class_id, const float score){
    // Update sums
    obs_sums[class_id] -= (obs[class_id][obs_idx[class_id]] - score);
    total_sum -= (obs[class_id][obs_idx[class_id]] - score);

    // Store observation
    obs[class_id][obs_idx[class_id]] = score;
    
    // Update rolling index
    obs_idx[class_id] = (obs_idx[class_id] + 1) % memory_length;
    UpdateClass();
}

void ClassObsMemory::UpdateClass() {
    float max_score = 0;
    int max_cls = -1;
    for (int i = 0; i < obs_sums.size(); i++){
        if (obs_sums[i] / total_sum > max_score){
            max_score = obs_sums[i] / total_sum;
            max_cls = i;
        }
    }
    class_id_ = max_cls;
    score_ = max_score;
}

