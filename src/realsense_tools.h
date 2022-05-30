#pragma once

class RealsensePCLProvider
{
public:
    RealsensePCLProvider(  int color_width_, int color_height_, int depth_width_, int depth_height_, int fps,
                            rs2_format color_format = RS2_FORMAT_YUYV, rs2_format depth_format = RS2_FORMAT_Z16,
                            rs2_rs400_visual_preset visual_preset = RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);

    pcl::PointCloud<pcl::PointXYZ>::Ptr get_pcl_point_cloud(unsigned int wait = 15000U);
    std::unique_ptr<rs2::frame> get_color_frame() const;
    std::unique_ptr<std::vector<std::vector<double>>> get_intrinsic_matrix() const;

    void calculate_extrinsic_matrix(const Eigen::VectorXf &planeCoeffs);
    std::unique_ptr<Eigen::Matrix4f> get_extrinsic_matrix() const;

    const int color_width;
    const int color_height;
    const std::vector<float> rvec = {0.0, 0.0, 0.0};
    const std::vector<float> tvec = {0.0, 0.0, 0.0};
    const std::vector<float> distortion;

private:
    rs2::config cfg;
    rs2::pipeline pipe;
    rs2::pipeline_profile pipe_profile;
    rs2::frameset frames;
    rs2::pointcloud pc;
    rs2::points points;
    rs2::frame depth_frame;
    rs2::frame color_frame;
    std::vector<std::vector<double>> intrinsic_matrix;
    Eigen::Matrix4f extrinsic_matrix;
};