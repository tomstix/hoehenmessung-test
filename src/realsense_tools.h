#pragma once

class RealsensePCLProvider
{
public:
    RealsensePCLProvider(  int color_width_, int color_height_, int depth_width_, int depth_height_, int fps,
                        rs2_format color_format = RS2_FORMAT_YUYV, rs2_format depth_format = RS2_FORMAT_Z16, rs2_rs400_visual_preset visual_preset = RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);

    pcl::PointCloud<pcl::PointXYZ>::Ptr get_pcl_point_cloud(unsigned int wait = 15000U);
    std::vector<std::vector<double>> get_intrinsic_matrix();
    boost::shared_ptr<rs2::frame> get_color_frame();
    std::vector<float> rvec;
    std::vector<float> tvec;
    std::vector<float> distortion;
    int color_width;
    int color_height;

private:
    rs2::config cfg;
    rs2::pipeline pipe;
    rs2::pipeline_profile pipe_profile;
    rs2::frameset frames;
    rs2_format rgb_format;
    rs2_format depth_format;
    rs2::pointcloud pc;
    rs2::points points;
    rs2::frame depth_frame;
    rs2::frame color_frame;

    std::vector<std::vector<double>> intrinsic_matrix;
};