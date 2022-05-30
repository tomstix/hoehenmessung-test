#include <iostream>
#include <omp.h>

#include <librealsense2/rs.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

#include "realsense_tools.h"

RealsensePCLProvider::RealsensePCLProvider(  int color_width_, int color_height_, int depth_width_, int depth_height_, int fps,
                                            rs2_format color_format, rs2_format depth_format, rs2_rs400_visual_preset visual_preset)
                                            :color_width(color_width_), color_height(color_height_)
{
    std::cout << "Starting Realsense Camera" << std::endl;
    cfg.enable_stream(RS2_STREAM_DEPTH, depth_width_, depth_height_, depth_format, fps);
    cfg.enable_stream(RS2_STREAM_COLOR, color_width_, color_height_, color_format, fps);
    pipe_profile = pipe.start(cfg);
    auto sensor = pipe_profile.get_device().first<rs2::depth_sensor>();
    sensor.set_option(RS2_OPTION_VISUAL_PRESET, visual_preset);
    auto depth_stream = pipe_profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = depth_stream.get_intrinsics();
    std::vector<std::vector<double>> mat{   {   intrinsics.fx,    0,              intrinsics.ppx  },
                                            {   0,                intrinsics.fy,  intrinsics.ppy  },
                                            {   0,                0,              1               }};
    intrinsic_matrix = mat;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr RealsensePCLProvider::get_pcl_point_cloud(unsigned int wait)
{
    frames = pipe.wait_for_frames(wait);
    depth_frame = frames.get_depth_frame();
    color_frame = frames.get_color_frame();

    points = pc.calculate(depth_frame);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto &p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

std::unique_ptr<rs2::frame> RealsensePCLProvider::get_color_frame() const
{
    auto frame_ptr = std::make_unique<rs2::frame>(color_frame);
    return frame_ptr;
}

std::unique_ptr<std::vector<std::vector<double>>> RealsensePCLProvider::get_intrinsic_matrix() const
{
    auto mat_ptr = std::make_unique<std::vector<std::vector<double>>>(intrinsic_matrix);
    return mat_ptr;
}

void RealsensePCLProvider::calculate_extrinsic_matrix(const Eigen::VectorXf &planeCoeffs)
{
    Eigen::Vector3f y_vector = {0.0, -1.0, 0.0};
    Eigen::Vector3f rot_vector = y_vector.cross(planeCoeffs.head<3>());
    std::cout << rot_vector << std::endl;
    Eigen::Matrix4f mat;
    extrinsic_matrix = mat;
}

std::unique_ptr<Eigen::Matrix4f> RealsensePCLProvider::get_extrinsic_matrix() const
{
    auto mat_ptr = std::make_unique<Eigen::Matrix4f>(extrinsic_matrix);
    return mat_ptr;
}