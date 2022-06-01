#include <iostream>
#include <omp.h>

#include <librealsense2/rs.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/real_sense_2_grabber.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "realsense_tools.h"

int main(int argc, char **argv)
try
{
    RealsensePCLProvider rs(1280, 720, 1280, 720, 30, RS2_FORMAT_BGR8);

    cv::Mat intrinsicMat(3, 3, cv::DataType<double>::type); // Intrisic matrix
    intrinsicMat.at<double>(0, 0) = rs.get_intrinsic_matrix()->at(0).at(0);
    intrinsicMat.at<double>(1, 0) = 0;
    intrinsicMat.at<double>(2, 0) = 0;
    intrinsicMat.at<double>(0, 1) = 0;
    intrinsicMat.at<double>(1, 1) = rs.get_intrinsic_matrix()->at(1).at(1);
    intrinsicMat.at<double>(2, 1) = 0;
    intrinsicMat.at<double>(0, 2) = rs.get_intrinsic_matrix()->at(0).at(2);
    intrinsicMat.at<double>(1, 2) = rs.get_intrinsic_matrix()->at(1).at(2);
    intrinsicMat.at<double>(2, 2) = 1;

    // create OpenCV window
    const auto window_name = "Display Image";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    int min_distance = 20;
    int min_distance_max = 1500;
    cv::createTrackbar("min. Z Distance (cm)", window_name, &min_distance, min_distance_max);

    int max_distance = 1500;
    int max_distance_max = 2500;
    cv::createTrackbar("max. Z Distance (cm)", window_name, &max_distance, max_distance_max);

    int x_width = 760;
    int x_width_max = 2000;
    cv::createTrackbar("X Width", window_name, &x_width, x_width_max);

    int voxel_size = 1;
    int voxel_size_max = 10;
    cv::createTrackbar("Voxel Size (cm)", window_name, &voxel_size, voxel_size_max);

    int ransac_threshold = 1;
    int ransac_threshold_max = 10;
    cv::createTrackbar("RANSAC Threshold (cm)", window_name, &ransac_threshold, ransac_threshold_max);

    int angle_threshold = 30;
    int angle_threshold_max = 180;
    cv::createTrackbar("RANSAC angle Threshold (Deg)", window_name, &angle_threshold, angle_threshold_max);

    int ransac_iterations = 200;
    int ransac_iterations_max = 2000;
    cv::createTrackbar("RANSAC Iterations", window_name, &ransac_iterations, ransac_iterations_max);

    // variables for exponential moving average
    int ma_alpha_int = 10;
    int ma_alpha_max = 100;
    cv::createTrackbar("Alpha (Averaging)", window_name, &ma_alpha_int, ma_alpha_max);
    float groundPlaneDistance = 1.0F;
    Eigen::Vector4f groundPlaneCoefficients = {0.0, 0.0, 0.0, 0.0};

    cv::createButton("Tare", nullptr);

    cv::Mat image_twochannel(cv::Size(rs.color_width, rs.color_height), CV_8UC2);
    cv::Mat image_threechannel(cv::Size(rs.color_width, rs.color_height), CV_8UC3);
    cv::Mat image_bgr(cv::Size(rs.color_width, rs.color_height), CV_8UC3);

    // initialize PCL Filters
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> passz;
    pcl::PassThrough<pcl::PointXYZ> passy;
    pcl::PassThrough<pcl::PointXYZ> passx;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> downsample;

    while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        // wait for and get frames from Camera
        auto pcl_points = rs.get_pcl_point_cloud();
        auto rs2_color_frame = rs.get_color_frame();

        // convert rs2 frame to OpenCV Mat
        if ( rs2_color_frame->get_profile().format() == RS2_FORMAT_YUYV )
        {
            image_twochannel.data = (uchar *)rs2_color_frame->get_data();
            cv::cvtColor(image_twochannel, image_bgr, cv::COLOR_YUV2BGR_YUYV);
        }
        else if ( rs2_color_frame->get_profile().format() == RS2_FORMAT_BGR8 )
        {
            image_bgr.data = (uchar *)rs2_color_frame->get_data();
        }
        else if ( rs2_color_frame->get_profile().format() == RS2_FORMAT_RGB8 )
        {
            image_threechannel.data = (uchar *)rs2_color_frame->get_data();
            cv::cvtColor(image_threechannel, image_bgr, cv::COLOR_RGB2BGR);
        }
        else
        {
            throw std::invalid_argument("Color format not supported");
        }

        // filter point cloud by z distance
        passz.setInputCloud(pcl_points);
        passz.setFilterFieldName("z");
        passz.setFilterLimits((float)min_distance / 100.0F, (float)max_distance / 100.0F);
        passz.filter(*cloud_filtered);

        // filter point cloud by y distance
        passy.setInputCloud(cloud_filtered);
        passy.setFilterFieldName("y");
        passy.setFilterLimits(0.5, 4.0);
        passy.filter(*cloud_filtered);

        // filter point cloud by x distance
        passx.setInputCloud(cloud_filtered);
        passx.setFilterFieldName("x");
        passx.setFilterLimits(-(float)x_width / 200.0F, (float)x_width / 200.0F);
        passx.filter(*cloud_filtered);

        // downsample point cloud
        downsample.setInputCloud(cloud_filtered);
        float voxel_size_f = (float)voxel_size / 100.0F;
        downsample.setLeafSize(voxel_size_f, voxel_size_f, voxel_size_f);
        downsample.filter(*cloud_filtered);

        // detect ground plane using ransac and perpendicular plane model
        pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZ>::Ptr groundPlaneModel(new pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZ>(cloud_filtered));
        pcl::RandomSampleConsensus<pcl::PointXYZ> groundPlaneRansac(groundPlaneModel);
        pcl::PointCloud<pcl::PointXYZ>::Ptr groundPlaneCloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> groundPlaneInliers;
        groundPlaneModel->setAxis(Eigen::Vector3f(0.0, -1.0, 0.0));
        groundPlaneModel->setEpsAngle((float)angle_threshold * M_PI / 180.0);
        groundPlaneRansac.setDistanceThreshold(float(ransac_threshold) / 100.0);
        groundPlaneRansac.setMaxIterations(ransac_iterations);
        groundPlaneRansac.setNumberOfThreads(0);
        bool success = groundPlaneRansac.computeModel();
        if (success)
        {
            Eigen::VectorXf groundPlaneCoeffsRaw;
            groundPlaneRansac.getModelCoefficients(groundPlaneCoeffsRaw);
            if (groundPlaneCoeffsRaw.w() < 0)
            {
                groundPlaneCoeffsRaw = -groundPlaneCoeffsRaw;
            }

            // moving average of plane equation
            float ma_alpha = (float)ma_alpha_int / 100.0F;
            Eigen::Vector4f s1 = groundPlaneCoeffsRaw.head<4>() * ma_alpha;
            Eigen::Vector4f s2 = groundPlaneCoefficients * (1.0F - ma_alpha);
            groundPlaneCoefficients = s1 + s2;
            groundPlaneDistance = groundPlaneCoefficients.w();

            groundPlaneRansac.getInliers(groundPlaneInliers);
            pcl::copyPointCloud(*cloud_filtered, groundPlaneInliers, *groundPlaneCloud);

            // print Plane Distance to image
            std::stringstream distanceBuffer;
            distanceBuffer << std::fixed << std::setprecision(2) << groundPlaneDistance;
            cv::putText(image_bgr, distanceBuffer.str(), cv::Point(10, 50), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);

            // turn ground plane pcl cloud into opencv points
            std::vector<cv::Point3f> points_cv;
            for (auto point : groundPlaneCloud->points)
            {
                points_cv.push_back(cv::Point3d(point.x, point.y, point.z));
            }

            // project point cloud to camera plane
            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(points_cv, rs.rvec, rs.tvec, intrinsicMat, rs.distortion, projectedPoints);

            // color points in image
#pragma omp parallel for default(none) shared(image_bgr, projectedPoints)
            for (auto point : projectedPoints)
            {
                auto ix((unsigned int)std::round(point.x));
                auto iy((unsigned int)std::round(point.y));
                image_bgr.at<cv::Vec3b>(iy, ix) = cv::Vec3b (0, 255, 0);
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

        cv::putText(image_bgr, std::to_string(ms_int.count()), cv::Point(10, 100), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);

        //  Update the window with new data
        imshow(window_name, image_bgr);
    }

    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
catch (const rs2::error &e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}