#include <iostream>
#include <thread>
#include <chrono>

#include <librealsense2/rs.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/opencv.hpp>

pcl::PointCloud<pcl::PointXYZ>::Ptr rs2_points_to_pcl(const rs2::points &points)
{
    // Function to turn uncoloured rs2 point cloud to pcl point cloud
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

int main(int argc, char *argv[])
try
{
    // configure realsense camera
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_YUYV, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

    // create rs2 point cloud
    rs2::pointcloud pc;
    rs2::points points;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with high accuracy configuration
    rs2::pipeline_profile profile = pipe.start(cfg);
    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);

    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = depth_stream.get_intrinsics();

    // camera data for projectPoints
    std::vector<float> rvec = {0.0, 0.0, 0.0};             // Rotation Vector
    std::vector<float> tvec = {0.0, 0.0, 0.0};             // Translation Vector
    std::vector<float> distortion;                         // no distortion
    cv::Mat intrisicMat(3, 3, cv::DataType<double>::type); // Intrisic matrix
    intrisicMat.at<double>(0, 0) = intrinsics.fx;
    intrisicMat.at<double>(1, 0) = 0;
    intrisicMat.at<double>(2, 0) = 0;
    intrisicMat.at<double>(0, 1) = 0;
    intrisicMat.at<double>(1, 1) = intrinsics.fy;
    intrisicMat.at<double>(2, 1) = 0;
    intrisicMat.at<double>(0, 2) = intrinsics.ppx;
    intrisicMat.at<double>(1, 2) = intrinsics.ppy;
    intrisicMat.at<double>(2, 2) = 1;

    // create OpenCV window
    const auto window_name = "Display Image";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    const auto trackbar_window_name = "Settings";
    cv::namedWindow(trackbar_window_name, cv::WINDOW_AUTOSIZE);

    int min_distance = 20;
    int min_distance_max = 1500;
    cv::createTrackbar("min. Distance (cm)", trackbar_window_name, &min_distance, min_distance_max);

    int max_distance = 1500;
    int max_distance_max = 2500;
    cv::createTrackbar("max. Distance (cm)", trackbar_window_name, &max_distance, max_distance_max);

    int voxel_size = 1;
    int voxel_size_max = 10;
    cv::createTrackbar("Voxel Size (cm)", trackbar_window_name, &voxel_size, voxel_size_max);

    int ransac_threshold = 1;
    int ransac_threshold_max = 10;
    cv::createTrackbar("RANSAC Threshold (cm)", trackbar_window_name, &ransac_threshold, ransac_threshold_max);

    int ransac_iterations = 200;
    int ransac_iterations_max = 2000;
    cv::createTrackbar("RANSAC Iterations", trackbar_window_name, &ransac_iterations, ransac_iterations_max);

    // variables for exponential moving average
    int ma_alpha_int = 10;
    int ma_alpha_max = 100;
    cv::createTrackbar("Alpha (Averaging)", trackbar_window_name, &ma_alpha_int, ma_alpha_max);
    float ma_alpha = (float)ma_alpha_int / 100.0;
    float groundPlaneDistance = 1.0;

    while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        // wait for and get frames from Camera
        rs2::frameset data = pipe.wait_for_frames();
        rs2::frame depth = data.get_depth_frame();
        rs2::frame yuyv = data.get_color_frame();

        // Query frame size (width and height)
        const int w = yuyv.as<rs2::video_frame>().get_width();
        const int h = yuyv.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the color image
        cv::Mat image_yuyv(cv::Size(w, h), CV_8UC2, (void *)yuyv.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat image_bgr(cv::Size(w, h), CV_8UC3);
        cv::cvtColor(image_yuyv, image_bgr, cv::COLOR_YUV2BGR_YUYV);

        // turn depth image to rs2 and pcl point cloud
        points = pc.calculate(depth);
        auto pcl_points = rs2_points_to_pcl(points);

        // filter point cloud by z distance
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(pcl_points);
        pass.setFilterFieldName("z");
        pass.setFilterLimits((float)min_distance / 100.0, (float)max_distance / 100.0);
        pass.filter(*cloud_filtered);

        // downsample pointc cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> downsample;
        downsample.setInputCloud(cloud_filtered);
        float voxel_size_f = (float)voxel_size / 100.0;
        downsample.setLeafSize(voxel_size_f, voxel_size_f, voxel_size_f);
        downsample.filter(*cloud_filtered);

        // detect ground plane using ransac and perpendicular plane model
        pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZ>::Ptr groundPlaneModel(new pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZ>(cloud_filtered));
        pcl::RandomSampleConsensus<pcl::PointXYZ> groundPlaneRansac(groundPlaneModel);
        pcl::PointCloud<pcl::PointXYZ>::Ptr groundPlaneCloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> groundPlaneInliers;
        groundPlaneModel->setAxis(Eigen::Vector3f(0.0, -1.0, 0.0));
        groundPlaneModel->setEpsAngle(30.0 * M_PI / 180.0);
        groundPlaneRansac.setDistanceThreshold(float(ransac_threshold) / 100.0);
        groundPlaneRansac.setMaxIterations(ransac_iterations);
        bool success = groundPlaneRansac.computeModel();
        if (success)
        {
            Eigen::VectorXf groundPlaneCoeffs;
            groundPlaneRansac.getModelCoefficients(groundPlaneCoeffs);
            float groundPlaneDistanceRaw = std::abs(groundPlaneCoeffs(3));
            groundPlaneDistance = (ma_alpha * groundPlaneDistanceRaw) + (1.0 - ma_alpha) * groundPlaneDistance;
            groundPlaneRansac.getInliers(groundPlaneInliers);
            pcl::copyPointCloud(*cloud_filtered, groundPlaneInliers, *groundPlaneCloud);

            // print Plane Distance to image
            char distanceBuffer[10];
            sprintf(distanceBuffer, "%.2f m ", groundPlaneDistance);
            cv::putText(image_bgr, distanceBuffer, cv::Point(10, 50), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);

            // turn ground plane pcl cloud into opencv points
            std::vector<cv::Point3f> points_cv;
#pragma omp parallel for
            for (int i = 0; i < groundPlaneCloud->points.size(); i++)
            {
                points_cv.push_back(cv::Point3d(groundPlaneCloud->points[i].x, groundPlaneCloud->points[i].y, groundPlaneCloud->points[i].z));
            }

            // project point cloud to camera plane
            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(points_cv, rvec, tvec, intrisicMat, distortion, projectedPoints);

            // color points in image
            cv::Vec3b color(0, 255, 0);
#pragma omp parallel for
            for (unsigned int i = 0; i < projectedPoints.size(); i++)
            {
                auto pt = projectedPoints[i];
                unsigned int ix(std::round(pt.x)), iy(std::round(pt.y));
                image_bgr.at<cv::Vec3b>(iy, ix) = color;
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

        cv::putText(image_bgr, std::to_string(ms_int.count()), cv::Point(10, 100), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);
        std::cout << ms_int.count() << std::endl;
        // Update the window with new data
        imshow(window_name, image_bgr);
    }

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