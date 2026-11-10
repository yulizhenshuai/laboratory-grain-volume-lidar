#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/search/kdtree.h>

using PointT = pcl::PointXYZ;
using PointNT = pcl::PointNormal;
using CloudT = pcl::PointCloud<PointT>;
using CloudNT = pcl::PointCloud<PointNT>;

namespace
{
    constexpr double kPi = 3.14159265358979323846;
    constexpr std::size_t kLevelCount = 3;

    struct Config
    {

        int frames = 8;


        std::string input_prefix = "scan45_";
        std::string input_suffix = ".pcd";
        int input_index_width = 2;

        std::string output_file = "";
        std::string coarse_output_file = "scan45_scheme3_accumulated_coarse.pcd";
        std::string csv_file = "scan45_scheme3_accumulated_transforms.csv";


        float pitch_axis_deg = 45.0f;


        bool use_direct_pitch_y = false;
        float direct_pitch_y_deg = 45.0f;

        bool auto_yaw_step = true;
        float yaw_step_deg = 45.0f;
        bool yaw_ccw = false;


        float offset_x_m = 0.070f;
        float offset_y_m = 0.1600f;
        float offset_z_m = 0.0f;


        int target_window = 2;
        int edge_target_window = 1;
        int edge_frame_count = 1;

        int anchor_frame = 1;

        bool check_loop_closure = true;


        float hypothesis_angle_deg = 6.0f;
        float edge_hypothesis_angle_deg = 10.0f;

        float max_icp_correction_translation_m = 0.80f;
        float max_icp_correction_rotation_deg = 25.0f;

        float edge_max_icp_correction_translation_m = 1.00f;
        float edge_max_icp_correction_rotation_deg = 35.0f;


        float max_total_from_direct_translation_m = -1.0f;
        float max_total_from_direct_rotation_deg = -1.0f;

        bool soft_accept_rejected_icp = false;
        float soft_rejected_damping = 1.0f;
        float edge_soft_rejected_damping = 1.0f;
        float max_applied_icp_translation_m = 0.06f;
        float max_applied_icp_rotation_deg = 3.0f;
        float edge_max_applied_icp_translation_m = 0.10f;
        float edge_max_applied_icp_rotation_deg = 6.0f;

        bool enable_quality_gate = true;
        float quality_eval_distance_m = 0.15f;
        float min_overlap_ratio = 0.08f;
        float max_alignment_rmse_m = 0.12f;
        float min_rmse_improvement_ratio = 0.00f;
        float min_overlap_gain = -0.02f;

        int local_refine_iterations = 0;
        int local_refine_target_window = 2;
        bool lock_anchor_during_refine = true;

 
        bool fallback_to_mechanical = true;


        int threads = 0;
        int ransac_iterations = 30;
        bool verbose = false;


        bool enable_icp_filter = true;
        float icp_min_range_m = 0.20f;
        float icp_max_range_m = 5.00f;
        float icp_voxel_leaf_m = 0.025f;
        int icp_sor_mean_k = 40;
        float icp_sor_stddev = 1.5f;
        bool enable_ror = true;
        float icp_ror_radius_m = 0.08f;
        int icp_ror_min_neighbors = 4;

        float x_min = -std::numeric_limits<float>::infinity();
        float x_max = std::numeric_limits<float>::infinity();
        float y_min = -std::numeric_limits<float>::infinity();
        float y_max = std::numeric_limits<float>::infinity();
        float z_min = -10.0f;
        float z_max = 10.0f;

        bool save_aligned_frames = true;
        bool save_icp_debug_clouds = false;
    };

    struct ICPLevel
    {
        float max_correspondence_m;
        int max_iterations;
        int normal_k;
        float ransac_threshold_m;
    };


    const std::array<ICPLevel, kLevelCount> kLevels = { {
        {0.45f, 35, 25, 0.18f},
        {0.22f, 45, 30, 0.09f},
        {0.08f, 70, 35, 0.04f},
    } };

    struct ICPResult
    {
        bool converged = false;
        bool accepted = false;
        Eigen::Matrix4f correction = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f initial_hypothesis = Eigen::Matrix4f::Identity();
        double fitness = std::numeric_limits<double>::infinity();
        int successful_levels = 0;
        std::string status = "not_run";
        std::size_t source_points = 0;
        std::size_t target_points = 0;
        float translation_m = 0.0f;
        float rotation_deg = 0.0f;
        double selection_score = std::numeric_limits<double>::infinity();
        double overlap_ratio = 0.0;
        double rmse_m = std::numeric_limits<double>::infinity();
        double base_overlap_ratio = 0.0;
        double base_rmse_m = std::numeric_limits<double>::infinity();
    };

    struct QualityMetrics
    {
        double overlap_ratio = 0.0;
        double rmse_m = std::numeric_limits<double>::infinity();
        int inlier_count = 0;
    };

    bool ensureDirectory(const std::string& path)
    {
#ifdef _WIN32
        const int rc = _mkdir(path.c_str());
#else
        const int rc = mkdir(path.c_str(), 0755);
#endif
        return rc == 0 || errno == EEXIST;
    }

    float deg2rad(float deg)
    {
        return deg * static_cast<float>(kPi / 180.0);
    }

    float rad2deg(float rad)
    {
        return rad * static_cast<float>(180.0 / kPi);
    }

    float resolvedPitchYDeg(const Config& cfg)
    {
        return cfg.use_direct_pitch_y
            ? cfg.direct_pitch_y_deg
            : 90.0f - cfg.pitch_axis_deg;
    }

    float resolvedYawStepDeg(const Config& cfg)
    {
        if (cfg.auto_yaw_step)
            return 360.0f / static_cast<float>(cfg.frames);
        return cfg.yaw_step_deg;
    }

    int autoMiddleAnchorFrame(int frames)
    {

        return std::max(1, (frames + 1) / 2);
    }

    Eigen::Matrix4f rotationX(float deg)
    {
        Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
        t.block<3, 3>(0, 0) = Eigen::AngleAxisf(deg2rad(deg), Eigen::Vector3f::UnitX()).toRotationMatrix();
        return t;
    }

    Eigen::Matrix4f rotationY(float deg)
    {
        Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
        t.block<3, 3>(0, 0) = Eigen::AngleAxisf(deg2rad(deg), Eigen::Vector3f::UnitY()).toRotationMatrix();
        return t;
    }

    Eigen::Matrix4f rotationZ(float deg)
    {
        Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
        t.block<3, 3>(0, 0) = Eigen::AngleAxisf(deg2rad(deg), Eigen::Vector3f::UnitZ()).toRotationMatrix();
        return t;
    }

    bool finiteMatrix(const Eigen::Matrix4f& transform)
    {
        return transform.array().isFinite().all();
    }

    float translationMagnitude(const Eigen::Matrix4f& transform)
    {
        return transform.block<3, 1>(0, 3).norm();
    }

    float rotationMagnitudeDeg(const Eigen::Matrix4f& transform)
    {
        Eigen::Matrix3f r = transform.block<3, 3>(0, 0);
        float c = (r.trace() - 1.0f) * 0.5f;
        c = std::max(-1.0f, std::min(1.0f, c));
        return rad2deg(std::acos(c));
    }

    Eigen::Matrix4f interpolateCorrection(const Eigen::Matrix4f& correction, float fraction)
    {
        fraction = std::max(0.0f, std::min(1.0f, fraction));
        Eigen::Quaternionf q(correction.block<3, 3>(0, 0));
        q.normalize();
        Eigen::Quaternionf qi = Eigen::Quaternionf::Identity().slerp(fraction, q).normalized();

        Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
        out.block<3, 3>(0, 0) = qi.toRotationMatrix();
        out.block<3, 1>(0, 3) = fraction * correction.block<3, 1>(0, 3);
        return out;
    }

    Eigen::Matrix4f limitCorrection(const Eigen::Matrix4f& correction, float max_translation_m, float max_rotation_deg)
    {
        Eigen::Matrix4f out = correction;

        const Eigen::Vector3f t = correction.block<3, 1>(0, 3);
        const float tn = t.norm();
        if (tn > max_translation_m && tn > 1e-9f)
            out.block<3, 1>(0, 3) = t * (max_translation_m / tn);

        const float rd = rotationMagnitudeDeg(correction);
        if (rd > max_rotation_deg && rd > 1e-6f)
        {
            Eigen::Quaternionf q(correction.block<3, 3>(0, 0));
            q.normalize();
            const float f = max_rotation_deg / rd;
            Eigen::Quaternionf ql = Eigen::Quaternionf::Identity().slerp(f, q).normalized();
            out.block<3, 3>(0, 0) = ql.toRotationMatrix();
        }
        return out;
    }

    std::string buildFilename(int frame, const Config& cfg)
    {
        std::ostringstream oss;
        oss << cfg.input_prefix;
        if (cfg.input_index_width > 0)
            oss << std::setw(cfg.input_index_width) << std::setfill('0') << frame;
        else
            oss << frame;
        oss << cfg.input_suffix;
        return oss.str();
    }

    Eigen::Matrix4f makeDirectPoseToFrame1(int frame_index, const Config& cfg)
    {

        const float yaw_sign = cfg.yaw_ccw ? 1.0f : -1.0f;
        const float yaw_step_deg = resolvedYawStepDeg(cfg);
        const float theta_deg =
            yaw_sign * yaw_step_deg * static_cast<float>(frame_index);
        const float theta_rad = deg2rad(theta_deg);


        Eigen::Matrix4f yaw_motion = rotationZ(theta_deg);
        const Eigen::Vector3f offset(
            cfg.offset_x_m,
            cfg.offset_y_m,
            cfg.offset_z_m);

        const Eigen::Matrix3f rz =
            Eigen::AngleAxisf(
                theta_rad,
                Eigen::Vector3f::UnitZ()).toRotationMatrix();

        yaw_motion.block<3, 1>(0, 3) =
            rz * offset - offset;

        return yaw_motion * rotationY(resolvedPitchYDeg(cfg));
    }

    CloudT::Ptr transformedCloud(const CloudT::Ptr& input, const Eigen::Matrix4f& transform)
    {
        CloudT::Ptr output(new CloudT);
        pcl::transformPointCloud(*input, *output, transform);
        return output;
    }

    CloudT::Ptr finiteCloud(const CloudT::Ptr& input)
    {
        CloudT::Ptr output(new CloudT);
#ifdef _WIN32
        if (!input)
            return output;

        output->reserve(input->size());
        for (const auto& p : input->points)
        {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
                output->push_back(p);
        }
#else
        std::vector<int> mapping;
        pcl::removeNaNFromPointCloud(*input, *output, mapping);
#endif
        output->width = static_cast<std::uint32_t>(output->size());
        output->height = 1;
        output->is_dense = true;
        return output;
    }

    CloudT::Ptr filterAxis(const CloudT::Ptr& input, const std::string& axis, float min_v, float max_v)
    {
        if (!std::isfinite(min_v) && !std::isfinite(max_v))
            return input;
#ifdef _WIN32
        CloudT::Ptr output(new CloudT);
        output->reserve(input->size());
        for (const auto& p : input->points)
        {
            const float value = axis == "x" ? p.x : (axis == "y" ? p.y : p.z);
            if (value >= min_v && value <= max_v)
                output->push_back(p);
        }
#else
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(input);
        pass.setFilterFieldName(axis);
        pass.setFilterLimits(min_v, max_v);
        CloudT::Ptr output(new CloudT);
        pass.filter(*output);
#endif
        output->width = static_cast<std::uint32_t>(output->size());
        output->height = 1;
        output->is_dense = true;
        return output;
    }

    CloudT::Ptr filterRange(const CloudT::Ptr& input, float min_range, float max_range)
    {
        CloudT::Ptr output(new CloudT);
        output->reserve(input->size());
        const float min2 = min_range * min_range;
        const float max2 = max_range * max_range;
        for (const auto& p : input->points)
        {
            const float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
            if (r2 >= min2 && r2 <= max2)
                output->push_back(p);
        }
        output->width = static_cast<std::uint32_t>(output->size());
        output->height = 1;
        output->is_dense = true;
        return output;
    }

    CloudT::Ptr preprocessForICP(const CloudT::Ptr& raw, const Config& cfg)
    {
        CloudT::Ptr cloud = finiteCloud(raw);
        if (!cfg.enable_icp_filter)
            return cloud;

        cloud = filterRange(cloud, cfg.icp_min_range_m, cfg.icp_max_range_m);
        cloud = filterAxis(cloud, "x", cfg.x_min, cfg.x_max);
        cloud = filterAxis(cloud, "y", cfg.y_min, cfg.y_max);
        cloud = filterAxis(cloud, "z", cfg.z_min, cfg.z_max);

        if (cloud->size() < 80)
            return cloud;

        if (cfg.icp_voxel_leaf_m > 0.0f)
        {
#ifdef _WIN32
            struct VoxelAccumulator
            {
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                std::size_t count = 0;
            };

            const double inverse_leaf = 1.0 / static_cast<double>(cfg.icp_voxel_leaf_m);
            std::map<std::tuple<int, int, int>, VoxelAccumulator> voxels;
            for (const auto& p : cloud->points)
            {
                const auto key = std::make_tuple(
                    static_cast<int>(std::floor(static_cast<double>(p.x) * inverse_leaf)),
                    static_cast<int>(std::floor(static_cast<double>(p.y) * inverse_leaf)),
                    static_cast<int>(std::floor(static_cast<double>(p.z) * inverse_leaf)));
                auto& accumulator = voxels[key];
                accumulator.x += p.x;
                accumulator.y += p.y;
                accumulator.z += p.z;
                ++accumulator.count;
            }

            CloudT::Ptr voxel_cloud(new CloudT);
            voxel_cloud->reserve(voxels.size());
            for (const auto& entry : voxels)
            {
                const auto& accumulator = entry.second;
                PointT centroid;
                centroid.x = static_cast<float>(accumulator.x / accumulator.count);
                centroid.y = static_cast<float>(accumulator.y / accumulator.count);
                centroid.z = static_cast<float>(accumulator.z / accumulator.count);
                voxel_cloud->push_back(centroid);
            }
            cloud = voxel_cloud;
#else
            pcl::VoxelGrid<PointT> voxel;
            voxel.setInputCloud(cloud);
            voxel.setLeafSize(cfg.icp_voxel_leaf_m, cfg.icp_voxel_leaf_m, cfg.icp_voxel_leaf_m);
            CloudT::Ptr voxel_cloud(new CloudT);
            voxel.filter(*voxel_cloud);
            cloud = voxel_cloud;
#endif
        }

        if (cloud->size() >= static_cast<std::size_t>(cfg.icp_sor_mean_k + 5))
        {
#ifdef _WIN32
            pcl::search::KdTree<PointT> tree;
            tree.setInputCloud(cloud);
            std::vector<double> mean_distances(cloud->size(), std::numeric_limits<double>::quiet_NaN());
            std::vector<int> indices;
            std::vector<float> squared_distances;
            indices.reserve(static_cast<std::size_t>(cfg.icp_sor_mean_k + 1));
            squared_distances.reserve(static_cast<std::size_t>(cfg.icp_sor_mean_k + 1));
            double distance_sum = 0.0;
            std::size_t valid_count = 0;

            for (std::size_t i = 0; i < cloud->size(); ++i)
            {
                indices.clear();
                squared_distances.clear();
                const int found = tree.nearestKSearch(
                    cloud->points[i], cfg.icp_sor_mean_k + 1, indices, squared_distances);
                if (found <= 1)
                    continue;

                double local_sum = 0.0;
                int neighbor_count = 0;
                for (int j = 0; j < found && neighbor_count < cfg.icp_sor_mean_k; ++j)
                {
                    if (indices[static_cast<std::size_t>(j)] == static_cast<int>(i))
                        continue;
                    local_sum += std::sqrt(std::max(0.0f, squared_distances[static_cast<std::size_t>(j)]));
                    ++neighbor_count;
                }
                if (neighbor_count == 0)
                    continue;

                const double mean_distance = local_sum / static_cast<double>(neighbor_count);
                mean_distances[i] = mean_distance;
                distance_sum += mean_distance;
                ++valid_count;
            }

            if (valid_count > 1)
            {
                const double global_mean = distance_sum / static_cast<double>(valid_count);
                double variance_sum = 0.0;
                for (const double distance : mean_distances)
                {
                    if (std::isfinite(distance))
                    {
                        const double delta = distance - global_mean;
                        variance_sum += delta * delta;
                    }
                }
                const double standard_deviation = std::sqrt(
                    variance_sum / static_cast<double>(valid_count - 1));
                const double threshold = global_mean +
                    static_cast<double>(cfg.icp_sor_stddev) * standard_deviation;

                CloudT::Ptr sor_cloud(new CloudT);
                sor_cloud->reserve(cloud->size());
                for (std::size_t i = 0; i < cloud->size(); ++i)
                {
                    if (std::isfinite(mean_distances[i]) && mean_distances[i] <= threshold)
                        sor_cloud->push_back(cloud->points[i]);
                }
                cloud = sor_cloud;
            }
#else
            pcl::StatisticalOutlierRemoval<PointT> sor;
            sor.setInputCloud(cloud);
            sor.setMeanK(cfg.icp_sor_mean_k);
            sor.setStddevMulThresh(cfg.icp_sor_stddev);
            CloudT::Ptr sor_cloud(new CloudT);
            sor.filter(*sor_cloud);
            cloud = sor_cloud;
#endif
        }

        if (cfg.enable_ror && cloud->size() > 100)
        {
#ifdef _WIN32
            pcl::search::KdTree<PointT> tree;
            tree.setInputCloud(cloud);
            CloudT::Ptr ror_cloud(new CloudT);
            ror_cloud->reserve(cloud->size());
            std::vector<int> indices;
            std::vector<float> squared_distances;
            for (std::size_t i = 0; i < cloud->size(); ++i)
            {
                indices.clear();
                squared_distances.clear();
                const int found = tree.radiusSearch(
                    cloud->points[i], cfg.icp_ror_radius_m, indices, squared_distances);
                int neighbor_count = found;
                for (const int index : indices)
                {
                    if (index == static_cast<int>(i))
                    {
                        --neighbor_count;
                        break;
                    }
                }
                if (neighbor_count >= cfg.icp_ror_min_neighbors)
                    ror_cloud->push_back(cloud->points[i]);
            }
#else
            pcl::RadiusOutlierRemoval<PointT> ror;
            ror.setInputCloud(cloud);
            ror.setRadiusSearch(cfg.icp_ror_radius_m);
            ror.setMinNeighborsInRadius(cfg.icp_ror_min_neighbors);
            CloudT::Ptr ror_cloud(new CloudT);
            ror.filter(*ror_cloud);
#endif
            if (ror_cloud->size() >= 80)
                cloud = ror_cloud;
        }

        cloud->width = static_cast<std::uint32_t>(cloud->size());
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

    CloudNT::Ptr addNormals(const CloudT::Ptr& cloud, int requested_k)
    {
        CloudNT::Ptr output(new CloudNT);
        if (!cloud || cloud->size() < 10)
            return output;

        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setNumberOfThreads(std::max(1, omp_get_max_threads()));
        ne.setInputCloud(cloud);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        ne.setSearchMethod(tree);
        ne.setKSearch(std::max(3, std::min<int>(requested_k, static_cast<int>(cloud->size()) - 1)));

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.compute(*normals);

        output->reserve(cloud->size());
        const std::size_t n = std::min(cloud->size(), normals->size());
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& p = cloud->points[i];
            const auto& normal = normals->points[i];
            if (!std::isfinite(normal.normal_x) || !std::isfinite(normal.normal_y) || !std::isfinite(normal.normal_z))
                continue;
            PointNT pn;
            pn.x = p.x; pn.y = p.y; pn.z = p.z;
            pn.normal_x = normal.normal_x;
            pn.normal_y = normal.normal_y;
            pn.normal_z = normal.normal_z;
            pn.curvature = normal.curvature;
            pn.data[3] = 1.0f;
            pn.normal[3] = 0.0f;
            output->push_back(pn);
        }
        output->width = static_cast<std::uint32_t>(output->size());
        output->height = 1;
        output->is_dense = false;
        return output;
    }

    CloudT::Ptr mergeClouds(const std::vector<CloudT::Ptr>& clouds)
    {
        CloudT::Ptr merged(new CloudT);
        std::size_t reserve_n = 0;
        for (const auto& c : clouds)
            if (c) reserve_n += c->size();
        merged->reserve(reserve_n);
        for (const auto& c : clouds)
            if (c) *merged += *c;
        merged->width = static_cast<std::uint32_t>(merged->size());
        merged->height = 1;
        merged->is_dense = false;
        return merged;
    }

    std::vector<Eigen::Matrix4f> buildInitialHypotheses(float angle_deg)
    {
        std::vector<Eigen::Matrix4f> hs;
        hs.push_back(Eigen::Matrix4f::Identity());
        if (angle_deg > 0.0f)
        {

            for (const float s : { -1.0f, 1.0f })
            {
                hs.push_back(rotationZ(s * angle_deg));
                hs.push_back(rotationY(s * angle_deg));
                hs.push_back(rotationX(s * angle_deg));
            }
        }
        return hs;
    }

    ICPResult runPointToPlaneICP(
        const CloudT::Ptr& source_global_initial,
        const CloudT::Ptr& target_global,
        const Eigen::Matrix4f& initial_hypothesis,
        const Config& cfg)
    {
        ICPResult result;
        result.initial_hypothesis = initial_hypothesis;

        CloudT::Ptr source_initial = transformedCloud(source_global_initial, initial_hypothesis);
        Eigen::Matrix4f accumulated = Eigen::Matrix4f::Identity();

        for (std::size_t level_index = 0; level_index < kLevelCount; ++level_index)
        {
            const ICPLevel& level = kLevels[level_index];
            CloudT::Ptr source_level = (level_index == 0)
                ? source_initial
                : transformedCloud(source_initial, accumulated);

            result.source_points = source_level->size();
            result.target_points = target_global->size();
            if (source_level->size() < 60 || target_global->size() < 80)
            {
                result.status = "too_few_points_level_" + std::to_string(level_index + 1);
                return result;
            }

            CloudNT::Ptr source_nt = addNormals(source_level, level.normal_k);
            CloudNT::Ptr target_nt = addNormals(target_global, level.normal_k);
            if (source_nt->size() < 60 || target_nt->size() < 80)
            {
                result.status = "too_few_normals_level_" + std::to_string(level_index + 1);
                return result;
            }

            pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
            pcl::registration::TransformationEstimationPointToPlaneLLS<PointNT, PointNT>::Ptr estimator(
                new pcl::registration::TransformationEstimationPointToPlaneLLS<PointNT, PointNT>);
            icp.setTransformationEstimation(estimator);
            icp.setInputSource(source_nt);
            icp.setInputTarget(target_nt);
            icp.setMaximumIterations(level.max_iterations);
            icp.setMaxCorrespondenceDistance(level.max_correspondence_m);
            icp.setTransformationEpsilon(1e-8);
            icp.setEuclideanFitnessEpsilon(1e-7);
            icp.setRANSACIterations(cfg.ransac_iterations);
            icp.setRANSACOutlierRejectionThreshold(level.ransac_threshold_m);
            icp.setUseReciprocalCorrespondences(false);

            CloudNT aligned;
            icp.align(aligned);
            if (!icp.hasConverged())
            {
                result.status = "icp_not_converged_level_" + std::to_string(level_index + 1);
                return result;
            }

            const Eigen::Matrix4f delta = icp.getFinalTransformation();
            if (!finiteMatrix(delta))
            {
                result.status = "invalid_transform_level_" + std::to_string(level_index + 1);
                return result;
            }

            accumulated = delta * accumulated;
            result.successful_levels++;
            if (level_index + 1 == kLevelCount)
                result.fitness = icp.getFitnessScore(level.max_correspondence_m);
        }


        result.correction = accumulated * initial_hypothesis;
        result.translation_m = translationMagnitude(result.correction);
        result.rotation_deg = rotationMagnitudeDeg(result.correction);
        result.converged = true;
        result.status = "converged_all_levels";
        return result;
    }

    QualityMetrics evaluateAlignmentQuality(
        const CloudT::Ptr& source_global_predicted,
        const CloudT::Ptr& target_global,
        const Eigen::Matrix4f& correction,
        float max_distance_m)
    {
        QualityMetrics q;
        if (!source_global_predicted || !target_global ||
            source_global_predicted->empty() || target_global->empty())
            return q;

        CloudT::Ptr source_eval = transformedCloud(source_global_predicted, correction);
        pcl::search::KdTree<PointT> tree;
        tree.setInputCloud(target_global);

        const float max_d2 = max_distance_m * max_distance_m;
        std::vector<int> index(1);
        std::vector<float> dist2(1);
        double sum = 0.0;
        int inliers = 0;

        for (const auto& p : source_eval->points)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            if (tree.nearestKSearch(p, 1, index, dist2) > 0 && dist2[0] <= max_d2)
            {
                sum += static_cast<double>(dist2[0]);
                ++inliers;
            }
        }

        q.inlier_count = inliers;
        q.overlap_ratio = static_cast<double>(inliers) /
            std::max<std::size_t>(1, source_eval->size());
        if (inliers > 0)
            q.rmse_m = std::sqrt(sum / static_cast<double>(inliers));
        return q;
    }

    ICPResult chooseBestICP(
        const CloudT::Ptr& source_global_predicted,
        const CloudT::Ptr& target_global,
        const Config& cfg,
        bool edge_frame)
    {
        ICPResult best;
        ICPResult best_rejected;
        best.source_points = source_global_predicted ? source_global_predicted->size() : 0;
        best.target_points = target_global ? target_global->size() : 0;
        best_rejected.source_points = best.source_points;
        best_rejected.target_points = best.target_points;

        if (!source_global_predicted || !target_global ||
            source_global_predicted->size() < 60 || target_global->size() < 80)
        {
            best.status = "too_few_points_before_icp";
            return best;
        }

        const float trans_guard = edge_frame
            ? cfg.edge_max_icp_correction_translation_m
            : cfg.max_icp_correction_translation_m;
        const float rot_guard = edge_frame
            ? cfg.edge_max_icp_correction_rotation_deg
            : cfg.max_icp_correction_rotation_deg;
        const float hypothesis_angle = edge_frame
            ? cfg.edge_hypothesis_angle_deg
            : cfg.hypothesis_angle_deg;

        const QualityMetrics base_q = evaluateAlignmentQuality(
            source_global_predicted,
            target_global,
            Eigen::Matrix4f::Identity(),
            cfg.quality_eval_distance_m);

        const std::vector<Eigen::Matrix4f> hypotheses = buildInitialHypotheses(hypothesis_angle);
        for (const auto& h : hypotheses)
        {
            ICPResult r = runPointToPlaneICP(source_global_predicted, target_global, h, cfg);
            r.base_overlap_ratio = base_q.overlap_ratio;
            r.base_rmse_m = base_q.rmse_m;

            if (!r.converged)
            {
                if (cfg.verbose)
                    std::cout << "  [ICP trial] failed, status=" << r.status
                    << ", source=" << r.source_points
                    << ", target=" << r.target_points << std::endl;
                if (best_rejected.status == "not_run" || r.successful_levels > best_rejected.successful_levels)
                    best_rejected = r;
                continue;
            }

            const QualityMetrics q = evaluateAlignmentQuality(
                source_global_predicted,
                target_global,
                r.correction,
                cfg.quality_eval_distance_m);
            r.overlap_ratio = q.overlap_ratio;
            r.rmse_m = q.rmse_m;

            const bool correction_ok =
                r.translation_m <= trans_guard &&
                r.rotation_deg <= rot_guard;

            const bool quality_absolute_ok =
                (!cfg.enable_quality_gate) ||
                ((q.overlap_ratio >= cfg.min_overlap_ratio) &&
                    (q.rmse_m <= cfg.max_alignment_rmse_m));

            const bool quality_improved =
                (!cfg.enable_quality_gate) ||
                (!std::isfinite(base_q.rmse_m)) ||
                (q.rmse_m <= base_q.rmse_m * (1.0 - cfg.min_rmse_improvement_ratio)) ||
                (q.overlap_ratio >= base_q.overlap_ratio + cfg.min_overlap_gain);

            const double safe_overlap = std::max(0.03, q.overlap_ratio);
            r.selection_score =
                (std::isfinite(q.rmse_m) ? q.rmse_m : 1e3) / safe_overlap +
                0.20 * r.fitness;

            if (cfg.verbose)
            {
                std::cout << "  [ICP trial] fitness=" << r.fitness
                    << ", score=" << r.selection_score
                    << ", correction=" << r.translation_m << " m / "
                    << r.rotation_deg << " deg"
                    << ", overlap=" << q.overlap_ratio
                    << ", rmse=" << q.rmse_m
                    << ", base_overlap=" << base_q.overlap_ratio
                    << ", base_rmse=" << base_q.rmse_m
                    << ", guard=" << trans_guard << " m / " << rot_guard << " deg"
                    << ((!correction_ok || !quality_absolute_ok || !quality_improved) ? ", rejected" : "")
                    << std::endl;
            }

            if (!correction_ok || !quality_absolute_ok || !quality_improved)
            {
                r.status = !correction_ok ? "rejected_by_large_error_guard" : "rejected_by_quality_gate";
                if (!best_rejected.converged || r.selection_score < best_rejected.selection_score)
                    best_rejected = r;
                continue;
            }

            if (!best.converged || r.selection_score < best.selection_score)
                best = r;
        }

        if (best.converged)
        {
            best.accepted = true;
            best.status = "accepted_quality";
            return best;
        }

        if (best_rejected.status != "not_run")
        {
            best_rejected.accepted = false;
            if (best_rejected.converged)
                best_rejected.status = "no_valid_icp_candidate_best_rejected";
            return best_rejected;
        }

        best.status = "no_valid_icp_candidate";
        return best;
    }

    CloudT::Ptr buildTargetMap(
        const std::vector<CloudT::Ptr>& icp_clouds,
        const std::vector<Eigen::Matrix4f>& refined_poses,
        int current_frame,
        int target_window)
    {
        std::vector<CloudT::Ptr> parts;
        const int begin = std::max(0, current_frame - std::max(1, target_window));
        for (int j = begin; j < current_frame; ++j)
            parts.push_back(transformedCloud(icp_clouds[j], refined_poses[j]));
        return mergeClouds(parts);
    }

    std::vector<int> collectTargetIndices(
        const std::vector<std::uint8_t>& has_refined_pose,
        int current_frame,
        int target_window,
        bool forward_direction)
    {
        std::vector<int> indices;
        if (forward_direction)
        {

            for (int j = current_frame - 1; j >= 0 && static_cast<int>(indices.size()) < target_window; --j)
                if (has_refined_pose[j])
                    indices.push_back(j);
        }
        else
        {

            for (int j = current_frame + 1; j < static_cast<int>(has_refined_pose.size()) &&
                static_cast<int>(indices.size()) < target_window; ++j)
                if (has_refined_pose[j])
                    indices.push_back(j);
        }
        return indices;
    }

    std::vector<int> collectLocalRefineTargetIndices(
        const std::vector<std::uint8_t>& has_refined_pose,
        int current_frame,
        int target_window)
    {
        std::vector<int> indices;
        const int n = static_cast<int>(has_refined_pose.size());
        for (int d = 1; d < n && static_cast<int>(indices.size()) < target_window; ++d)
        {
            const int left = current_frame - d;
            const int right = current_frame + d;
            if (left >= 0 && has_refined_pose[left] && static_cast<int>(indices.size()) < target_window)
                indices.push_back(left);
            if (right < n && has_refined_pose[right] && static_cast<int>(indices.size()) < target_window)
                indices.push_back(right);
        }
        return indices;
    }

    CloudT::Ptr buildTargetMapFromIndices(
        const std::vector<CloudT::Ptr>& icp_clouds,
        const std::vector<Eigen::Matrix4f>& refined_poses,
        const std::vector<int>& target_indices)
    {
        std::vector<CloudT::Ptr> parts;
        parts.reserve(target_indices.size());
        for (const int idx : target_indices)
            parts.push_back(transformedCloud(icp_clouds[idx], refined_poses[idx]));
        return mergeClouds(parts);
    }

    void writeMatrixCSV(std::ofstream& output, const Eigen::Matrix4f& transform)
    {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                output << ',' << std::setprecision(9) << transform(r, c);
    }

    bool parseBoolDirection(const std::string& text, bool& ccw)
    {
        if (text == "ccw" || text == "left" || text == "+" || text == "positive")
        {
            ccw = true;
            return true;
        }
        if (text == "cw" || text == "right" || text == "-" || text == "negative")
        {
            ccw = false;
            return true;
        }
        return false;
    }

    void printUsage(const char* program)
    {
        std::cout
            << "Usage: " << program << " [options]\n\n"
            << "  --frames N                         default 8\n"
            << "  --input-prefix TEXT                default scan45_\n"
            << "  --input-index-width N              default 2, set 0 for no zero padding\n"
            << "  --pitch-axis-deg DEG               actual pitch-axis angle, default 50\n"
            << "  --pitch-y-deg DEG                  directly override effective Ry\n"
            << "  --yaw-step-deg DEG                 manual step; passing it disables auto step\n"
            << "  --auto-yaw-step                    use 360/frames, default enabled\n"
            << "  --yaw-direction ccw|cw             default cw (negative mathematical yaw)\n"
            << "  --radius-m M                       set offset=(0,+M,0)\n"
            << "  --offset-x-m M / --offset-y-m M / --offset-z-m M\n"
            << "  --target-window N                  nearby refined frames as target map, default 2\n"
            << "  --edge-target-window N             boundary frame target window, default 1\n"
            << "  --edge-frame-count N               boundary frame count per side, default 1\n"
            << "  --anchor-frame N                   1-based anchor; default auto-middle\n"
            << "  --hypothesis-angle-deg DEG         default 6, set 0 to disable multi-start\n"
            << "  --edge-hypothesis-angle-deg DEG    default 10\n"
            << "  --max-icp-correction-translation-m M  default 0.80\n"
            << "  --max-icp-correction-rotation-deg DEG default 25\n"
            << "  --edge-max-icp-correction-translation-m M  default 1.00\n"
            << "  --edge-max-icp-correction-rotation-deg DEG default 35\n"
            << "  --max-total-from-direct-translation-m M default disabled, <=0 disables\n"
            << "  --max-total-from-direct-rotation-deg DEG default disabled, <=0 disables\n"
            << "  --soft-accept-rejected-icp         enable old damping fallback for rejected candidates\n"
            << "  --soft-rejected-damping A          default 1.0\n"
            << "  --edge-soft-rejected-damping A     default 1.0\n"
            << "  --max-applied-icp-translation-m M  soft correction cap, default 0.06\n"
            << "  --max-applied-icp-rotation-deg DEG soft correction cap, default 3\n"
            << "  --edge-max-applied-icp-translation-m M default 0.10\n"
            << "  --edge-max-applied-icp-rotation-deg DEG default 6\n"
            << "  --threads N                        OpenMP threads, 0=maximum\n"
            << "  --ransac-iterations N              default 30\n"
            << "  --verbose\n"
            << "  --no-icp-filter                    disable ICP-only preprocessing\n"
            << "  --icp-min-range-m M                default 0.20\n"
            << "  --icp-max-range-m M                default 5.00\n"
            << "  --icp-voxel-leaf-m M               default 0.025\n"
            << "  --icp-sor-mean-k N                 default 40\n"
            << "  --icp-sor-stddev X                 default 1.5\n"
            << "  --no-ror                           disable RadiusOutlierRemoval\n"
            << "  --icp-ror-radius-m M               default 0.08\n"
            << "  --icp-ror-min-neighbors N          default 4\n"
            << "  --x-min/--x-max/--y-min/--y-max/--z-min/--z-max M\n"
            << "  --no-quality-gate                  accept by fitness/geometry score without hard quality gate\n"
            << "  --quality-eval-distance-m M        default 0.15\n"
            << "  --min-overlap-ratio R              default 0.08\n"
            << "  --max-alignment-rmse-m M           default 0.12\n"
            << "  --local-refine-iterations N        default 1\n"
            << "  --local-refine-target-window N     default 2\n"
            << "  --output FILE\n"
            << "  --coarse-output FILE\n"
            << "  --csv FILE\n"
            << "  --no-aligned-frames\n"
            << "  --save-icp-debug-clouds\n"
            << "  --help\n";
    }

    bool parseArguments(int argc, char** argv, Config& cfg)
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            auto needValue = [&](const std::string& name) -> std::string
                {
                    if (i + 1 >= argc)
                        throw std::runtime_error("Missing value for " + name);
                    return argv[++i];
                };

            if (arg == "--help" || arg == "-h")
            {
                printUsage(argv[0]);
                return false;
            }
            else if (arg == "--frames") cfg.frames = std::stoi(needValue(arg));
            else if (arg == "--input-prefix") cfg.input_prefix = needValue(arg);
            else if (arg == "--input-index-width") cfg.input_index_width = std::stoi(needValue(arg));
            else if (arg == "--pitch-axis-deg" || arg == "--actual-pitch-deg")
            {
                cfg.pitch_axis_deg = std::stof(needValue(arg));
                cfg.use_direct_pitch_y = false;
            }
            else if (arg == "--pitch-y-deg" || arg == "--pitch-deg")
            {
                cfg.direct_pitch_y_deg = std::stof(needValue(arg));
                cfg.use_direct_pitch_y = true;
            }
            else if (arg == "--yaw-step-deg")
            {
                cfg.yaw_step_deg = std::stof(needValue(arg));
                cfg.auto_yaw_step = false;
            }
            else if (arg == "--auto-yaw-step") cfg.auto_yaw_step = true;
            else if (arg == "--yaw-direction")
            {
                const std::string v = needValue(arg);
                if (!parseBoolDirection(v, cfg.yaw_ccw))
                    throw std::runtime_error("Invalid yaw direction: " + v);
            }
            else if (arg == "--radius-m")
            {
                const float r = std::stof(needValue(arg));
                cfg.offset_x_m = 0.0f; cfg.offset_y_m = r; cfg.offset_z_m = 0.0f;
            }
            else if (arg == "--offset-x-m" || arg == "--offset-x") cfg.offset_x_m = std::stof(needValue(arg));
            else if (arg == "--offset-y-m" || arg == "--offset-y") cfg.offset_y_m = std::stof(needValue(arg));
            else if (arg == "--offset-z-m" || arg == "--offset-z") cfg.offset_z_m = std::stof(needValue(arg));
            else if (arg == "--target-window") cfg.target_window = std::stoi(needValue(arg));
            else if (arg == "--edge-target-window") cfg.edge_target_window = std::stoi(needValue(arg));
            else if (arg == "--edge-frame-count") cfg.edge_frame_count = std::stoi(needValue(arg));
            else if (arg == "--anchor-frame") cfg.anchor_frame = std::stoi(needValue(arg));
            else if (arg == "--hypothesis-angle-deg") cfg.hypothesis_angle_deg = std::stof(needValue(arg));
            else if (arg == "--edge-hypothesis-angle-deg") cfg.edge_hypothesis_angle_deg = std::stof(needValue(arg));
            else if (arg == "--max-icp-correction-translation-m") cfg.max_icp_correction_translation_m = std::stof(needValue(arg));
            else if (arg == "--max-icp-correction-rotation-deg") cfg.max_icp_correction_rotation_deg = std::stof(needValue(arg));
            else if (arg == "--edge-max-icp-correction-translation-m") cfg.edge_max_icp_correction_translation_m = std::stof(needValue(arg));
            else if (arg == "--edge-max-icp-correction-rotation-deg") cfg.edge_max_icp_correction_rotation_deg = std::stof(needValue(arg));
            else if (arg == "--max-total-from-direct-translation-m") cfg.max_total_from_direct_translation_m = std::stof(needValue(arg));
            else if (arg == "--max-total-from-direct-rotation-deg") cfg.max_total_from_direct_rotation_deg = std::stof(needValue(arg));
            else if (arg == "--soft-accept-rejected-icp") cfg.soft_accept_rejected_icp = true;
            else if (arg == "--soft-rejected-damping") cfg.soft_rejected_damping = std::stof(needValue(arg));
            else if (arg == "--edge-soft-rejected-damping") cfg.edge_soft_rejected_damping = std::stof(needValue(arg));
            else if (arg == "--max-applied-icp-translation-m") cfg.max_applied_icp_translation_m = std::stof(needValue(arg));
            else if (arg == "--max-applied-icp-rotation-deg") cfg.max_applied_icp_rotation_deg = std::stof(needValue(arg));
            else if (arg == "--edge-max-applied-icp-translation-m") cfg.edge_max_applied_icp_translation_m = std::stof(needValue(arg));
            else if (arg == "--edge-max-applied-icp-rotation-deg") cfg.edge_max_applied_icp_rotation_deg = std::stof(needValue(arg));
            else if (arg == "--threads") cfg.threads = std::stoi(needValue(arg));
            else if (arg == "--ransac-iterations") cfg.ransac_iterations = std::stoi(needValue(arg));
            else if (arg == "--verbose") cfg.verbose = true;
            else if (arg == "--no-icp-filter") cfg.enable_icp_filter = false;
            else if (arg == "--icp-min-range-m") cfg.icp_min_range_m = std::stof(needValue(arg));
            else if (arg == "--icp-max-range-m") cfg.icp_max_range_m = std::stof(needValue(arg));
            else if (arg == "--icp-voxel-leaf-m") cfg.icp_voxel_leaf_m = std::stof(needValue(arg));
            else if (arg == "--icp-sor-mean-k") cfg.icp_sor_mean_k = std::stoi(needValue(arg));
            else if (arg == "--icp-sor-stddev") cfg.icp_sor_stddev = std::stof(needValue(arg));
            else if (arg == "--no-ror") cfg.enable_ror = false;
            else if (arg == "--icp-ror-radius-m") cfg.icp_ror_radius_m = std::stof(needValue(arg));
            else if (arg == "--icp-ror-min-neighbors") cfg.icp_ror_min_neighbors = std::stoi(needValue(arg));
            else if (arg == "--x-min") cfg.x_min = std::stof(needValue(arg));
            else if (arg == "--x-max") cfg.x_max = std::stof(needValue(arg));
            else if (arg == "--y-min") cfg.y_min = std::stof(needValue(arg));
            else if (arg == "--y-max") cfg.y_max = std::stof(needValue(arg));
            else if (arg == "--z-min") cfg.z_min = std::stof(needValue(arg));
            else if (arg == "--z-max") cfg.z_max = std::stof(needValue(arg));
            else if (arg == "--output") cfg.output_file = needValue(arg);
            else if (arg == "--coarse-output") cfg.coarse_output_file = needValue(arg);
            else if (arg == "--csv") cfg.csv_file = needValue(arg);
            else if (arg == "--no-aligned-frames") cfg.save_aligned_frames = false;
            else if (arg == "--save-icp-debug-clouds") cfg.save_icp_debug_clouds = true;
            else if (arg == "--no-quality-gate") cfg.enable_quality_gate = false;
            else if (arg == "--quality-eval-distance-m") cfg.quality_eval_distance_m = std::stof(needValue(arg));
            else if (arg == "--min-overlap-ratio") cfg.min_overlap_ratio = std::stof(needValue(arg));
            else if (arg == "--max-alignment-rmse-m") cfg.max_alignment_rmse_m = std::stof(needValue(arg));
            else if (arg == "--min-rmse-improvement-ratio") cfg.min_rmse_improvement_ratio = std::stof(needValue(arg));
            else if (arg == "--min-overlap-gain") cfg.min_overlap_gain = std::stof(needValue(arg));
            else if (arg == "--local-refine-iterations") cfg.local_refine_iterations = std::stoi(needValue(arg));
            else if (arg == "--local-refine-target-window") cfg.local_refine_target_window = std::stoi(needValue(arg));
            else if (arg == "--unlock-anchor-during-refine") cfg.lock_anchor_during_refine = false;
            else throw std::runtime_error("Unknown argument: " + arg);
        }

        if (cfg.frames < 2) throw std::runtime_error("frames must be >= 2");

        if (cfg.anchor_frame == 0)
            cfg.anchor_frame = autoMiddleAnchorFrame(cfg.frames);

        if (cfg.input_index_width < 0) throw std::runtime_error("input index width must be >= 0");
        if (cfg.target_window < 1) throw std::runtime_error("target window must be >= 1");
        if (cfg.edge_target_window < 1) throw std::runtime_error("edge target window must be >= 1");
        if (cfg.edge_frame_count < 0) throw std::runtime_error("edge frame count must be >= 0");
        if (cfg.anchor_frame < 1 || cfg.anchor_frame > cfg.frames)
            throw std::runtime_error("anchor frame must be within [1, frames]");
        if (cfg.threads < 0) throw std::runtime_error("threads must be >= 0");
        if (cfg.ransac_iterations < 0) throw std::runtime_error("ransac iterations must be >= 0");
        if (cfg.icp_min_range_m < 0.0f || cfg.icp_max_range_m <= cfg.icp_min_range_m)
            throw std::runtime_error("invalid ICP range limits");
        if (cfg.icp_voxel_leaf_m < 0.0f) throw std::runtime_error("voxel leaf must be >= 0");
        if (cfg.max_icp_correction_translation_m <= 0.0f || cfg.max_icp_correction_rotation_deg <= 0.0f ||
            cfg.edge_max_icp_correction_translation_m <= 0.0f || cfg.edge_max_icp_correction_rotation_deg <= 0.0f)
            throw std::runtime_error("ICP correction guards must be > 0");
        if (!(cfg.soft_rejected_damping > 0.0f && cfg.soft_rejected_damping <= 1.0f) ||
            !(cfg.edge_soft_rejected_damping > 0.0f && cfg.edge_soft_rejected_damping <= 1.0f))
            throw std::runtime_error("soft rejected damping must satisfy 0 < damping <= 1");
        if (cfg.max_applied_icp_translation_m <= 0.0f || cfg.max_applied_icp_rotation_deg <= 0.0f ||
            cfg.edge_max_applied_icp_translation_m <= 0.0f || cfg.edge_max_applied_icp_rotation_deg <= 0.0f)
            throw std::runtime_error("soft applied ICP caps must be > 0");

        if (cfg.quality_eval_distance_m <= 0.0f) throw std::runtime_error("quality eval distance must be > 0");
        if (cfg.min_overlap_ratio < 0.0f || cfg.min_overlap_ratio > 1.0f) throw std::runtime_error("min overlap ratio must be in [0,1]");
        if (cfg.max_alignment_rmse_m <= 0.0f) throw std::runtime_error("max alignment RMSE must be > 0");
        if (cfg.local_refine_iterations < 0) throw std::runtime_error("local refine iterations must be >= 0");
        if (cfg.local_refine_target_window < 1) throw std::runtime_error("local refine target window must be >= 1");

        const float offset_norm = std::sqrt(
            cfg.offset_x_m * cfg.offset_x_m +
            cfg.offset_y_m * cfg.offset_y_m +
            cfg.offset_z_m * cfg.offset_z_m);
        if (offset_norm <= 0.0f)
            throw std::runtime_error("lidar offset vector magnitude must be > 0");
        return true;
    }
}

int main(int argc, char** argv)
{
    Config cfg;
    try
    {
        if (!parseArguments(argc, argv, cfg))
            return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ARG ERROR] " << e.what() << std::endl;
        printUsage(argv[0]);
        return 2;
    }

    const auto total_start = std::chrono::steady_clock::now();
    if (cfg.threads > 0)
        omp_set_num_threads(cfg.threads);
    cfg.threads = std::max(1, omp_get_max_threads());

    const float effective_pitch_y_deg = resolvedPitchYDeg(cfg);
    const float effective_yaw_step_deg =
        (cfg.yaw_ccw ? 1.0f : -1.0f) * resolvedYawStepDeg(cfg);

    std::cout << "========== Scheme 3: sequential accumulated-map point-to-plane ICP + latest mechanical coarse pose ==========" << std::endl;
    std::cout << "frames=" << cfg.frames
        << ", prefix=" << cfg.input_prefix
        << ", pitch_axis=" << cfg.pitch_axis_deg << " deg"
        << ", effective Ry=" << effective_pitch_y_deg << " deg"
        << ", yaw step=" << effective_yaw_step_deg << " deg"
        << ", yaw mode=" << (cfg.auto_yaw_step ? "auto(360/frames)" : "manual")
        << ", offset=(" << cfg.offset_x_m << ", " << cfg.offset_y_m << ", " << cfg.offset_z_m << ") m\n";
    std::cout << "Mode: scheme 3 accumulated-map registration. frame1 is fixed; "
        << "each frame starts from its direct mechanical coarse pose and registers against all previously refined frames."
        << std::endl;
    std::cout << "ICP-only preprocessing=" << (cfg.enable_icp_filter ? "ON" : "OFF")
        << ", voxel=" << cfg.icp_voxel_leaf_m
        << " m, SOR K=" << cfg.icp_sor_mean_k
        << ", ROR=" << (cfg.enable_ror ? "ON" : "OFF")
        << ", hypothesis angle=" << cfg.hypothesis_angle_deg << " deg"
        << ", edge target window=" << cfg.edge_target_window
        << ", edge hypothesis=" << cfg.edge_hypothesis_angle_deg << " deg"
        << ", normal guard=" << cfg.max_icp_correction_translation_m << " m / "
        << cfg.max_icp_correction_rotation_deg << " deg"
        << ", edge guard=" << cfg.edge_max_icp_correction_translation_m << " m / "
        << cfg.edge_max_icp_correction_rotation_deg << " deg"
        << ", quality gate=" << (cfg.enable_quality_gate ? "ON" : "OFF")
        << " (eval_dist=" << cfg.quality_eval_distance_m
        << " m, min_overlap=" << cfg.min_overlap_ratio
        << ", max_rmse=" << cfg.max_alignment_rmse_m << " m)"
        << ", local_refine_iterations=" << cfg.local_refine_iterations
        << std::endl;

    // ── Timing accumulators ──────────────────────────────────────────
    double t_load_s = 0.0, t_preprocess_s = 0.0, t_coarse_pose_s = 0.0;
    double t_accumulated_icp_s = 0.0, t_local_refine_s = 0.0;
    double t_loop_check_s = 0.0, t_transform_save_s = 0.0;
    std::vector<double> per_frame_load_s;
    std::vector<double> per_frame_icp_s;
    int icp_pair_count = 0;

    std::vector<CloudT::Ptr> raw_clouds;
    std::vector<CloudT::Ptr> icp_clouds;
    raw_clouds.reserve(cfg.frames);
    icp_clouds.reserve(cfg.frames);

    const auto load_all_start = std::chrono::steady_clock::now();
    for (int frame = 1; frame <= cfg.frames; ++frame)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string filename = buildFilename(frame, cfg);
        CloudT::Ptr raw(new CloudT);
        if (pcl::io::loadPCDFile<PointT>(filename, *raw) < 0)
        {
            std::cerr << "[ERROR] Failed to load " << filename << std::endl;
            return 3;
        }
        if (raw->size() < 100)
        {
            std::cerr << "[ERROR] Too few raw points in " << filename << std::endl;
            return 4;
        }
        const auto t1 = std::chrono::steady_clock::now();
        CloudT::Ptr icp_cloud = preprocessForICP(raw, cfg);
        if (icp_cloud->size() < 80)
        {
            std::cerr << "[ERROR] Too few ICP points after preprocessing in " << filename
                << ": " << icp_cloud->size() << std::endl;
            return 5;
        }
        const auto t2 = std::chrono::steady_clock::now();
        raw_clouds.push_back(raw);
        icp_clouds.push_back(icp_cloud);
        const double ld = std::chrono::duration<double>(t1 - t0).count();
        const double pd = std::chrono::duration<double>(t2 - t1).count();
        per_frame_load_s.push_back(ld + pd);
        t_load_s += ld;
        t_preprocess_s += pd;
        std::cout << "[LOAD] " << filename
            << ": raw=" << raw->size()
            << ", icp_preprocessed=" << icp_cloud->size()
            << ", load=" << ld << " s, preprocess=" << pd << " s" << std::endl;
    }
    t_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_all_start).count() - t_preprocess_s;

    const auto coarse_pose_start = std::chrono::steady_clock::now();
    std::vector<Eigen::Matrix4f> direct_poses(cfg.frames, Eigen::Matrix4f::Identity());
    for (int i = 0; i < cfg.frames; ++i)
        direct_poses[i] = makeDirectPoseToFrame1(i, cfg);
    t_coarse_pose_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - coarse_pose_start).count();


    std::vector<Eigen::Matrix4f> refined_poses = direct_poses;
    std::vector<std::uint8_t> has_refined_pose(cfg.frames, 0);

    const int anchor_index = 0;
    refined_poses[anchor_index] = direct_poses[anchor_index];
    has_refined_pose[anchor_index] = 1;

    std::ofstream csv(cfg.csv_file);
    if (!csv)
    {
        std::cerr << "[ERROR] Failed to create CSV: " << cfg.csv_file << std::endl;
        return 6;
    }

    csv << "frame,direction,predicted_from,target_frames,icp_accepted,status,fitness,source_points,target_points,"
        "icp_translation_m,icp_rotation_deg,selection_score,pose_from_direct_translation_m,pose_from_direct_rotation_deg";
    for (const std::string& prefix : { "direct_pose", "predicted_pose", "icp_correction", "refined_pose" })
    {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                csv << ',' << prefix << "_m" << r << c;
    }
    csv << '\n';

    const std::string coarse_dir = "coarse_aligned_scheme3_accumulated";
    const std::string refined_dir = "refined_aligned_scheme3_accumulated";
    const std::string debug_dir = "icp_debug_scheme3_accumulated";

    if (cfg.save_aligned_frames)
    {
        if (!ensureDirectory(coarse_dir) || !ensureDirectory(refined_dir))
        {
            std::cerr << "[ERROR] Failed to create output directories. errno=" << errno << std::endl;
            return 7;
        }
    }

    if (cfg.save_icp_debug_clouds && !ensureDirectory(debug_dir))
    {
        std::cerr << "[ERROR] Failed to create debug directory. errno=" << errno << std::endl;
        return 8;
    }

    auto indicesToText = [](const std::vector<int>& indices) -> std::string
        {
            if (indices.empty())
                return "none";

            std::ostringstream oss;
            for (std::size_t k = 0; k < indices.size(); ++k)
            {
                if (k > 0)
                    oss << "|";
                oss << indices[k] + 1;
            }
            return oss.str();
        };

    auto writeCsvRow = [&](int frame_index,
        const std::string& direction,
        int predicted_from,
        const std::vector<int>& target_indices,
        const ICPResult& icp,
        const Eigen::Matrix4f& predicted_pose)
        {
            const Eigen::Matrix4f total_from_direct =
                refined_poses[frame_index] *
                direct_poses[frame_index].inverse();

            csv << frame_index + 1 << ','
                << direction << ','
                << (predicted_from >= 0 ? predicted_from + 1 : 0) << ','
                << indicesToText(target_indices) << ','
                << (icp.accepted ? 1 : 0) << ','
                << icp.status << ','
                << std::setprecision(10) << icp.fitness << ','
                << icp.source_points << ','
                << icp.target_points << ','
                << icp.translation_m << ','
                << icp.rotation_deg << ','
                << icp.selection_score << ','
                << translationMagnitude(total_from_direct) << ','
                << rotationMagnitudeDeg(total_from_direct);

            writeMatrixCSV(csv, direct_poses[frame_index]);
            writeMatrixCSV(csv, predicted_pose);
            writeMatrixCSV(
                csv,
                icp.accepted
                ? icp.correction
                : Eigen::Matrix4f::Identity());
            writeMatrixCSV(csv, refined_poses[frame_index]);
            csv << '\n';
        };


    {
        ICPResult anchor_result;
        anchor_result.accepted = true;
        anchor_result.converged = true;
        anchor_result.status = "anchor_frame1";
        anchor_result.source_points = icp_clouds[0]->size();
        anchor_result.target_points = 0;

        writeCsvRow(
            0,
            "anchor",
            -1,
            {},
            anchor_result,
            direct_poses[0]);

        std::cout
            << "[ANCHOR] frame=1 is fixed as the global reference."
            << std::endl;
    }

    // frame2 -> frame1
    // frame3 -> frame1+frame2
    // ...
    // frameN -> frame1+...+frame(N-1)
    const auto accumulated_icp_start =
        std::chrono::steady_clock::now();

    for (int current_index = 1;
         current_index < cfg.frames;
         ++current_index)
    {
        std::cout
            << "\n================ Scheme3 accumulated frame "
            << current_index + 1
            << " / "
            << cfg.frames
            << " ================"
            << std::endl;


        const Eigen::Matrix4f predicted_pose =
            direct_poses[current_index];

        std::vector<int> target_indices;
        target_indices.reserve(
            static_cast<std::size_t>(current_index));

        for (int j = 0; j < current_index; ++j)
        {
            if (has_refined_pose[j])
                target_indices.push_back(j);
        }

        if (target_indices.empty())
        {
            std::cerr
                << "[ERROR] No accumulated target frames available for frame "
                << current_index + 1
                << std::endl;
            return 9;
        }

        CloudT::Ptr source_global_predicted =
            transformedCloud(
                icp_clouds[current_index],
                predicted_pose);

        CloudT::Ptr target_global =
            buildTargetMapFromIndices(
                icp_clouds,
                refined_poses,
                target_indices);

        const bool edge_frame =
            (current_index < cfg.edge_frame_count) ||
            (current_index >= cfg.frames - cfg.edge_frame_count);

        if (cfg.save_icp_debug_clouds)
        {
            std::ostringstream sp;
            std::ostringstream tp;

            sp << debug_dir
                << "/frame_"
                << std::setw(2)
                << std::setfill('0')
                << current_index + 1
                << "_source_direct_coarse.pcd";

            tp << debug_dir
                << "/frame_"
                << std::setw(2)
                << std::setfill('0')
                << current_index + 1
                << "_accumulated_target_01_to_"
                << std::setw(2)
                << std::setfill('0')
                << current_index
                << ".pcd";

            pcl::io::savePCDFileBinary(
                sp.str(),
                *source_global_predicted);

            pcl::io::savePCDFileBinary(
                tp.str(),
                *target_global);
        }

        const auto icp_start =
            std::chrono::steady_clock::now();

        ICPResult icp =
            chooseBestICP(
                source_global_predicted,
                target_global,
                cfg,
                edge_frame);

        const auto icp_end =
            std::chrono::steady_clock::now();

        Eigen::Matrix4f applied_correction =
            Eigen::Matrix4f::Identity();

        std::string pose_source =
            "mechanical_fallback";

        bool used_soft_clamped = false;

        if (icp.accepted)
        {
            applied_correction =
                icp.correction;

            pose_source =
                "icp_full";
        }
        else if (cfg.soft_accept_rejected_icp &&
                 icp.converged)
        {
            const float damping =
                edge_frame
                ? cfg.edge_soft_rejected_damping
                : cfg.soft_rejected_damping;

            const float max_t =
                edge_frame
                ? cfg.edge_max_applied_icp_translation_m
                : cfg.max_applied_icp_translation_m;

            const float max_r =
                edge_frame
                ? cfg.edge_max_applied_icp_rotation_deg
                : cfg.max_applied_icp_rotation_deg;

            applied_correction =
                limitCorrection(
                    interpolateCorrection(
                        icp.correction,
                        damping),
                    max_t,
                    max_r);

            used_soft_clamped = true;
            pose_source = "icp_soft_clamped";
        }

        const Eigen::Matrix4f proposed_pose =
            applied_correction *
            predicted_pose;

        const Eigen::Matrix4f proposed_total_from_direct =
            proposed_pose *
            direct_poses[current_index].inverse();

        const float proposed_total_t =
            translationMagnitude(
                proposed_total_from_direct);

        const float proposed_total_r =
            rotationMagnitudeDeg(
                proposed_total_from_direct);

        bool total_guard_ok = true;

        if (cfg.max_total_from_direct_translation_m > 0.0f &&
            proposed_total_t >
                cfg.max_total_from_direct_translation_m)
        {
            total_guard_ok = false;
        }

        if (cfg.max_total_from_direct_rotation_deg > 0.0f &&
            proposed_total_r >
                cfg.max_total_from_direct_rotation_deg)
        {
            total_guard_ok = false;
        }

        if ((icp.accepted || used_soft_clamped) &&
            total_guard_ok)
        {
            refined_poses[current_index] =
                proposed_pose;

            if (used_soft_clamped)
                icp.status = "soft_accepted_after_guard";
        }
        else
        {

            refined_poses[current_index] =
                direct_poses[current_index];

            if (icp.accepted || used_soft_clamped)
            {
                icp.accepted = false;
                icp.status =
                    "rejected_by_total_from_direct_guard_after_soft";
            }

            if (!cfg.fallback_to_mechanical)
            {
                std::cerr
                    << "[ERROR] ICP failed and fallback disabled at frame "
                    << current_index + 1
                    << std::endl;
                return 10;
            }
        }

        has_refined_pose[current_index] = 1;

        const Eigen::Matrix4f total_from_direct =
            refined_poses[current_index] *
            direct_poses[current_index].inverse();

        std::cout
            << "[SCHEME3 ICP] frame="
            << current_index + 1
            << ", predicted_from=direct_mechanical"
            << ", target_frames="
            << indicesToText(target_indices)
            << ", accepted="
            << ((icp.accepted || used_soft_clamped) &&
                total_guard_ok
                ? 1
                : 0)
            << ", status="
            << icp.status
            << ", pose_source="
            << pose_source
            << ", fitness="
            << icp.fitness
            << ", score="
            << icp.selection_score
            << ", overlap="
            << icp.overlap_ratio
            << ", rmse="
            << icp.rmse_m
            << ", base_overlap="
            << icp.base_overlap_ratio
            << ", base_rmse="
            << icp.base_rmse_m
            << ", source="
            << icp.source_points
            << ", target="
            << icp.target_points
            << ", icp_candidate="
            << icp.translation_m
            << " m / "
            << icp.rotation_deg
            << " deg"
            << ", applied_correction="
            << translationMagnitude(applied_correction)
            << " m / "
            << rotationMagnitudeDeg(applied_correction)
            << " deg"
            << ", total_from_direct="
            << translationMagnitude(total_from_direct)
            << " m / "
            << rotationMagnitudeDeg(total_from_direct)
            << " deg"
            << ", time="
            << std::chrono::duration<double>(
                   icp_end - icp_start).count()
            << " s"
            << std::endl;

        writeCsvRow(
            current_index,
            "accumulated_map",
            -1,
            target_indices,
            icp,
            predicted_pose);
    }

    t_accumulated_icp_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            accumulated_icp_start).count();


    t_local_refine_s = 0.0;

    if (cfg.check_loop_closure && cfg.frames >= 3)
    {
        std::cout << "\n================ Loop closure diagnostic only ================" << std::endl;
        const auto loop_start = std::chrono::steady_clock::now();
        CloudT::Ptr source_last = transformedCloud(icp_clouds.back(), refined_poses.back());
        CloudT::Ptr target_first = transformedCloud(icp_clouds.front(), refined_poses.front());
        ICPResult loop = chooseBestICP(source_last, target_first, cfg, true);
        t_loop_check_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
        std::cout << "[LOOP CHECK] last -> first, accepted=" << (loop.accepted ? 1 : 0)
            << ", status=" << loop.status
            << ", correction=" << loop.translation_m << " m / "
            << loop.rotation_deg << " deg, fitness=" << loop.fitness
            << ", time=" << t_loop_check_s << " s" << std::endl;
        std::cout << "[LOOP CHECK] This program does NOT force-distribute loop closure. If this is large, the last-first overlap is still unreliable." << std::endl;
    }

    csv.close();

    const auto save_start = std::chrono::steady_clock::now();
    CloudT::Ptr coarse_global(new CloudT);
    CloudT::Ptr refined_global(new CloudT);
    for (int i = 0; i < cfg.frames; ++i)
    {
        CloudT::Ptr coarse_aligned = transformedCloud(raw_clouds[i], direct_poses[i]);
        CloudT::Ptr refined_aligned = transformedCloud(raw_clouds[i], refined_poses[i]);
        *coarse_global += *coarse_aligned;
        *refined_global += *refined_aligned;

        if (cfg.save_aligned_frames)
        {
            std::ostringstream cp, rp;
            cp << coarse_dir << "/" << cfg.input_prefix
                << std::setw(cfg.input_index_width > 0 ? cfg.input_index_width : 0)
                << std::setfill('0') << i + 1 << "_coarse.pcd";
            rp << refined_dir << "/" << cfg.input_prefix
                << std::setw(cfg.input_index_width > 0 ? cfg.input_index_width : 0)
                << std::setfill('0') << i + 1 << "_refined.pcd";
            pcl::io::savePCDFileBinary(cp.str(), *coarse_aligned);
            pcl::io::savePCDFileBinary(rp.str(), *refined_aligned);
        }

        const Eigen::Matrix4f total_from_direct = refined_poses[i] * direct_poses[i].inverse();
        std::cout << "[FINAL POSE] frame=" << i + 1
            << ", total_from_direct_translation=" << translationMagnitude(total_from_direct)
            << " m, total_from_direct_rotation=" << rotationMagnitudeDeg(total_from_direct) << " deg"
            << std::endl;
    }

    if (pcl::io::savePCDFileBinary(cfg.coarse_output_file, *coarse_global) < 0)
    {
        std::cerr << "[ERROR] Failed to save coarse output: " << cfg.coarse_output_file << std::endl;
        return 10;
    }
    if (pcl::io::savePCDFileBinary(cfg.output_file, *refined_global) < 0)
    {
        std::cerr << "[ERROR] Failed to save refined output: " << cfg.output_file << std::endl;
        return 11;
    }

    const auto total_end = std::chrono::steady_clock::now();
    t_transform_save_s = std::chrono::duration<double>(total_end - save_start).count();

    const double total_s = std::chrono::duration<double>(total_end - total_start).count();
    const double t_registration_s = total_s - t_load_s - t_preprocess_s;

    std::cout << "\n========== Scheme 3 accumulated-map registration complete ==========" << std::endl;
    std::cout << "Coarse cloud: " << cfg.coarse_output_file << ", points=" << coarse_global->size() << std::endl;
    std::cout << "Refined cloud: " << cfg.output_file << ", points=" << refined_global->size() << std::endl;
    std::cout << "CSV log: " << cfg.csv_file << std::endl;
    if (cfg.save_aligned_frames)
    {
        std::cout << "Coarse per-frame: " << coarse_dir << "/" << std::endl;
        std::cout << "Refined per-frame: " << refined_dir << "/" << std::endl;
    }

    // ── Timing summary ───────────────────────────────────────────────
    std::cout << "\n==================== TIMING REPORT ====================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Load PCD files:          " << t_load_s << " s" << std::endl;
    std::cout << "  Preprocess for ICP:      " << t_preprocess_s << " s" << std::endl;
    std::cout << "  Coarse pose computation: " << t_coarse_pose_s << " s" << std::endl;
    std::cout << "  Accumulated-map ICP:     " << t_accumulated_icp_s << " s" << std::endl;
    std::cout << "  Local refine:            " << t_local_refine_s << " s" << std::endl;
    std::cout << "  Loop closure check:      " << t_loop_check_s << " s" << std::endl;
    std::cout << "  Transform + save:        " << t_transform_save_s << " s" << std::endl;
    std::cout << "  ─────────────────────────────────────" << std::endl;
    std::cout << "  Total registration time: " << total_s << " s" << std::endl;
    std::cout << "  Pure ICP time (bidir+refine+loop): "
        << (t_accumulated_icp_s + t_loop_check_s) << " s" << std::endl;
    std::cout << "  Coarse output points: " << coarse_global->size() << std::endl;
    std::cout << "  Refined output points: " << refined_global->size() << std::endl;
    std::cout << "  Frames: " << cfg.frames << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    // ── Write timing JSON report ─────────────────────────────────────
    {
        const std::string timing_file = "registration_timing_report.json";
        std::ofstream tr(timing_file);
        if (tr)
        {
            tr << std::fixed << std::setprecision(6);
            tr << "{\n";
            tr << "  \"frames\": " << cfg.frames << ",\n";
            tr << "  \"pitch_axis_deg\": " << cfg.pitch_axis_deg << ",\n";
            tr << "  \"effective_pitch_y_deg\": " << resolvedPitchYDeg(cfg) << ",\n";
            tr << "  \"yaw_step_deg\": " << (cfg.yaw_ccw ? 1.0f : -1.0f) * resolvedYawStepDeg(cfg) << ",\n";
            tr << "  \"auto_yaw_step\": " << (cfg.auto_yaw_step ? "true" : "false") << ",\n";
            tr << "  \"offset_x_m\": " << cfg.offset_x_m << ",\n";
            tr << "  \"offset_y_m\": " << cfg.offset_y_m << ",\n";
            tr << "  \"offset_z_m\": " << cfg.offset_z_m << ",\n";
            tr << "  \"anchor_frame\": 1,\n";
            tr << "  \"timing_seconds\": {\n";
            tr << "    \"load_pcd\": " << t_load_s << ",\n";
            tr << "    \"preprocess_icp\": " << t_preprocess_s << ",\n";
            tr << "    \"coarse_pose_computation\": " << t_coarse_pose_s << ",\n";
            tr << "    \"accumulated_map_icp\": " << t_accumulated_icp_s << ",\n";
            tr << "    \"local_refine\": " << t_local_refine_s << ",\n";
            tr << "    \"loop_closure_check\": " << t_loop_check_s << ",\n";
            tr << "    \"transform_and_save\": " << t_transform_save_s << ",\n";
            tr << "    \"total\": " << total_s << "\n";
            tr << "  },\n";
            tr << "  \"output\": {\n";
            tr << "    \"coarse_points\": " << coarse_global->size() << ",\n";
            tr << "    \"refined_points\": " << refined_global->size() << ",\n";
            tr << "    \"coarse_file\": \"" << cfg.coarse_output_file << "\",\n";
            tr << "    \"refined_file\": \"" << cfg.output_file << "\"\n";
            tr << "  }\n";
            tr << "}\n";
            tr.close();
            std::cout << "[TIMING REPORT] saved to: " << timing_file << std::endl;
        }
    }
    return 0;
}
