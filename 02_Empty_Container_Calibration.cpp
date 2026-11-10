#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include <random>
#include <cstdint>

#include <Eigen/Eigenvalues>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;


const std::string INPUT_FILE = "empty_box_raw.pcd";
const std::string CALIB_TRANSFORM_FILE = "calib_transform.txt";


const float ROUGH_X_MIN = -1.0f;
const float ROUGH_X_MAX = 1.0f;
const float ROUGH_Y_MIN = -1.0f;
const float ROUGH_Y_MAX = 1.0f;
const float ROUGH_Z_MIN = -1.0f;
const float ROUGH_Z_MAX = 1.2f;


const float WALL_Z_MIN = 0.10f;
const float WALL_Z_MAX = 0.42f;

const float BOX_LENGTH_NOMINAL = 1.00f;
const float BOX_WIDTH_NOMINAL = 0.60f;
const float BOX_HEIGHT_NOMINAL = 0.40f;


const int   WALL_MAX_PLANES = 6;
const int   WALL_MIN_INLIERS = 80;
const float WALL_DISTANCE_THRESHOLD = 0.018f;
const float WALL_EPS_ANGLE_DEG = 8.0f;
const float MAX_WALL_NORMAL_Z = 0.30f;


const int   CENTER_MAX_PLANES = 8;
const int   CENTER_MIN_INLIERS = 60;
const float CENTER_DIST_TOL = 0.10f;


const float YAW_MANUAL_OFFSET_DEG = 0.0f;


const float WALL_CHECK_BAND = 0.04f;
const float WALL_CHECK_Z_MIN = 0.08f;
const float WALL_CHECK_Z_MAX = 0.45f;


const float BOTTOM_Z_MIN = -0.05f;
const float BOTTOM_Z_MAX = 0.08f;


const float BOTTOM_DISTANCE_THRESHOLD = 0.012f;
const float BOTTOM_EPS_ANGLE_DEG = 10.0f;
const int   BOTTOM_MIN_INLIERS = 100;

const float BOTTOM_GRID_SIZE = 0.01f;



float deg2rad(float deg)
{
    return deg * static_cast<float>(M_PI) / 180.0f;
}

float getMedian(std::vector<float> values)
{
    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());
    size_t n = values.size();

    if (n % 2 == 1)
        return values[n / 2];

    return 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

CloudT::Ptr cropAxis(const CloudT::Ptr& cloud, const std::string& axis, float min_v, float max_v)
{
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    for (const auto& p : cloud->points)
    {
        float value = 0.0f;
        if (axis == "x") value = p.x;
        else if (axis == "y") value = p.y;
        else value = p.z;
        if (std::isfinite(value) && value >= min_v && value <= max_v)
            out->points.push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = true;

    return out;
}

CloudT::Ptr cropXYZ(
    const CloudT::Ptr& cloud,
    float xmin, float xmax,
    float ymin, float ymax,
    float zmin, float zmax
)
{
    CloudT::Ptr out = cropAxis(cloud, "x", xmin, xmax);
    out = cropAxis(out, "y", ymin, ymax);
    out = cropAxis(out, "z", zmin, zmax);
    return out;
}

struct PlaneInfo
{
    Eigen::Vector3f normal;
    int inliers;
};

struct AxisPlane
{
    char axis;
    float pos;
    int inliers;
    CloudT::Ptr cloud;
};

struct PairResult
{
    bool ok = false;
    int i = -1;
    int j = -1;
    float p1 = 0.0f;
    float p2 = 0.0f;
    float center = 0.0f;
    float dist = 0.0f;
    float error = std::numeric_limits<float>::max();
    int total_inliers = 0;
};

struct BottomCalib
{
    bool ok = false;

    float a = 0.0f;
    float b = 0.0f;
    float c = 1.0f;
    float d = 0.0f;

    float z_mean = 0.0f;
    float z_median = 0.0f;
    float z_min = 0.0f;
    float z_max = 0.0f;
    float z_std = 0.0f;

    int inlier_count = 0;
};


struct PlaneFitResult
{
    bool ok = false;
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    float d = 0.0f;
    std::vector<int> inliers;
};

CloudT::Ptr cloudWithoutIndices(const CloudT::Ptr& cloud, const std::vector<int>& inliers)
{
    std::vector<unsigned char> removed(cloud->size(), 0);
    for (int id : inliers)
    {
        if (id >= 0 && static_cast<size_t>(id) < removed.size())
            removed[static_cast<size_t>(id)] = 1;
    }

    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size() - std::min(cloud->size(), inliers.size()));
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        if (!removed[i])
            out->points.push_back(cloud->points[i]);
    }
    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

PlaneFitResult fitPlaneRansac(
    const CloudT::Ptr& cloud,
    float distance_threshold,
    int iterations,
    float min_abs_normal_z,
    float max_abs_normal_z
)
{
    PlaneFitResult result;
    if (!cloud || cloud->size() < 3)
        return result;

    std::vector<int> candidates;
    const size_t max_candidates = 40000;
    const size_t stride_count = std::min(max_candidates, cloud->size());
    candidates.reserve(stride_count);
    for (size_t i = 0; i < stride_count; ++i)
    {
        const size_t id = (i * cloud->size()) / stride_count;
        if (id < cloud->size())
            candidates.push_back(static_cast<int>(id));
    }
    if (candidates.size() < 3)
        return result;

    std::mt19937 rng(20250724u + static_cast<unsigned>(cloud->size()));
    std::uniform_int_distribution<size_t> pick(0, candidates.size() - 1);
    size_t best_count = 0;
    Eigen::Vector3f best_n = Eigen::Vector3f::UnitZ();
    float best_d = 0.0f;

    for (int it = 0; it < iterations; ++it)
    {
        const int ia = candidates[pick(rng)];
        const int ib = candidates[pick(rng)];
        const int ic = candidates[pick(rng)];
        if (ia == ib || ia == ic || ib == ic)
            continue;

        const Eigen::Vector3f a = cloud->points[ia].getVector3fMap();
        const Eigen::Vector3f b = cloud->points[ib].getVector3fMap();
        const Eigen::Vector3f c = cloud->points[ic].getVector3fMap();
        Eigen::Vector3f n = (b - a).cross(c - a);
        const float n_norm = n.norm();
        if (n_norm < 1e-6f)
            continue;
        n /= n_norm;
        const float abs_nz = std::abs(n.z());
        if (abs_nz < min_abs_normal_z || abs_nz > max_abs_normal_z)
            continue;

        const float d = -n.dot(a);
        size_t count = 0;
        for (const auto& p : cloud->points)
        {
            const float residual = std::abs(n.x() * p.x + n.y() * p.y + n.z() * p.z + d);
            if (residual <= distance_threshold)
                ++count;
        }
        if (count > best_count)
        {
            best_count = count;
            best_n = n;
            best_d = d;
        }
    }

    if (best_count < 3)
        return result;

 
    std::vector<int> initial_inliers;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const auto& p = cloud->points[i];
        const float residual = std::abs(best_n.x() * p.x + best_n.y() * p.y + best_n.z() * p.z + best_d);
        if (residual <= distance_threshold)
        {
            initial_inliers.push_back(static_cast<int>(i));
            centroid += p.getVector3fMap();
        }
    }
    if (initial_inliers.size() < 3)
        return result;
    centroid /= static_cast<float>(initial_inliers.size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int id : initial_inliers)
    {
        const Eigen::Vector3f delta = cloud->points[id].getVector3fMap() - centroid;
        covariance += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() == Eigen::Success)
    {
        best_n = solver.eigenvectors().col(0).normalized();
        const float abs_nz = std::abs(best_n.z());
        if (abs_nz < min_abs_normal_z || abs_nz > max_abs_normal_z)
            return result;
        best_d = -best_n.dot(centroid);
    }

    result.inliers.clear();
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const auto& p = cloud->points[i];
        const float residual = std::abs(best_n.x() * p.x + best_n.y() * p.y + best_n.z() * p.z + best_d);
        if (residual <= distance_threshold)
            result.inliers.push_back(static_cast<int>(i));
    }
    if (result.inliers.size() < 3)
        return result;

    if (best_n.z() < 0.0f)
    {
        best_n = -best_n;
        best_d = -best_d;
    }
    result.ok = true;
    result.normal = best_n;
    result.d = best_d;
    return result;
}

CloudT::Ptr transformCloudEigen(const CloudT::Ptr& cloud, const Eigen::Affine3f& tf)
{
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    for (const auto& p : cloud->points)
    {
        const Eigen::Vector3f q = tf * p.getVector3fMap();
        PointT transformed;
        transformed.x = q.x();
        transformed.y = q.y();
        transformed.z = q.z();
        out->points.push_back(transformed);
    }
    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

bool savePCDAscii(const std::string& filename, const CloudT::Ptr& cloud)
{
    std::ofstream fout(filename);
    if (!fout.is_open())
        return false;

    fout << "# .PCD v0.7 - Point Cloud Data file format\n"
         << "VERSION 0.7\n"
         << "FIELDS x y z\n"
         << "SIZE 4 4 4\n"
         << "TYPE F F F\n"
         << "COUNT 1 1 1\n"
         << "WIDTH " << cloud->points.size() << "\n"
         << "HEIGHT 1\n"
         << "VIEWPOINT 0 0 0 1 0 0 0\n"
         << "POINTS " << cloud->points.size() << "\n"
         << "DATA ascii\n";
    fout << std::setprecision(9);
    for (const auto& p : cloud->points)
        fout << p.x << ' ' << p.y << ' ' << p.z << '\n';
    return static_cast<bool>(fout);
}

bool levelCloud(const CloudT::Ptr& cloud_raw, CloudT::Ptr& cloud_leveled, Eigen::Affine3f& level_tf)
{
    const PlaneFitResult plane = fitPlaneRansac(cloud_raw, 0.02f, 500, std::cos(deg2rad(45.0f)), 1.0f);
    if (!plane.ok || plane.inliers.empty())
        return false;

    Eigen::Vector3f n = plane.normal.normalized();

    Eigen::Quaternionf q;
    q.setFromTwoVectors(n, Eigen::Vector3f(0, 0, 1));

    Eigen::Affine3f rot = Eigen::Affine3f::Identity();
    rot.rotate(q);

    CloudT::Ptr cloud_rot = transformCloudEigen(cloud_raw, rot);

    float avg_z = 0.0f;
    for (int id : plane.inliers)
        avg_z += cloud_rot->points[id].z;

    avg_z /= static_cast<float>(plane.inliers.size());

    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.rotate(q);
    tf.translation() << 0, 0, -avg_z;

    level_tf = tf;
    cloud_leveled = transformCloudEigen(cloud_raw, level_tf);

    return true;
}




std::vector<PlaneInfo> extractWallPlanesForYaw(const CloudT::Ptr& wall_cloud)
{
    std::vector<PlaneInfo> planes;

    CloudT::Ptr remain(new CloudT);
    *remain = *wall_cloud;

    for (int i = 0; i < WALL_MAX_PLANES; ++i)
    {
        const PlaneFitResult plane = fitPlaneRansac(
            remain,
            WALL_DISTANCE_THRESHOLD,
            500,
            0.0f,
            MAX_WALL_NORMAL_Z
        );

        if (!plane.ok || plane.inliers.size() < WALL_MIN_INLIERS)
            break;

        const Eigen::Vector3f n = plane.normal.normalized();

        if (std::abs(n.z()) <= MAX_WALL_NORMAL_Z)
        {
            planes.push_back({
                n,
                static_cast<int>(plane.inliers.size())
                });
        }

        remain = cloudWithoutIndices(remain, plane.inliers);

        if (remain->size() < 100)
            break;
    }

    return planes;
}

bool computeYawRotation(const std::vector<PlaneInfo>& planes, float& rotation_rad)
{
    if (planes.empty())
        return false;

    double sum_sin = 0.0;
    double sum_cos = 0.0;

    for (const auto& p : planes)
    {
        Eigen::Vector3f n = p.normal.normalized();
        float theta = std::atan2(n.y(), n.x());

        double w = static_cast<double>(p.inliers);


        sum_sin += w * std::sin(4.0 * theta);
        sum_cos += w * std::cos(4.0 * theta);
    }

    if (std::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) < 1e-8)
        return false;

    float yaw = 0.25f * static_cast<float>(std::atan2(sum_sin, sum_cos));
    rotation_rad = -yaw + deg2rad(YAW_MANUAL_OFFSET_DEG);

    return true;
}


std::vector<AxisPlane> extractAxisPlanesForCenter(const CloudT::Ptr& cloud)
{
    std::vector<AxisPlane> planes;

    CloudT::Ptr remain(new CloudT);
    *remain = *cloud;

    for (int k = 0; k < CENTER_MAX_PLANES; ++k)
    {
        const PlaneFitResult plane = fitPlaneRansac(
            remain,
            WALL_DISTANCE_THRESHOLD,
            500,
            0.0f,
            MAX_WALL_NORMAL_Z
        );

        if (!plane.ok || plane.inliers.size() < CENTER_MIN_INLIERS)
            break;

        const Eigen::Vector3f n = plane.normal.normalized();

        CloudT::Ptr one_plane(new CloudT);
        one_plane->points.reserve(plane.inliers.size());
        for (int id : plane.inliers)
            one_plane->points.push_back(remain->points[id]);
        one_plane->width = static_cast<std::uint32_t>(one_plane->points.size());
        one_plane->height = 1;
        one_plane->is_dense = true;

        if (std::abs(n.z()) <= MAX_WALL_NORMAL_Z && one_plane->size() > 0)
        {
            char axis;
            float sum = 0.0f;

            if (std::abs(n.x()) >= std::abs(n.y()))
            {
                axis = 'x';
                for (const auto& p : one_plane->points)
                    sum += p.x;
            }
            else
            {
                axis = 'y';
                for (const auto& p : one_plane->points)
                    sum += p.y;
            }

            float pos = sum / static_cast<float>(one_plane->size());

                planes.push_back({
                    axis,
                    pos,
                static_cast<int>(plane.inliers.size()),
                one_plane
                });

            std::cout << "[CALIB] plane " << k + 1
                << " axis = " << axis
                << ", pos = " << pos
                << ", inliers = " << plane.inliers.size()
                << std::endl;
        }

        remain = cloudWithoutIndices(remain, plane.inliers);

        if (remain->size() < 100)
            break;
    }

    return planes;
}

PairResult selectBestOppositePair(
    const std::vector<AxisPlane>& planes,
    char axis,
    float target_distance
)
{
    PairResult best;

    for (size_t i = 0; i < planes.size(); ++i)
    {
        if (planes[i].axis != axis)
            continue;

        for (size_t j = i + 1; j < planes.size(); ++j)
        {
            if (planes[j].axis != axis)
                continue;

            float p1 = planes[i].pos;
            float p2 = planes[j].pos;
            float dist = std::abs(p1 - p2);
            float error = std::abs(dist - target_distance);
            int total_inliers = planes[i].inliers + planes[j].inliers;

            bool better = false;

            if (!best.ok)
                better = true;
            else if (error < best.error)
                better = true;
            else if (std::abs(error - best.error) < 1e-4f &&
                total_inliers > best.total_inliers)
                better = true;

            if (better)
            {
                best.ok = true;
                best.i = static_cast<int>(i);
                best.j = static_cast<int>(j);
                best.p1 = p1;
                best.p2 = p2;
                best.dist = dist;
                best.center = 0.5f * (p1 + p2);
                best.error = error;
                best.total_inliers = total_inliers;
            }
        }
    }

    if (best.ok && best.error > CENTER_DIST_TOL)
    {
        std::cout << "[REJECT] Best pair on axis " << axis
            << " rejected. distance error = " << best.error
            << " m" << std::endl;

        best.ok = false;
    }

    return best;
}

bool estimateCalibrationByWallPairs(
    const std::vector<AxisPlane>& planes,
    float& cx,
    float& cy,
    bool& long_axis_in_x,
    PairResult& final_x_pair,
    PairResult& final_y_pair
)
{

    PairResult xA = selectBestOppositePair(planes, 'x', BOX_LENGTH_NOMINAL);
    PairResult yA = selectBestOppositePair(planes, 'y', BOX_WIDTH_NOMINAL);


    PairResult xB = selectBestOppositePair(planes, 'x', BOX_WIDTH_NOMINAL);
    PairResult yB = selectBestOppositePair(planes, 'y', BOX_LENGTH_NOMINAL);

    bool A_ok = xA.ok && yA.ok;
    bool B_ok = xB.ok && yB.ok;

    if (!A_ok && !B_ok)
    {
        std::cerr << "[ERROR] Cannot find two valid opposite wall pairs." << std::endl;
        std::cerr << "[HINT] For calibration, both X and Y wall pairs are required." << std::endl;
        return false;
    }

    float scoreA = A_ok ? xA.error + yA.error : std::numeric_limits<float>::max();
    float scoreB = B_ok ? xB.error + yB.error : std::numeric_limits<float>::max();

    if (scoreA <= scoreB)
    {
        long_axis_in_x = true;
        cx = xA.center;
        cy = yA.center;
        final_x_pair = xA;
        final_y_pair = yA;

        std::cout << "[CALIB] Layout A selected: X ≈ 1.0 m, Y ≈ 0.6 m" << std::endl;
    }
    else
    {
        long_axis_in_x = false;
        cx = xB.center;
        cy = yB.center;
        final_x_pair = xB;
        final_y_pair = yB;

        std::cout << "[CALIB] Layout B selected: X ≈ 0.6 m, Y ≈ 1.0 m" << std::endl;
    }

    std::cout << "[CALIB] X pair distance = " << final_x_pair.dist
        << ", error = " << final_x_pair.error << std::endl;

    std::cout << "[CALIB] Y pair distance = " << final_y_pair.dist
        << ", error = " << final_y_pair.error << std::endl;

    std::cout << "[CALIB] center before correction: cx = "
        << cx << ", cy = " << cy << std::endl;

    return true;
}




CloudT::Ptr extractCleanBoxWallsForCheck(
    const CloudT::Ptr& cloud,
    float calib_x_half,
    float calib_y_half
)
{
    CloudT::Ptr walls(new CloudT);

    for (const auto& p : cloud->points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (p.z < WALL_CHECK_Z_MIN || p.z > WALL_CHECK_Z_MAX)
            continue;

        bool near_x_wall =
            std::abs(std::abs(p.x) - calib_x_half) <= WALL_CHECK_BAND &&
            std::abs(p.y) <= calib_y_half + WALL_CHECK_BAND;

        bool near_y_wall =
            std::abs(std::abs(p.y) - calib_y_half) <= WALL_CHECK_BAND &&
            std::abs(p.x) <= calib_x_half + WALL_CHECK_BAND;

        if (near_x_wall || near_y_wall)
        {
            walls->points.push_back(p);
        }
    }

    walls->width = static_cast<uint32_t>(walls->points.size());
    walls->height = 1;
    walls->is_dense = false;

    return walls;
}




BottomCalib fitBottomPlane(
    const CloudT::Ptr& cloud_centered,
    float calib_x_half,
    float calib_y_half,
    CloudT::Ptr& bottom_inliers_cloud
)
{
    BottomCalib result;

    bottom_inliers_cloud.reset(new CloudT);

    CloudT::Ptr bottom_candidate = cropXYZ(
        cloud_centered,
        -calib_x_half,
        calib_x_half,
        -calib_y_half,
        calib_y_half,
        BOTTOM_Z_MIN,
        BOTTOM_Z_MAX
    );

    savePCDAscii("bottom_candidates.pcd", bottom_candidate);

    if (bottom_candidate->size() < BOTTOM_MIN_INLIERS)
    {
        std::cerr << "[ERROR] Too few bottom candidate points: "
            << bottom_candidate->size() << std::endl;
        return result;
    }

    const PlaneFitResult plane = fitPlaneRansac(
        bottom_candidate,
        BOTTOM_DISTANCE_THRESHOLD,
        600,
        std::cos(deg2rad(BOTTOM_EPS_ANGLE_DEG)),
        1.0f
    );

    if (!plane.ok || plane.inliers.size() < BOTTOM_MIN_INLIERS)
    {
        std::cerr << "[ERROR] Bottom plane fitting failed. Inliers = "
            << plane.inliers.size() << std::endl;
        return result;
    }

    bottom_inliers_cloud->points.reserve(plane.inliers.size());
    for (int id : plane.inliers)
        bottom_inliers_cloud->points.push_back(bottom_candidate->points[id]);
    bottom_inliers_cloud->width = static_cast<std::uint32_t>(bottom_inliers_cloud->points.size());
    bottom_inliers_cloud->height = 1;
    bottom_inliers_cloud->is_dense = true;

    float a = plane.normal.x();
    float b = plane.normal.y();
    float c = plane.normal.z();
    float d = plane.d;

    float norm = std::sqrt(a * a + b * b + c * c);
    if (norm < 1e-6f)
        return result;

    a /= norm;
    b /= norm;
    c /= norm;
    d /= norm;

    if (c < 0)
    {
        a = -a;
        b = -b;
        c = -c;
        d = -d;
    }

    std::vector<float> zs;
    zs.reserve(bottom_inliers_cloud->size());

    float sum_z = 0.0f;
    float z_min = std::numeric_limits<float>::max();
    float z_max = -std::numeric_limits<float>::max();

    for (const auto& p : bottom_inliers_cloud->points)
    {
        zs.push_back(p.z);
        sum_z += p.z;
        z_min = std::min(z_min, p.z);
        z_max = std::max(z_max, p.z);
    }

    float mean_z = sum_z / static_cast<float>(zs.size());

    float var = 0.0f;
    for (float z : zs)
    {
        float e = z - mean_z;
        var += e * e;
    }
    var /= static_cast<float>(zs.size());

    result.ok = true;
    result.a = a;
    result.b = b;
    result.c = c;
    result.d = d;
    result.z_mean = mean_z;
    result.z_median = getMedian(zs);
    result.z_min = z_min;
    result.z_max = z_max;
    result.z_std = std::sqrt(var);
    result.inlier_count = static_cast<int>(bottom_inliers_cloud->size());

    std::cout << "[BOTTOM] Plane: "
        << result.a << " x + "
        << result.b << " y + "
        << result.c << " z + "
        << result.d << " = 0" << std::endl;

    std::cout << "[BOTTOM] Inliers = " << result.inlier_count << std::endl;
    std::cout << "[BOTTOM] z_mean = " << result.z_mean
        << ", z_median = " << result.z_median
        << ", z_std = " << result.z_std << std::endl;

    return result;
}

bool saveBottomHeightGrid(
    const std::string& filename,
    const CloudT::Ptr& bottom_cloud,
    float calib_x_half,
    float calib_y_half,
    float grid_size,
    float default_z
)
{
    int nx = static_cast<int>(std::ceil((2.0f * calib_x_half) / grid_size));
    int ny = static_cast<int>(std::ceil((2.0f * calib_y_half) / grid_size));

    if (nx <= 0 || ny <= 0)
        return false;

    std::vector<std::vector<std::vector<float>>> grid(
        nx,
        std::vector<std::vector<float>>(ny)
    );

    for (const auto& p : bottom_cloud->points)
    {
        if (p.x < -calib_x_half || p.x > calib_x_half ||
            p.y < -calib_y_half || p.y > calib_y_half)
            continue;

        int ix = static_cast<int>((p.x + calib_x_half) / grid_size);
        int iy = static_cast<int>((p.y + calib_y_half) / grid_size);

        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny)
        {
            grid[ix][iy].push_back(p.z);
        }
    }

    std::ofstream fout(filename);

    if (!fout.is_open())
    {
        std::cerr << "[ERROR] Cannot write " << filename << std::endl;
        return false;
    }

    fout << "ix,iy,x_center,y_center,z_bottom,filled,num_points\n";

    int filled_count = 0;

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            float x_center = -calib_x_half + (ix + 0.5f) * grid_size;
            float y_center = -calib_y_half + (iy + 0.5f) * grid_size;

            bool filled = !grid[ix][iy].empty();
            float z_bottom = default_z;

            if (filled)
            {
                z_bottom = getMedian(grid[ix][iy]);
                filled_count++;
            }

            fout << ix << ","
                << iy << ","
                << x_center << ","
                << y_center << ","
                << z_bottom << ","
                << (filled ? 1 : 0) << ","
                << grid[ix][iy].size()
                << "\n";
        }
    }

    fout.close();

    float coverage = static_cast<float>(filled_count) /
        static_cast<float>(nx * ny);

    std::cout << "[BOTTOM] Saved bottom height grid: " << filename << std::endl;
    std::cout << "[BOTTOM] Grid size = " << grid_size << " m" << std::endl;
    std::cout << "[BOTTOM] Grid coverage = " << coverage * 100.0f << " %" << std::endl;

    return true;
}




bool saveCalibrationParams(
    const std::string& filename,
    bool long_axis_in_x,
    float x_size,
    float y_size,
    float yaw_rotation_deg,
    float cx_before,
    float cy_before,
    const BottomCalib& bottom
)
{
    std::ofstream fout(filename);

    if (!fout.is_open())
    {
        std::cerr << "[ERROR] Cannot write " << filename << std::endl;
        return false;
    }

    float x_min = -0.5f * x_size;
    float x_max = 0.5f * x_size;
    float y_min = -0.5f * y_size;
    float y_max = 0.5f * y_size;

    fout << std::fixed << std::setprecision(8);

    fout << "# Empty box calibration parameters\n";
    fout << "# Unit: meter\n";
    fout << "layout=" << (long_axis_in_x ? "A_X_long_Y_short" : "B_X_short_Y_long") << "\n";
    fout << "x_size=" << x_size << "\n";
    fout << "y_size=" << y_size << "\n";
    fout << "x_min=" << x_min << "\n";
    fout << "x_max=" << x_max << "\n";
    fout << "y_min=" << y_min << "\n";
    fout << "y_max=" << y_max << "\n";
    fout << "box_height_nominal=" << BOX_HEIGHT_NOMINAL << "\n";
    fout << "yaw_rotation_deg=" << yaw_rotation_deg << "\n";
    fout << "center_before_x=" << cx_before << "\n";
    fout << "center_before_y=" << cy_before << "\n";

    fout << "\n# Bottom calibration\n";
    fout << "bottom_ok=" << (bottom.ok ? 1 : 0) << "\n";
    fout << "bottom_plane_a=" << bottom.a << "\n";
    fout << "bottom_plane_b=" << bottom.b << "\n";
    fout << "bottom_plane_c=" << bottom.c << "\n";
    fout << "bottom_plane_d=" << bottom.d << "\n";
    fout << "bottom_z_mean=" << bottom.z_mean << "\n";
    fout << "bottom_z_median=" << bottom.z_median << "\n";
    fout << "bottom_z_min=" << bottom.z_min << "\n";
    fout << "bottom_z_max=" << bottom.z_max << "\n";
    fout << "bottom_z_std=" << bottom.z_std << "\n";
    fout << "bottom_inlier_count=" << bottom.inlier_count << "\n";
    fout << "bottom_grid_size=" << BOTTOM_GRID_SIZE << "\n";
    fout << "bottom_grid_file=bottom_height_grid.csv\n";
    fout << "calib_transform_file=" << CALIB_TRANSFORM_FILE << "\n";

    fout.close();

    std::cout << "[SUCCESS] Saved calibration file: " << filename << std::endl;

    return true;
}





bool saveTransformMatrix(const std::string& filename, const Eigen::Affine3f& tf)
{
    std::ofstream fout(filename);

    if (!fout.is_open())
    {
        std::cerr << "[ERROR] Cannot write " << filename << std::endl;
        return false;
    }

    Eigen::Matrix4f m = tf.matrix();
    fout << std::fixed << std::setprecision(10);

    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            fout << m(r, c);
            if (c < 3)
                fout << " ";
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "[SUCCESS] Saved transform matrix: " << filename << std::endl;
    return true;
}



void showCloud(const CloudT::Ptr& cloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Empty Box Bottom Calibration")
    );

    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> color(cloud, 255, 165, 0);
    viewer->addPointCloud<PointT>(cloud, color, "bottom_or_walls");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "bottom_or_walls"
    );

    viewer->addCoordinateSystem(0.2);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }
}




int main()
{
    std::cout << "[INFO] Start empty box calibration with bottom calibration..." << std::endl;

    CloudT::Ptr cloud_raw(new CloudT);

    if (pcl::io::loadPCDFile<PointT>(INPUT_FILE, *cloud_raw) == -1)
    {
        std::cerr << "[ERROR] Cannot find " << INPUT_FILE << std::endl;
        return -1;
    }

    CloudT::Ptr finite_cloud(new CloudT);
    finite_cloud->points.reserve(cloud_raw->points.size());
    for (const auto& p : cloud_raw->points)
    {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
            finite_cloud->points.push_back(p);
    }
    finite_cloud->width = static_cast<std::uint32_t>(finite_cloud->points.size());
    finite_cloud->height = 1;
    finite_cloud->is_dense = true;
    cloud_raw = finite_cloud;

    std::cout << "[INFO] Raw points: " << cloud_raw->size() << std::endl;


    CloudT::Ptr cloud_leveled(new CloudT);
    Eigen::Affine3f level_tf = Eigen::Affine3f::Identity();

    if (!levelCloud(cloud_raw, cloud_leveled, level_tf))
    {
        std::cerr << "[ERROR] Leveling failed." << std::endl;
        return -1;
    }

    savePCDAscii("calib_01_leveled.pcd", cloud_leveled);


    CloudT::Ptr cloud_rough = cropXYZ(
        cloud_leveled,
        ROUGH_X_MIN, ROUGH_X_MAX,
        ROUGH_Y_MIN, ROUGH_Y_MAX,
        ROUGH_Z_MIN, ROUGH_Z_MAX
    );

    savePCDAscii("calib_02_rough.pcd", cloud_rough);


    CloudT::Ptr wall_cloud = cropAxis(cloud_rough, "z", WALL_Z_MIN, WALL_Z_MAX);

    if (wall_cloud->size() < 100)
    {
        std::cerr << "[ERROR] Too few wall candidate points." << std::endl;
        return -1;
    }

    savePCDAscii("calib_03_wall_candidates.pcd", wall_cloud);


    std::vector<PlaneInfo> yaw_planes = extractWallPlanesForYaw(wall_cloud);

    if (yaw_planes.empty())
    {
        std::cerr << "[ERROR] No valid wall plane found for yaw alignment." << std::endl;
        return -1;
    }

    float rotation_rad = 0.0f;

    if (!computeYawRotation(yaw_planes, rotation_rad))
    {
        std::cerr << "[ERROR] Yaw calculation failed." << std::endl;
        return -1;
    }

    float yaw_rotation_deg = rotation_rad * 180.0f / static_cast<float>(M_PI);

    std::cout << "[INFO] Yaw rotation = " << yaw_rotation_deg << " deg" << std::endl;

    Eigen::Affine3f yaw_tf = Eigen::Affine3f::Identity();
    yaw_tf.rotate(Eigen::AngleAxisf(rotation_rad, Eigen::Vector3f::UnitZ()));

    CloudT::Ptr cloud_rotated(new CloudT);
    CloudT::Ptr wall_rotated(new CloudT);

    cloud_rotated = transformCloudEigen(cloud_rough, yaw_tf);
    wall_rotated = transformCloudEigen(wall_cloud, yaw_tf);

    savePCDAscii("calib_04_wall_rotated.pcd", wall_rotated);


    std::vector<AxisPlane> center_planes = extractAxisPlanesForCenter(wall_rotated);

    if (center_planes.size() < 4)
    {
        std::cerr << "[ERROR] Not enough wall planes for calibration." << std::endl;
        std::cerr << "[HINT] Check WALL_Z_MIN / WALL_Z_MAX / CENTER_MIN_INLIERS." << std::endl;
        return -1;
    }

    float cx = 0.0f;
    float cy = 0.0f;
    bool long_axis_in_x = true;
    PairResult x_pair;
    PairResult y_pair;

    if (!estimateCalibrationByWallPairs(center_planes, cx, cy, long_axis_in_x, x_pair, y_pair))
    {
        std::cerr << "[ERROR] Calibration failed." << std::endl;
        return -1;
    }

    float calib_x_size = x_pair.dist;
    float calib_y_size = y_pair.dist;
    float calib_x_half = 0.5f * calib_x_size;
    float calib_y_half = 0.5f * calib_y_size;


    Eigen::Affine3f center_tf = Eigen::Affine3f::Identity();
    center_tf.translation() << -cx, -cy, 0.0f;

    CloudT::Ptr cloud_centered(new CloudT);
    cloud_centered = transformCloudEigen(cloud_rotated, center_tf);


    Eigen::Affine3f calib_total_tf = center_tf * yaw_tf * level_tf;
    saveTransformMatrix(CALIB_TRANSFORM_FILE, calib_total_tf);

    savePCDAscii("empty_box_aligned_centered.pcd", cloud_centered);


    CloudT::Ptr clean_walls_check = extractCleanBoxWallsForCheck(
        cloud_centered,
        calib_x_half,
        calib_y_half
    );

    savePCDAscii("empty_box_walls_centered_check.pcd", clean_walls_check);

    CloudT::Ptr bottom_plane_cloud(new CloudT);
    BottomCalib bottom = fitBottomPlane(
        cloud_centered,
        calib_x_half,
        calib_y_half,
        bottom_plane_cloud
    );

    if (!bottom.ok)
    {
        std::cerr << "[ERROR] Bottom calibration failed." << std::endl;
        return -1;
    }

    savePCDAscii("bottom_plane.pcd", bottom_plane_cloud);

    saveBottomHeightGrid(
        "bottom_height_grid.csv",
        bottom_plane_cloud,
        calib_x_half,
        calib_y_half,
        BOTTOM_GRID_SIZE,
        bottom.z_median
    );


    saveCalibrationParams(
        "calib_params.txt",
        long_axis_in_x,
        calib_x_size,
        calib_y_size,
        yaw_rotation_deg,
        cx,
        cy,
        bottom
    );

    std::cout << "========================================" << std::endl;
    std::cout << "[SUCCESS] Empty box calibration finished." << std::endl;
    std::cout << "[RESULT] Layout: "
        << (long_axis_in_x ? "A: X long, Y short" : "B: X short, Y long")
        << std::endl;
    std::cout << "[RESULT] Calibrated X size = " << calib_x_size << " m" << std::endl;
    std::cout << "[RESULT] Calibrated Y size = " << calib_y_size << " m" << std::endl;
    std::cout << "[RESULT] Calibrated X range = ["
        << -calib_x_half << ", " << calib_x_half << "]" << std::endl;
    std::cout << "[RESULT] Calibrated Y range = ["
        << -calib_y_half << ", " << calib_y_half << "]" << std::endl;
    std::cout << "[RESULT] Bottom z_median = " << bottom.z_median << " m" << std::endl;
    std::cout << "[RESULT] Bottom z_std = " << bottom.z_std << " m" << std::endl;
    std::cout << "[INFO] Saved:" << std::endl;
    std::cout << "  - calib_params.txt" << std::endl;
    std::cout << "  - " << CALIB_TRANSFORM_FILE << std::endl;
    std::cout << "  - bottom_height_grid.csv" << std::endl;
    std::cout << "  - bottom_candidates.pcd" << std::endl;
    std::cout << "  - bottom_plane.pcd" << std::endl;
    std::cout << "  - calib_01_leveled.pcd" << std::endl;
    std::cout << "  - calib_02_rough.pcd" << std::endl;
    std::cout << "  - calib_03_wall_candidates.pcd" << std::endl;
    std::cout << "  - calib_04_wall_rotated.pcd" << std::endl;
    std::cout << "  - empty_box_aligned_centered.pcd" << std::endl;
    std::cout << "  - empty_box_walls_centered_check.pcd" << std::endl;



    return 0;
}
