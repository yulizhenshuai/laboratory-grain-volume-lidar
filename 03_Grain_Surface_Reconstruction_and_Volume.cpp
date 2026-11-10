#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include <map>
#include <queue>
#include <cstdint>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
const std::string INPUT_FILE = "grain_bin_raw.pcd";
const std::string CALIB_FILE = "calib_params.txt";
const std::string BOTTOM_GRID_FILE = "bottom_height_grid.csv";
const std::string CALIB_TRANSFORM_FILE = "calib_transform.txt";
const std::string OUT_DEBUG_WALL_ROTATED = "debug_wall_rotated.pcd";
const std::string OUT_WALLS_CHECK = "box_walls_centered_check.pcd";
const std::string OUT_ALIGNED = "cloud_aligned_xy_centered.pcd";
const std::string OUT_EXPANDED_ROI = "cloud_fine_expanded_roi.pcd";
const std::string OUT_GRAIN_SURFACE = "final_grain_surface.pcd";
const std::string OUT_GRAIN_SURFACE_RAW = "final_grain_surface_raw.pcd";
const std::string OUT_GRAIN_SURFACE_RIM_FILTERED = "final_grain_surface_rim_filtered.pcd";
const std::string OUT_GRAIN_SURFACE_COMPONENT_FILTERED = "final_grain_surface_component_filtered.pcd";
const std::string OUT_RECON_PCD = "reconstructed_grain_surface.pcd";
const std::string OUT_HEIGHT_GRID_CSV = "grain_height_grid.csv";
const std::string OUT_SURFACE_PLY = "grain_surface_mesh.ply";
const std::string OUT_SOLID_PLY = "grain_volume_solid_mesh.ply";
const std::string OUT_VOLUME_RESULT = "volume_result.txt";
const float ROUGH_X_MIN = -1.0f;
const float ROUGH_X_MAX = 1.0f;
const float ROUGH_Y_MIN = -1.0f;
const float ROUGH_Y_MAX = 1.0f;
const float ROUGH_Z_MIN = -1.0f;
const float ROUGH_Z_MAX = 1.2f;
const float WALL_Z_MIN = 0.10f;
const float WALL_Z_MAX = 0.42f;
const float BOX_LENGTH = 1.00f;
const float BOX_WIDTH = 0.60f;
const float BOX_HEIGHT = 0.40f;
const int   WALL_MAX_PLANES = 6;
const int   WALL_MIN_INLIERS = 80;
const float WALL_DISTANCE_THRESHOLD = 0.018f;
const float WALL_EPS_ANGLE_DEG = 8.0f;
const float MAX_WALL_NORMAL_Z = 0.30f;
const int   CENTER_MAX_PLANES = 8;
const int   CENTER_MIN_INLIERS = 60;
const float CENTER_DIST_TOL = 0.08f;
const float YAW_MANUAL_OFFSET_DEG = 0.0f;
const float FINAL_Z_MIN = -0.02f;
const float FINAL_Z_MAX = 0.45f;
const float WALL_CHECK_BAND = 0.04f;
const float WALL_CHECK_Z_MIN = 0.08f;
const float WALL_CHECK_Z_MAX = 0.45f;
const float ROI_EXPAND_MARGIN = 0.035f;
const float WALL_REMOVE_BAND = 0.06f;
const float GRAIN_NORMAL_Z_MIN = 0.30f;
const int NORMAL_K = 30;
const int   SOR_MEAN_K = 20;
const float SOR_STDDEV = 1.5f;
const float RIM_REMOVE_BAND = 0.08f;
const float RIM_Z_MIN = 0.32f;
const float FINAL_XY_MARGIN = 0.005f;
const float COMPONENT_GRID_SIZE = 0.01f;
const int   COMPONENT_MIN_POINTS = 30;
const float RECON_GRID_SIZE = 0.01f;   // 1 cm
const int   MIN_POINTS_PER_GRID = 1;
const float GRAIN_HEIGHT_PERCENTILE = 0.60f;
const int   FILL_MAX_ITER = 8;
const bool  FILL_REMAINING_WITH_GLOBAL_MEDIAN = true;
const float MAX_REASONABLE_GRAIN_HEIGHT = 1.0f;
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

struct SinglePlaneResult
{
    bool ok = false;
    int index = -1;
    float pos = 0.0f;
    int inliers = 0;
};

struct CalibParams
{
    bool ok = false;

    float x_min = -0.3f;
    float x_max = 0.3f;
    float y_min = -0.5f;
    float y_max = 0.5f;

    float bottom_a = 0.0f;
    float bottom_b = 0.0f;
    float bottom_c = 1.0f;
    float bottom_d = 0.0f;

    float bottom_z_median = 0.0f;
    float bottom_grid_size = RECON_GRID_SIZE;
};

struct BottomGrid
{
    bool ok = false;
    int nx = 0;
    int ny = 0;
    float grid_size = RECON_GRID_SIZE;
    std::vector<float> z_values;
    std::vector<int> filled;
};

struct GridCell
{
    bool valid = false;
    bool interpolated = false;
    bool global_filled = false;

    float x = 0.0f;
    float y = 0.0f;

    float z_grain = 0.0f;
    float z_bottom = 0.0f;
    float height = 0.0f;

    int point_count = 0;
};

struct ReconResult
{
    int nx = 0;
    int ny = 0;

    int raw_filled = 0;
    int interp_filled = 0;
    int global_filled = 0;
    int final_filled = 0;

    float grid_size = RECON_GRID_SIZE;
    float raw_coverage = 0.0f;
    float final_coverage = 0.0f;
    float volume_m3 = 0.0f;

    std::vector<GridCell> cells;
    CloudT::Ptr reconstructed_cloud;
};

struct Vertex
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Face
{
    int a = 0;
    int b = 0;
    int c = 0;
};


float deg2rad(float deg)
{
    return deg * static_cast<float>(M_PI) / 180.0f;
}

CloudT::Ptr cropAxis(const CloudT::Ptr& cloud, const std::string& axis, float min_v, float max_v)
{
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    for (const auto& p : cloud->points)
    {
        float value = (axis == "x") ? p.x : ((axis == "y") ? p.y : p.z);
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
         << "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
         << "COUNT 1 1 1\nWIDTH " << cloud->points.size() << "\nHEIGHT 1\n"
         << "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << cloud->points.size() << "\n"
         << "DATA ascii\n" << std::setprecision(9);
    for (const auto& p : cloud->points)
        fout << p.x << ' ' << p.y << ' ' << p.z << '\n';
    return static_cast<bool>(fout);
}

float percentileValue(std::vector<float> values, float q)
{
    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());

    if (q <= 0.0f)
        return values.front();

    if (q >= 1.0f)
        return values.back();

    float pos = q * static_cast<float>(values.size() - 1);
    int low = static_cast<int>(std::floor(pos));
    int high = static_cast<int>(std::ceil(pos));

    if (low == high)
        return values[low];

    float alpha = pos - static_cast<float>(low);
    return values[low] * (1.0f - alpha) + values[high] * alpha;
}

float medianValue(std::vector<float> values)
{
    return percentileValue(values, 0.50f);
}



bool loadCalibParams(const std::string& filename, CalibParams& calib)
{
    std::ifstream fin(filename);

    if (!fin.is_open())
    {
        std::cerr << "[ERROR] Cannot open calibration file: " << filename << std::endl;
        return false;
    }

    std::map<std::string, std::string> kv;
    std::string line;

    while (std::getline(fin, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        size_t pos = line.find('=');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        kv[key] = value;
    }

    auto getFloat = [&](const std::string& key, float default_value)
        {
            if (kv.find(key) == kv.end())
                return default_value;

            try
            {
                return std::stof(kv[key]);
            }
            catch (...)
            {
                return default_value;
            }
        };

    calib.x_min = getFloat("x_min", calib.x_min);
    calib.x_max = getFloat("x_max", calib.x_max);
    calib.y_min = getFloat("y_min", calib.y_min);
    calib.y_max = getFloat("y_max", calib.y_max);

    calib.bottom_a = getFloat("bottom_plane_a", 0.0f);
    calib.bottom_b = getFloat("bottom_plane_b", 0.0f);
    calib.bottom_c = getFloat("bottom_plane_c", 1.0f);
    calib.bottom_d = getFloat("bottom_plane_d", 0.0f);

    calib.bottom_z_median = getFloat("bottom_z_median", 0.0f);
    calib.bottom_grid_size = getFloat("bottom_grid_size", RECON_GRID_SIZE);

    if (calib.x_max <= calib.x_min || calib.y_max <= calib.y_min)
    {
        std::cerr << "[ERROR] Invalid calibration boundary." << std::endl;
        return false;
    }

    calib.ok = true;

    std::cout << "[CALIB] Loaded calibration:" << std::endl;
    std::cout << "        x: [" << calib.x_min << ", " << calib.x_max << "]" << std::endl;
    std::cout << "        y: [" << calib.y_min << ", " << calib.y_max << "]" << std::endl;
    std::cout << "        bottom plane: "
        << calib.bottom_a << "x + "
        << calib.bottom_b << "y + "
        << calib.bottom_c << "z + "
        << calib.bottom_d << " = 0" << std::endl;
    std::cout << "        bottom_z_median = " << calib.bottom_z_median << std::endl;

    return true;
}

float bottomZFromPlane(const CalibParams& calib, float x, float y)
{
    if (std::abs(calib.bottom_c) < 1e-6f)
        return calib.bottom_z_median;

    return -(calib.bottom_a * x + calib.bottom_b * y + calib.bottom_d) / calib.bottom_c;
}


bool loadBottomHeightGrid(const std::string& filename, const CalibParams& calib, BottomGrid& grid)
{
    std::ifstream fin(filename);

    if (!fin.is_open())
    {
        std::cout << "[WARNING] Cannot open " << filename
            << ". Use bottom plane from calib_params.txt instead." << std::endl;
        return false;
    }

    std::string header;
    std::getline(fin, header);

    struct Row
    {
        int ix = 0;
        int iy = 0;
        float z = 0.0f;
        int filled = 0;
    };

    std::vector<Row> rows;
    std::string line;
    int max_ix = -1;
    int max_iy = -1;

    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> cols;

        while (std::getline(ss, item, ','))
            cols.push_back(item);

        if (cols.size() < 7)
            continue;

        Row r;
        r.ix = std::stoi(cols[0]);
        r.iy = std::stoi(cols[1]);
        r.z = std::stof(cols[4]);
        r.filled = std::stoi(cols[5]);

        rows.push_back(r);
        max_ix = std::max(max_ix, r.ix);
        max_iy = std::max(max_iy, r.iy);
    }

    if (rows.empty() || max_ix < 0 || max_iy < 0)
    {
        std::cout << "[WARNING] Empty bottom grid. Use bottom plane instead." << std::endl;
        return false;
    }

    grid.nx = max_ix + 1;
    grid.ny = max_iy + 1;
    grid.grid_size = calib.bottom_grid_size > 0.0f ? calib.bottom_grid_size : RECON_GRID_SIZE;
    grid.z_values.assign(grid.nx * grid.ny, calib.bottom_z_median);
    grid.filled.assign(grid.nx * grid.ny, 0);

    int filled_count = 0;

    for (const Row& r : rows)
    {
        int id = r.iy * grid.nx + r.ix;
        if (id < 0 || id >= static_cast<int>(grid.z_values.size()))
            continue;

        grid.z_values[id] = r.z;
        grid.filled[id] = r.filled;
        if (r.filled)
            filled_count++;
    }

    grid.ok = true;

    std::cout << "[BOTTOM_GRID] Loaded " << filename << std::endl;
    std::cout << "[BOTTOM_GRID] nx = " << grid.nx
        << ", ny = " << grid.ny
        << ", filled = " << filled_count << std::endl;

    return true;
}

float getBottomZ(const CalibParams& calib, const BottomGrid& bottom_grid, float x, float y)
{
    if (bottom_grid.ok)
    {
        int ix = static_cast<int>((x - calib.x_min) / bottom_grid.grid_size);
        int iy = static_cast<int>((y - calib.y_min) / bottom_grid.grid_size);

        if (ix >= 0 && ix < bottom_grid.nx && iy >= 0 && iy < bottom_grid.ny)
        {
            int id = iy * bottom_grid.nx + ix;
            if (id >= 0 && id < static_cast<int>(bottom_grid.z_values.size()) && bottom_grid.filled[id])
                return bottom_grid.z_values[id];
        }
    }

    return bottomZFromPlane(calib, x, y);
}




bool loadTransformMatrix(const std::string& filename, Eigen::Affine3f& tf)
{
    std::ifstream fin(filename);

    if (!fin.is_open())
    {
        std::cerr << "[ERROR] Cannot open transform file: " << filename << std::endl;
        return false;
    }

    Eigen::Matrix4f m;
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            if (!(fin >> m(r, c)))
            {
                std::cerr << "[ERROR] Invalid transform matrix file: " << filename << std::endl;
                return false;
            }
        }
    }

    tf = Eigen::Affine3f(m);

    std::cout << "[CALIB] Loaded transform matrix: " << filename << std::endl;
    return true;
}



bool levelCloud(const CloudT::Ptr& cloud_raw, CloudT::Ptr& cloud_leveled)
{
    pcl::SACSegmentation<PointT> seg;
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(2000);
    seg.setDistanceThreshold(0.02);
    seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setEpsAngle(deg2rad(45.0f));

    seg.setInputCloud(cloud_raw);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.empty())
        return false;

    Eigen::Vector3f n(coeff->values[0], coeff->values[1], coeff->values[2]);
    n.normalize();

    if (n.z() < 0)
        n = -n;

    Eigen::Quaternionf q;
    q.setFromTwoVectors(n, Eigen::Vector3f(0, 0, 1));

    Eigen::Affine3f rot = Eigen::Affine3f::Identity();
    rot.rotate(q);

    CloudT::Ptr cloud_rot(new CloudT);
    pcl::transformPointCloud(*cloud_raw, *cloud_rot, rot);

    float avg_z = 0.0f;
    for (int id : inliers->indices)
        avg_z += cloud_rot->points[id].z;

    avg_z /= static_cast<float>(inliers->indices.size());

    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.rotate(q);
    tf.translation() << 0, 0, -avg_z;

    pcl::transformPointCloud(*cloud_raw, *cloud_leveled, tf);

    return true;
}



std::vector<PlaneInfo> extractWallPlanesForYaw(const CloudT::Ptr& wall_cloud)
{
    std::vector<PlaneInfo> planes;

    CloudT::Ptr remain(new CloudT);
    *remain = *wall_cloud;

    pcl::SACSegmentation<PointT> seg;
    pcl::ExtractIndices<PointT> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(3000);
    seg.setDistanceThreshold(WALL_DISTANCE_THRESHOLD);
    seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setEpsAngle(deg2rad(WALL_EPS_ANGLE_DEG));

    for (int i = 0; i < WALL_MAX_PLANES; ++i)
    {
        pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setInputCloud(remain);
        seg.segment(*inliers, *coeff);

        if (inliers->indices.size() < WALL_MIN_INLIERS)
            break;

        Eigen::Vector3f n(coeff->values[0], coeff->values[1], coeff->values[2]);

        if (n.norm() < 1e-6f)
            break;

        n.normalize();

        if (std::abs(n.z()) <= MAX_WALL_NORMAL_Z)
            planes.push_back({ n, static_cast<int>(inliers->indices.size()) });

        CloudT::Ptr next(new CloudT);
        extract.setInputCloud(remain);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*next);
        remain = next;

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

    pcl::SACSegmentation<PointT> seg;
    pcl::ExtractIndices<PointT> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(3000);
    seg.setDistanceThreshold(WALL_DISTANCE_THRESHOLD);
    seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setEpsAngle(deg2rad(WALL_EPS_ANGLE_DEG));

    for (int k = 0; k < CENTER_MAX_PLANES; ++k)
    {
        pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setInputCloud(remain);
        seg.segment(*inliers, *coeff);

        if (inliers->indices.size() < CENTER_MIN_INLIERS)
            break;

        Eigen::Vector3f n(coeff->values[0], coeff->values[1], coeff->values[2]);

        if (n.norm() < 1e-6f)
            break;

        n.normalize();

        CloudT::Ptr one_plane(new CloudT);
        extract.setInputCloud(remain);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*one_plane);

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

            planes.push_back({ axis, pos, static_cast<int>(inliers->indices.size()), one_plane });

            std::cout << "[CENTER] plane " << k + 1
                << " axis = " << axis
                << ", pos = " << pos
                << ", inliers = " << inliers->indices.size()
                << std::endl;
        }

        CloudT::Ptr next(new CloudT);
        extract.setInputCloud(remain);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*next);
        remain = next;

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
            else if (std::abs(error - best.error) < 1e-4f && total_inliers > best.total_inliers)
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
            << " rejected. distance error = " << best.error << " m" << std::endl;
        best.ok = false;
    }

    return best;
}

SinglePlaneResult selectBestSinglePlane(
    const std::vector<AxisPlane>& planes,
    char axis,
    float expected_half_size
)
{
    SinglePlaneResult best;

    for (size_t i = 0; i < planes.size(); ++i)
    {
        if (planes[i].axis != axis)
            continue;

        float abs_pos = std::abs(planes[i].pos);
        float error = std::abs(abs_pos - expected_half_size);

        if (error > CENTER_DIST_TOL)
            continue;

        if (!best.ok || planes[i].inliers > best.inliers)
        {
            best.ok = true;
            best.index = static_cast<int>(i);
            best.pos = planes[i].pos;
            best.inliers = planes[i].inliers;
        }
    }

    return best;
}

bool estimateBoxCenterByWallPairs(
    const std::vector<AxisPlane>& planes,
    float& cx,
    float& cy,
    bool& long_axis_in_x,
    PairResult& final_x_pair,
    PairResult& final_y_pair
)
{
    PairResult xA = selectBestOppositePair(planes, 'x', BOX_LENGTH);
    PairResult yA = selectBestOppositePair(planes, 'y', BOX_WIDTH);

    PairResult xB = selectBestOppositePair(planes, 'x', BOX_WIDTH);
    PairResult yB = selectBestOppositePair(planes, 'y', BOX_LENGTH);

    bool A_ok = xA.ok && yA.ok;
    bool B_ok = xB.ok && yB.ok;

    if (A_ok || B_ok)
    {
        float scoreA = A_ok ? xA.error + yA.error : std::numeric_limits<float>::max();
        float scoreB = B_ok ? xB.error + yB.error : std::numeric_limits<float>::max();

        if (scoreA <= scoreB)
        {
            long_axis_in_x = true;
            cx = xA.center;
            cy = yA.center;
            final_x_pair = xA;
            final_y_pair = yA;
            std::cout << "[CENTER] Layout A selected: X = 1.0 m, Y = 0.6 m" << std::endl;
        }
        else
        {
            long_axis_in_x = false;
            cx = xB.center;
            cy = yB.center;
            final_x_pair = xB;
            final_y_pair = yB;
            std::cout << "[CENTER] Layout B selected: X = 0.6 m, Y = 1.0 m" << std::endl;
        }

        std::cout << "[CENTER] X pair distance = " << final_x_pair.dist
            << ", error = " << final_x_pair.error << std::endl;
        std::cout << "[CENTER] Y pair distance = " << final_y_pair.dist
            << ", error = " << final_y_pair.error << std::endl;
        std::cout << "[CENTER] center before correction: cx = " << cx
            << ", cy = " << cy << std::endl;

        return true;
    }

    if (xB.ok)
    {
        long_axis_in_x = false;
        cx = xB.center;
        final_x_pair = xB;

        SinglePlaneResult y_single = selectBestSinglePlane(planes, 'y', BOX_LENGTH * 0.5f);
        if (y_single.ok)
        {
            cy = y_single.pos - (y_single.pos >= 0.0f ? BOX_LENGTH * 0.5f : -BOX_LENGTH * 0.5f);

            final_y_pair.ok = false;
            final_y_pair.i = y_single.index;
            final_y_pair.j = -1;
            final_y_pair.p1 = y_single.pos;
            final_y_pair.p2 = y_single.pos;
            final_y_pair.center = cy;
            final_y_pair.total_inliers = y_single.inliers;

            std::cout << "[CENTER] Fallback selected: X pair = 0.6 m, Y single wall -> infer center" << std::endl;
            std::cout << "[CENTER] center before correction: cx = " << cx
                << ", cy = " << cy << std::endl;
            return true;
        }
    }

    if (yA.ok)
    {
        long_axis_in_x = true;
        cy = yA.center;
        final_y_pair = yA;

        SinglePlaneResult x_single = selectBestSinglePlane(planes, 'x', BOX_LENGTH * 0.5f);
        if (x_single.ok)
        {
            cx = x_single.pos - (x_single.pos >= 0.0f ? BOX_LENGTH * 0.5f : -BOX_LENGTH * 0.5f);

            final_x_pair.ok = false;
            final_x_pair.i = x_single.index;
            final_x_pair.j = -1;
            final_x_pair.p1 = x_single.pos;
            final_x_pair.p2 = x_single.pos;
            final_x_pair.center = cx;
            final_x_pair.total_inliers = x_single.inliers;

            std::cout << "[CENTER] Fallback selected: Y pair = 0.6 m, X single wall -> infer center" << std::endl;
            std::cout << "[CENTER] center before correction: cx = " << cx
                << ", cy = " << cy << std::endl;
            return true;
        }
    }

    std::cerr << "[ERROR] No valid complete or fallback wall configuration found." << std::endl;
    return false;
}



CloudT::Ptr extractCleanBoxWallsForCheck(
    const CloudT::Ptr& cloud,
    bool long_axis_in_x
)
{
    CloudT::Ptr walls(new CloudT);

    float half_x = long_axis_in_x ? BOX_LENGTH * 0.5f : BOX_WIDTH * 0.5f;
    float half_y = long_axis_in_x ? BOX_WIDTH * 0.5f : BOX_LENGTH * 0.5f;

    for (const auto& p : cloud->points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (p.z < WALL_CHECK_Z_MIN || p.z > WALL_CHECK_Z_MAX)
            continue;

        bool near_x_wall =
            std::abs(std::abs(p.x) - half_x) <= WALL_CHECK_BAND &&
            std::abs(p.y) <= half_y + WALL_CHECK_BAND;

        bool near_y_wall =
            std::abs(std::abs(p.y) - half_y) <= WALL_CHECK_BAND &&
            std::abs(p.x) <= half_x + WALL_CHECK_BAND;

        if (near_x_wall || near_y_wall)
            walls->points.push_back(p);
    }

    walls->width = static_cast<uint32_t>(walls->points.size());
    walls->height = 1;
    walls->is_dense = false;
    return walls;
}


CloudT::Ptr extractCleanBoxWallsForCheckByCalib(
    const CloudT::Ptr& cloud,
    const CalibParams& calib
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
            (std::abs(p.x - calib.x_min) <= WALL_CHECK_BAND ||
             std::abs(p.x - calib.x_max) <= WALL_CHECK_BAND) &&
            p.y >= calib.y_min - WALL_CHECK_BAND &&
            p.y <= calib.y_max + WALL_CHECK_BAND;

        bool near_y_wall =
            (std::abs(p.y - calib.y_min) <= WALL_CHECK_BAND ||
             std::abs(p.y - calib.y_max) <= WALL_CHECK_BAND) &&
            p.x >= calib.x_min - WALL_CHECK_BAND &&
            p.x <= calib.x_max + WALL_CHECK_BAND;

        if (near_x_wall || near_y_wall)
            walls->points.push_back(p);
    }

    walls->width = static_cast<uint32_t>(walls->points.size());
    walls->height = 1;
    walls->is_dense = false;
    return walls;
}



CloudT::Ptr makeExpandedROIByCalib(
    const CloudT::Ptr& cloud_aligned,
    const CalibParams& calib
)
{
    return cropXYZ(
        cloud_aligned,
        calib.x_min - ROI_EXPAND_MARGIN,
        calib.x_max + ROI_EXPAND_MARGIN,
        calib.y_min - ROI_EXPAND_MARGIN,
        calib.y_max + ROI_EXPAND_MARGIN,
        FINAL_Z_MIN,
        FINAL_Z_MAX
    );
}

CloudT::Ptr removeOutliers(const CloudT::Ptr& cloud)
{
    if (cloud->size() < 50)
        return cloud;

    CloudT::Ptr filtered(new CloudT);


    filtered->points.reserve(cloud->size());
    for (const auto& p : cloud->points)
    {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
            filtered->points.push_back(p);
    }
    filtered->width = static_cast<std::uint32_t>(filtered->points.size());
    filtered->height = 1;
    filtered->is_dense = true;

    return filtered;
}

CloudT::Ptr segmentGrainSurface(
    const CloudT::Ptr& cloud_candidate,
    const CalibParams& calib
)
{
    CloudT::Ptr grain(new CloudT);

    if (cloud_candidate->empty())
        return grain;

    float half_x = 0.5f * (calib.x_max - calib.x_min);
    float half_y = 0.5f * (calib.y_max - calib.y_min);

    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

    ne.setInputCloud(cloud_candidate);
    ne.setSearchMethod(tree);
    ne.setKSearch(NORMAL_K);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    if (normals->size() != cloud_candidate->size())
    {
        std::cerr << "[WARNING] Normal size mismatch. Return candidate directly." << std::endl;
        return cloud_candidate;
    }

    int removed_wall_like = 0;

    for (size_t i = 0; i < cloud_candidate->size(); ++i)
    {
        const PointT& p = cloud_candidate->points[i];
        const pcl::Normal& n = normals->points[i];

        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z))
            continue;

        float nz = std::abs(n.normal_z);

        bool near_x_wall =
            std::abs(std::abs(p.x) - half_x) <= WALL_REMOVE_BAND &&
            std::abs(p.y) <= half_y + ROI_EXPAND_MARGIN;

        bool near_y_wall =
            std::abs(std::abs(p.y) - half_y) <= WALL_REMOVE_BAND &&
            std::abs(p.x) <= half_x + ROI_EXPAND_MARGIN;

        bool near_wall = near_x_wall || near_y_wall;
        bool wall_like = near_wall && (nz < GRAIN_NORMAL_Z_MIN);

        if (wall_like)
        {
            removed_wall_like++;
            continue;
        }

        grain->points.push_back(p);
    }

    grain->width = static_cast<uint32_t>(grain->points.size());
    grain->height = 1;
    grain->is_dense = false;

    CloudT::Ptr grain_clean = removeOutliers(grain);

    std::cout << "[SEG] Candidate points: " << cloud_candidate->size() << std::endl;
    std::cout << "[SEG] Removed wall-like boundary points: " << removed_wall_like << std::endl;
    std::cout << "[SEG] Grain after normal filtering: " << grain->size() << std::endl;
    std::cout << "[SEG] Grain after outlier removal: " << grain_clean->size() << std::endl;

    return grain_clean;
}




bool isInsideCalibXY(
    const PointT& p,
    const CalibParams& calib,
    float margin = 0.0f
)
{
    return p.x >= calib.x_min - margin &&
           p.x <= calib.x_max + margin &&
           p.y >= calib.y_min - margin &&
           p.y <= calib.y_max + margin;
}

float distanceToCalibBoundaryXY(
    const PointT& p,
    const CalibParams& calib
)
{
    float dx_min = std::abs(p.x - calib.x_min);
    float dx_max = std::abs(p.x - calib.x_max);
    float dy_min = std::abs(p.y - calib.y_min);
    float dy_max = std::abs(p.y - calib.y_max);

    return std::min(std::min(dx_min, dx_max), std::min(dy_min, dy_max));
}

CloudT::Ptr removeRimResidualPoints(
    const CloudT::Ptr& grain_cloud,
    const CalibParams& calib
)
{
    CloudT::Ptr filtered(new CloudT);

    if (!grain_cloud || grain_cloud->empty())
        return filtered;

    int removed_outside = 0;
    int removed_rim = 0;

    for (const auto& p : grain_cloud->points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (!isInsideCalibXY(p, calib, FINAL_XY_MARGIN))
        {
            removed_outside++;
            continue;
        }

        float d_wall = distanceToCalibBoundaryXY(p, calib);
        bool near_boundary = d_wall <= RIM_REMOVE_BAND;
        bool high_rim_like = p.z >= RIM_Z_MIN;
        if (near_boundary && high_rim_like)
        {
            removed_rim++;
            continue;
        }

        filtered->points.push_back(p);
    }

    filtered->width = static_cast<uint32_t>(filtered->points.size());
    filtered->height = 1;
    filtered->is_dense = false;

    std::cout << "[POST] Rim filter input points: " << grain_cloud->size() << std::endl;
    std::cout << "[POST] Removed outside calib boundary: " << removed_outside << std::endl;
    std::cout << "[POST] Removed rim-like high boundary points: " << removed_rim << std::endl;
    std::cout << "[POST] After rim filter: " << filtered->size() << std::endl;

    return filtered;
}

CloudT::Ptr keepLargestXYComponent(
    const CloudT::Ptr& cloud,
    const CalibParams& calib
)
{
    CloudT::Ptr output(new CloudT);

    if (!cloud || cloud->empty())
        return output;

    const int nx = static_cast<int>(std::ceil((calib.x_max - calib.x_min) / COMPONENT_GRID_SIZE));
    const int ny = static_cast<int>(std::ceil((calib.y_max - calib.y_min) / COMPONENT_GRID_SIZE));

    if (nx <= 0 || ny <= 0)
    {
        std::cerr << "[WARNING] Invalid component grid size. Skip component filtering." << std::endl;
        *output = *cloud;
        return output;
    }

    std::vector<std::vector<int>> cell_point_indices(nx * ny);
    std::vector<int> point_cell_id(cloud->size(), -1);

    int valid_input_points = 0;

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const PointT& p = cloud->points[i];

        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (!isInsideCalibXY(p, calib, FINAL_XY_MARGIN))
            continue;

        int ix = static_cast<int>((p.x - calib.x_min) / COMPONENT_GRID_SIZE);
        int iy = static_cast<int>((p.y - calib.y_min) / COMPONENT_GRID_SIZE);

        if (ix < 0)
            ix = 0;
        if (ix >= nx)
            ix = nx - 1;
        if (iy < 0)
            iy = 0;
        if (iy >= ny)
            iy = ny - 1;

        int cid = iy * nx + ix;
        cell_point_indices[cid].push_back(static_cast<int>(i));
        point_cell_id[i] = cid;
        valid_input_points++;
    }

    std::vector<int> cell_component(nx * ny, -1);
    std::vector<int> component_cell_count;
    std::vector<int> component_point_count;

    int component_id = 0;
    const int dx8[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    const int dy8[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };

    for (int iy = 0; iy < ny; ++iy)
    {
        for (int ix = 0; ix < nx; ++ix)
        {
            int start = iy * nx + ix;

            if (cell_point_indices[start].empty() || cell_component[start] >= 0)
                continue;

            std::queue<int> q;
            q.push(start);
            cell_component[start] = component_id;

            int cells = 0;
            int points = 0;

            while (!q.empty())
            {
                int cid = q.front();
                q.pop();

                int cx = cid % nx;
                int cy = cid / nx;

                cells++;
                points += static_cast<int>(cell_point_indices[cid].size());

                for (int k = 0; k < 8; ++k)
                {
                    int xx = cx + dx8[k];
                    int yy = cy + dy8[k];

                    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny)
                        continue;

                    int nid = yy * nx + xx;

                    if (cell_point_indices[nid].empty() || cell_component[nid] >= 0)
                        continue;

                    cell_component[nid] = component_id;
                    q.push(nid);
                }
            }

            component_cell_count.push_back(cells);
            component_point_count.push_back(points);
            component_id++;
        }
    }

    if (component_id == 0)
    {
        std::cerr << "[WARNING] No XY component found. Skip component filtering." << std::endl;
        *output = *cloud;
        return output;
    }

    int best_component = 0;
    int best_points = component_point_count[0];

    for (int i = 1; i < component_id; ++i)
    {
        if (component_point_count[i] > best_points)
        {
            best_points = component_point_count[i];
            best_component = i;
        }
    }

    if (best_points < COMPONENT_MIN_POINTS)
    {
        std::cerr << "[WARNING] Largest component too small. Skip component filtering." << std::endl;
        *output = *cloud;
        return output;
    }

    int removed_small_components = 0;
    int kept_points = 0;

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        int cid = point_cell_id[i];

        if (cid < 0)
            continue;

        int comp = cell_component[cid];

        if (comp == best_component)
        {
            output->points.push_back(cloud->points[i]);
            kept_points++;
        }
        else
        {
            removed_small_components++;
        }
    }

    output->width = static_cast<uint32_t>(output->points.size());
    output->height = 1;
    output->is_dense = false;

    std::cout << "[POST] XY components found: " << component_id << std::endl;
    std::cout << "[POST] Largest component id: " << best_component
              << ", cells = " << component_cell_count[best_component]
              << ", points = " << component_point_count[best_component] << std::endl;
    std::cout << "[POST] Valid input points for component filtering: " << valid_input_points << std::endl;
    std::cout << "[POST] Removed points in small components: " << removed_small_components << std::endl;
    std::cout << "[POST] Final grain points after component filter: " << kept_points << std::endl;

    return output;
}



void fillMissingByNeighbors(std::vector<GridCell>& cells, int nx, int ny)
{
    for (int iter = 0; iter < FILL_MAX_ITER; ++iter)
    {
        std::vector<GridCell> old = cells;
        int filled_this_iter = 0;

        for (int ix = 0; ix < nx; ++ix)
        {
            for (int iy = 0; iy < ny; ++iy)
            {
                int id = iy * nx + ix;

                if (old[id].valid)
                    continue;

                float sum_z = 0.0f;
                int count = 0;

                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        if (dx == 0 && dy == 0)
                            continue;

                        int xx = ix + dx;
                        int yy = iy + dy;

                        if (xx < 0 || xx >= nx || yy < 0 || yy >= ny)
                            continue;

                        int nid = yy * nx + xx;

                        if (old[nid].valid)
                        {
                            sum_z += old[nid].z_grain;
                            count++;
                        }
                    }
                }

                if (count >= 3)
                {
                    cells[id].valid = true;
                    cells[id].interpolated = true;
                    cells[id].z_grain = sum_z / static_cast<float>(count);
                    filled_this_iter++;
                }
            }
        }

        if (filled_this_iter == 0)
            break;
    }
}

int fillRemainingByGlobalMedian(std::vector<GridCell>& cells)
{
    std::vector<float> valid_heights;
    valid_heights.reserve(cells.size());

    for (const auto& c : cells)
    {
        if (c.valid)
            valid_heights.push_back(c.z_grain);
    }

    if (valid_heights.empty())
        return 0;

    float z_med = medianValue(valid_heights);
    int filled = 0;

    for (auto& c : cells)
    {
        if (!c.valid)
        {
            c.valid = true;
            c.interpolated = true;
            c.global_filled = true;
            c.z_grain = z_med;
            filled++;
        }
    }

    return filled;
}

ReconResult reconstructAndComputeVolume(
    const CloudT::Ptr& grain_cloud,
    const CalibParams& calib,
    const BottomGrid& bottom_grid
)
{
    ReconResult result;
    result.grid_size = RECON_GRID_SIZE;
    result.reconstructed_cloud.reset(new CloudT);

    int nx = static_cast<int>(std::ceil((calib.x_max - calib.x_min) / RECON_GRID_SIZE));
    int ny = static_cast<int>(std::ceil((calib.y_max - calib.y_min) / RECON_GRID_SIZE));

    result.nx = nx;
    result.ny = ny;

    std::vector<std::vector<float>> z_values(nx * ny);
    std::vector<GridCell> cells(nx * ny);

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            int id = iy * nx + ix;
            float x = calib.x_min + (ix + 0.5f) * RECON_GRID_SIZE;
            float y = calib.y_min + (iy + 0.5f) * RECON_GRID_SIZE;

            cells[id].x = x;
            cells[id].y = y;
            cells[id].z_bottom = getBottomZ(calib, bottom_grid, x, y);
        }
    }

    for (const auto& p : grain_cloud->points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        if (p.x < calib.x_min || p.x > calib.x_max ||
            p.y < calib.y_min || p.y > calib.y_max)
            continue;

        int ix = static_cast<int>((p.x - calib.x_min) / RECON_GRID_SIZE);
        int iy = static_cast<int>((p.y - calib.y_min) / RECON_GRID_SIZE);

        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny)
            continue;

        int id = iy * nx + ix;
        z_values[id].push_back(p.z);
    }

    int raw_filled = 0;

    for (int id = 0; id < nx * ny; ++id)
    {
        if (static_cast<int>(z_values[id].size()) >= MIN_POINTS_PER_GRID)
        {
            cells[id].valid = true;
            cells[id].z_grain = percentileValue(z_values[id], GRAIN_HEIGHT_PERCENTILE);
            cells[id].point_count = static_cast<int>(z_values[id].size());
            raw_filled++;
        }
    }

    result.raw_filled = raw_filled;
    result.raw_coverage = static_cast<float>(raw_filled) / static_cast<float>(nx * ny);

    fillMissingByNeighbors(cells, nx, ny);

    int valid_after_neighbor = 0;
    for (const auto& c : cells)
    {
        if (c.valid)
            valid_after_neighbor++;
    }

    int global_filled = 0;
    if (FILL_REMAINING_WITH_GLOBAL_MEDIAN)
        global_filled = fillRemainingByGlobalMedian(cells);

    result.global_filled = global_filled;
    result.interp_filled = valid_after_neighbor - raw_filled;

    float area = RECON_GRID_SIZE * RECON_GRID_SIZE;
    float volume = 0.0f;
    int final_filled = 0;

    for (int id = 0; id < nx * ny; ++id)
    {
        if (!cells[id].valid)
            continue;

        final_filled++;

        float h = cells[id].z_grain - cells[id].z_bottom;

        if (h < 0.0f)
            h = 0.0f;

        if (h > MAX_REASONABLE_GRAIN_HEIGHT)
            h = MAX_REASONABLE_GRAIN_HEIGHT;

        cells[id].height = h;
        volume += h * area;

        PointT p;
        p.x = cells[id].x;
        p.y = cells[id].y;
        p.z = cells[id].z_grain;
        result.reconstructed_cloud->points.push_back(p);
    }

    result.reconstructed_cloud->width = static_cast<uint32_t>(result.reconstructed_cloud->points.size());
    result.reconstructed_cloud->height = 1;
    result.reconstructed_cloud->is_dense = false;

    result.final_filled = final_filled;
    result.final_coverage = static_cast<float>(final_filled) / static_cast<float>(nx * ny);
    result.volume_m3 = volume;
    result.cells = cells;

    std::cout << "========== 3D Reconstruction & Volume ==========" << std::endl;
    std::cout << "[RECON] Method: 2.5D height grid + regular triangular mesh" << std::endl;
    std::cout << "[RECON] Grid size = " << RECON_GRID_SIZE << " m" << std::endl;
    std::cout << "[RECON] nx = " << nx << ", ny = " << ny << std::endl;
    std::cout << "[RECON] Raw filled cells = " << raw_filled << " / " << nx * ny
        << " (" << result.raw_coverage * 100.0f << "%)" << std::endl;
    std::cout << "[RECON] Neighbor interpolated cells = " << result.interp_filled << std::endl;
    std::cout << "[RECON] Global median filled cells = " << result.global_filled << std::endl;
    std::cout << "[RECON] Final filled cells = " << final_filled << " / " << nx * ny
        << " (" << result.final_coverage * 100.0f << "%)" << std::endl;
    std::cout << "[VOLUME] Volume = " << result.volume_m3 << " m^3" << std::endl;
    std::cout << "[VOLUME] Volume = " << result.volume_m3 * 1000.0f << " L" << std::endl;

    return result;
}



bool saveGrainHeightGridCSV(const std::string& filename, const ReconResult& recon)
{
    std::ofstream fout(filename);

    if (!fout.is_open())
    {
        std::cerr << "[ERROR] Cannot write " << filename << std::endl;
        return false;
    }

    fout << std::fixed << std::setprecision(8);
    fout << "ix,iy,x_center,y_center,z_grain,z_bottom,height,valid,interpolated,global_filled,num_points\n";

    for (int ix = 0; ix < recon.nx; ++ix)
    {
        for (int iy = 0; iy < recon.ny; ++iy)
        {
            int id = iy * recon.nx + ix;
            const GridCell& c = recon.cells[id];

            fout << ix << ","
                << iy << ","
                << c.x << ","
                << c.y << ","
                << c.z_grain << ","
                << c.z_bottom << ","
                << c.height << ","
                << (c.valid ? 1 : 0) << ","
                << (c.interpolated ? 1 : 0) << ","
                << (c.global_filled ? 1 : 0) << ","
                << c.point_count << "\n";
        }
    }

    fout.close();
    return true;
}

bool saveVolumeResult(
    const std::string& filename,
    const ReconResult& recon,
    int input_grain_points,
    int candidate_points
)
{
    std::ofstream fout(filename);

    if (!fout.is_open())
    {
        std::cerr << "[ERROR] Cannot write " << filename << std::endl;
        return false;
    }

    fout << std::fixed << std::setprecision(8);
    fout << "method=full_pipeline_2.5D_height_grid_regular_triangular_mesh\n";
    fout << "candidate_points=" << candidate_points << "\n";
    fout << "input_grain_points=" << input_grain_points << "\n";
    fout << "grid_size=" << recon.grid_size << "\n";
    fout << "grid_nx=" << recon.nx << "\n";
    fout << "grid_ny=" << recon.ny << "\n";
    fout << "grid_total=" << recon.nx * recon.ny << "\n";
    fout << "raw_filled=" << recon.raw_filled << "\n";
    fout << "neighbor_interpolated=" << recon.interp_filled << "\n";
    fout << "global_median_filled=" << recon.global_filled << "\n";
    fout << "final_filled=" << recon.final_filled << "\n";
    fout << "raw_coverage=" << recon.raw_coverage << "\n";
    fout << "final_coverage=" << recon.final_coverage << "\n";
    fout << "volume_m3=" << recon.volume_m3 << "\n";
    fout << "volume_liter=" << recon.volume_m3 * 1000.0f << "\n";
    fout << "surface_mesh_file=" << OUT_SURFACE_PLY << "\n";
    fout << "solid_mesh_file=" << OUT_SOLID_PLY << "\n";

    fout.close();
    return true;
}


bool savePLY(
    const std::string& filename,
    const std::vector<Vertex>& vertices,
    const std::vector<Face>& faces
)
{
    std::ofstream fout(filename);

    if (!fout.is_open())
    {
        std::cerr << "[ERROR] Cannot write " << filename << std::endl;
        return false;
    }

    fout << std::fixed << std::setprecision(6);
    fout << "ply\n";
    fout << "format ascii 1.0\n";
    fout << "element vertex " << vertices.size() << "\n";
    fout << "property float x\n";
    fout << "property float y\n";
    fout << "property float z\n";
    fout << "element face " << faces.size() << "\n";
    fout << "property list uchar int vertex_indices\n";
    fout << "end_header\n";

    for (const auto& v : vertices)
        fout << v.x << " " << v.y << " " << v.z << "\n";

    for (const auto& f : faces)
        fout << "3 " << f.a << " " << f.b << " " << f.c << "\n";

    fout.close();
    return true;
}

void buildNodeHeights(
    const ReconResult& recon,
    std::vector<int>& node_valid,
    std::vector<float>& node_top_z,
    std::vector<float>& node_bottom_z,
    const CalibParams& calib,
    const BottomGrid& bottom_grid
)
{
    int nx = recon.nx;
    int ny = recon.ny;
    int node_nx = nx + 1;
    int node_ny = ny + 1;

    node_valid.assign(node_nx * node_ny, 0);
    node_top_z.assign(node_nx * node_ny, 0.0f);
    node_bottom_z.assign(node_nx * node_ny, 0.0f);

    for (int ix = 0; ix <= nx; ++ix)
    {
        for (int iy = 0; iy <= ny; ++iy)
        {
            float sum_z = 0.0f;
            int count = 0;

            for (int dx = -1; dx <= 0; ++dx)
            {
                for (int dy = -1; dy <= 0; ++dy)
                {
                    int cx = ix + dx;
                    int cy = iy + dy;

                    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny)
                        continue;

                    int cid = cy * nx + cx;

                    if (recon.cells[cid].valid)
                    {
                        sum_z += recon.cells[cid].z_grain;
                        count++;
                    }
                }
            }

            int nid = iy * node_nx + ix;

            float x = calib.x_min + ix * RECON_GRID_SIZE;
            float y = calib.y_min + iy * RECON_GRID_SIZE;

            node_bottom_z[nid] = getBottomZ(calib, bottom_grid, x, y);

            if (count > 0)
            {
                node_valid[nid] = 1;
                node_top_z[nid] = sum_z / static_cast<float>(count);
            }
        }
    }
}

bool saveSurfaceMeshPLY(
    const std::string& filename,
    const ReconResult& recon,
    const CalibParams& calib,
    const BottomGrid& bottom_grid
)
{
    int nx = recon.nx;
    int ny = recon.ny;
    int node_nx = nx + 1;
    int node_ny = ny + 1;

    std::vector<int> node_valid;
    std::vector<float> node_top_z;
    std::vector<float> node_bottom_z;
    buildNodeHeights(recon, node_valid, node_top_z, node_bottom_z, calib, bottom_grid);

    std::vector<int> node_id(node_nx * node_ny, -1);
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    for (int ix = 0; ix <= nx; ++ix)
    {
        for (int iy = 0; iy <= ny; ++iy)
        {
            int nid = iy * node_nx + ix;

            if (!node_valid[nid])
                continue;

            Vertex v;
            v.x = calib.x_min + ix * RECON_GRID_SIZE;
            v.y = calib.y_min + iy * RECON_GRID_SIZE;
            v.z = node_top_z[nid];

            node_id[nid] = static_cast<int>(vertices.size());
            vertices.push_back(v);
        }
    }

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            int cid = iy * nx + ix;
            if (!recon.cells[cid].valid)
                continue;

            int n00 = iy * node_nx + ix;
            int n10 = iy * node_nx + (ix + 1);
            int n01 = (iy + 1) * node_nx + ix;
            int n11 = (iy + 1) * node_nx + (ix + 1);

            if (node_id[n00] < 0 || node_id[n10] < 0 ||
                node_id[n01] < 0 || node_id[n11] < 0)
                continue;

            faces.push_back({ node_id[n00], node_id[n10], node_id[n11] });
            faces.push_back({ node_id[n00], node_id[n11], node_id[n01] });
        }
    }

    bool ok = savePLY(filename, vertices, faces);

    std::cout << "[MESH] Surface mesh: " << filename
        << ", vertices = " << vertices.size()
        << ", faces = " << faces.size() << std::endl;

    return ok;
}

bool saveSolidMeshPLY(
    const std::string& filename,
    const ReconResult& recon,
    const CalibParams& calib,
    const BottomGrid& bottom_grid
)
{
    int nx = recon.nx;
    int ny = recon.ny;
    int node_nx = nx + 1;
    int node_ny = ny + 1;

    std::vector<int> node_valid;
    std::vector<float> node_top_z;
    std::vector<float> node_bottom_z;
    buildNodeHeights(recon, node_valid, node_top_z, node_bottom_z, calib, bottom_grid);

    std::vector<int> top_id(node_nx * node_ny, -1);
    std::vector<int> bot_id(node_nx * node_ny, -1);
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    for (int ix = 0; ix <= nx; ++ix)
    {
        for (int iy = 0; iy <= ny; ++iy)
        {
            int nid = iy * node_nx + ix;
            if (!node_valid[nid])
                continue;

            float x = calib.x_min + ix * RECON_GRID_SIZE;
            float y = calib.y_min + iy * RECON_GRID_SIZE;

            Vertex vt;
            vt.x = x;
            vt.y = y;
            vt.z = node_top_z[nid];
            top_id[nid] = static_cast<int>(vertices.size());
            vertices.push_back(vt);

            Vertex vb;
            vb.x = x;
            vb.y = y;
            vb.z = node_bottom_z[nid];
            bot_id[nid] = static_cast<int>(vertices.size());
            vertices.push_back(vb);
        }
    }

    auto addQuad = [&](int a, int b, int c, int d)
        {
            if (a < 0 || b < 0 || c < 0 || d < 0)
                return;
            faces.push_back({ a, b, c });
            faces.push_back({ a, c, d });
        };

    auto cellValid = [&](int ix, int iy) -> bool
        {
            if (ix < 0 || ix >= nx || iy < 0 || iy >= ny)
                return false;
            return recon.cells[iy * nx + ix].valid;
        };

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            if (!cellValid(ix, iy))
                continue;

            int n00 = iy * node_nx + ix;
            int n10 = iy * node_nx + (ix + 1);
            int n01 = (iy + 1) * node_nx + ix;
            int n11 = (iy + 1) * node_nx + (ix + 1);

            addQuad(top_id[n00], top_id[n10], top_id[n11], top_id[n01]);
            addQuad(bot_id[n00], bot_id[n01], bot_id[n11], bot_id[n10]);

            if (!cellValid(ix - 1, iy))
                addQuad(top_id[n00], top_id[n01], bot_id[n01], bot_id[n00]);

            if (!cellValid(ix + 1, iy))
                addQuad(top_id[n10], bot_id[n10], bot_id[n11], top_id[n11]);

            if (!cellValid(ix, iy - 1))
                addQuad(top_id[n00], bot_id[n00], bot_id[n10], top_id[n10]);

            if (!cellValid(ix, iy + 1))
                addQuad(top_id[n01], top_id[n11], bot_id[n11], bot_id[n01]);
        }
    }

    bool ok = savePLY(filename, vertices, faces);

    std::cout << "[MESH] Solid mesh: " << filename
        << ", vertices = " << vertices.size()
        << ", faces = " << faces.size() << std::endl;

    return ok;
}



void showCloud(const CloudT::Ptr& cloud, const std::string& title)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> color(cloud, 255, 165, 0);
    viewer->addPointCloud<PointT>(cloud, color, "cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud"
    );

    viewer->addCoordinateSystem(0.2);

    while (!viewer->wasStopped())
        viewer->spinOnce(100);
}



int main()
{
    std::cout << "[INFO] Start final engineering pipeline..." << std::endl;

    // 0. 读取空箱标定
    CalibParams calib;
    if (!loadCalibParams(CALIB_FILE, calib))
    {
        std::cerr << "[ERROR] Please run empty-box calibration first." << std::endl;
        return -1;
    }

    BottomGrid bottom_grid;
    loadBottomHeightGrid(BOTTOM_GRID_FILE, calib, bottom_grid);

    // 1. 读取有粮点云
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

    Eigen::Affine3f calib_tf = Eigen::Affine3f::Identity();

    if (!loadTransformMatrix(CALIB_TRANSFORM_FILE, calib_tf))
    {
        std::cerr << "[ERROR] Please run empty-box calibration first and generate "
                  << CALIB_TRANSFORM_FILE << std::endl;
        return -1;
    }

    CloudT::Ptr cloud_aligned = transformCloudEigen(cloud_raw, calib_tf);
    savePCDAscii(OUT_ALIGNED, cloud_aligned);

    CloudT::Ptr clean_walls_check = extractCleanBoxWallsForCheckByCalib(cloud_aligned, calib);
    savePCDAscii(OUT_WALLS_CHECK, clean_walls_check);

    std::cout << "[INFO] Applied empty-box calibration transform." << std::endl;
    std::cout << "[INFO] Clean wall check points: " << clean_walls_check->size() << std::endl;


    CloudT::Ptr cloud_candidate = makeExpandedROIByCalib(cloud_aligned, calib);
    savePCDAscii(OUT_EXPANDED_ROI, cloud_candidate);


    CloudT::Ptr grain_surface_raw = segmentGrainSurface(cloud_candidate, calib);
    savePCDAscii(OUT_GRAIN_SURFACE_RAW, grain_surface_raw);


    CloudT::Ptr grain_surface_rim_filtered = removeRimResidualPoints(grain_surface_raw, calib);
    savePCDAscii(OUT_GRAIN_SURFACE_RIM_FILTERED, grain_surface_rim_filtered);


    CloudT::Ptr grain_surface = keepLargestXYComponent(grain_surface_rim_filtered, calib);
    savePCDAscii(OUT_GRAIN_SURFACE_COMPONENT_FILTERED, grain_surface);


    savePCDAscii(OUT_GRAIN_SURFACE, grain_surface);


    ReconResult recon = reconstructAndComputeVolume(grain_surface, calib, bottom_grid);

    savePCDAscii(OUT_RECON_PCD, recon.reconstructed_cloud);
    saveGrainHeightGridCSV(OUT_HEIGHT_GRID_CSV, recon);
    saveSurfaceMeshPLY(OUT_SURFACE_PLY, recon, calib, bottom_grid);
    saveSolidMeshPLY(OUT_SOLID_PLY, recon, calib, bottom_grid);
    saveVolumeResult(OUT_VOLUME_RESULT, recon,
        static_cast<int>(grain_surface->size()),
        static_cast<int>(cloud_candidate->size()));

    std::cout << "========================================" << std::endl;
    std::cout << "[SUCCESS] Final engineering pipeline finished." << std::endl;
    std::cout << "[INFO] Output files:" << std::endl;
    std::cout << "  - input transform: " << CALIB_TRANSFORM_FILE << std::endl;
    std::cout << "  - " << OUT_WALLS_CHECK << std::endl;
    std::cout << "  - " << OUT_ALIGNED << std::endl;
    std::cout << "  - " << OUT_EXPANDED_ROI << std::endl;
    std::cout << "  - " << OUT_GRAIN_SURFACE_RAW << std::endl;
    std::cout << "  - " << OUT_GRAIN_SURFACE_RIM_FILTERED << std::endl;
    std::cout << "  - " << OUT_GRAIN_SURFACE_COMPONENT_FILTERED << std::endl;
    std::cout << "  - " << OUT_GRAIN_SURFACE << std::endl;
    std::cout << "  - " << OUT_RECON_PCD << std::endl;
    std::cout << "  - " << OUT_HEIGHT_GRID_CSV << std::endl;
    std::cout << "  - " << OUT_SURFACE_PLY << std::endl;
    std::cout << "  - " << OUT_SOLID_PLY << std::endl;
    std::cout << "  - " << OUT_VOLUME_RESULT << std::endl;

    // Interactive visualization is disabled for unattended batch processing.

    return 0;
}
