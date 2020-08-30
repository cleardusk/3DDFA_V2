#ifndef MESH_CORE_HPP_
#define MESH_CORE_HPP_

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class Point3D {
public:
    float x;
    float y;
    float z;

public:
    Point3D() : x(0.f), y(0.f), z(0.f) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    void initialize(float x_, float y_, float z_){
        this->x = x_; this->y = y_; this->z = z_;
    }

    Point3D cross(Point3D &p){
        Point3D c;
        c.x = this->y * p.z - this->z * p.y;
        c.y = this->z * p.x - this->x * p.z;
        c.z = this->x * p.y - this->y * p.x;
        return c;
    }

    float dot(Point3D &p) {
        return this->x * p.x + this->y * p.y + this->z * p.z;
    }

    Point3D operator-(const Point3D &p) {
        Point3D np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        np.z = this->z - p.z;
        return np;
    }

};

class Point {
public:
    float x;
    float y;

public:
    Point() : x(0.f), y(0.f) {}
    Point(float x_, float y_) : x(x_), y(y_) {}
    float dot(Point p) {
        return this->x * p.x + this->y * p.y;
    }

    Point operator-(const Point &p) {
        Point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    Point operator+(const Point &p) {
        Point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    Point operator*(float s) {
        Point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
};


bool is_point_in_tri(Point p, Point p0, Point p1, Point p2);

void get_point_weight(float *weight, Point p, Point p0, Point p1, Point p2);

void _get_tri_normal(float *tri_normal, float *vertices, int *triangles, int ntri, bool norm_flg);

void _get_ver_normal(float *ver_normal, float *tri_normal, int *triangles, int nver, int ntri);

void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri);

void _rasterize_triangles(
        float *vertices, int *triangles, float *depth_buffer, int *triangle_buffer, float *barycentric_weight,
        int ntri, int h, int w);

void _rasterize(
        unsigned char *image, float *vertices, int *triangles, float *colors,
        float *depth_buffer, int ntri, int h, int w, int c, float alpha, bool reverse);

void _render_texture_core(
        float *image, float *vertices, int *triangles,
        float *texture, float *tex_coords, int *tex_triangles,
        float *depth_buffer,
        int nver, int tex_nver, int ntri,
        int h, int w, int c,
        int tex_h, int tex_w, int tex_c,
        int mapping_type);

void _write_obj_with_colors_texture(string filename, string mtl_name,
                                    float *vertices, int *triangles, float *colors, float *uv_coords,
                                    int nver, int ntri, int ntexver);

#endif