/*
 * Tesing cases
 */

#include <iostream>
#include <time.h>
#include "rasterize.h"
#include "io.h"

void test_isPointInTri() {
    Point p0(0, 0);
    Point p1(1, 0);
    Point p2(1, 1);

    Point p(0.2, 0.2);

    if (is_point_in_tri(p, p0, p1, p2))
        std::cout << "In";
    else
        std::cout << "Out";
    std::cout << std::endl;
}

void test_getPointWeight() {
    Point p0(0, 0);
    Point p1(1, 0);
    Point p2(1, 1);

    Point p(0.2, 0.2);

    float weight[3];
    get_point_weight(weight, p, p0, p1, p2);
    std::cout << weight[0] << " " << weight[1] << " " << weight[2] << std::endl;
}

void test_get_tri_normal() {
    float tri_normal[3];
//    float vertices[9] = {1, 0, 0, 0, 0, 0, 0, 1, 0};
    float vertices[9] = {1, 1.1, 0, 0, 0, 0, 0, 0.6, 0.7};
    int triangles[3] = {0, 1, 2};
    int ntri = 1;

    _get_tri_normal(tri_normal, vertices, triangles, ntri);

    for (int i = 0; i < 3; ++i)
        std::cout << tri_normal[i] << ", ";
    std::cout << std::endl;
}

void test_load_obj() {
    const char *fp = "../data/vd005_mesh.obj";
    int nver = 35709;
    int ntri = 70789;

    auto *vertices = new float[nver];
    auto *colors = new float[nver];
    auto *triangles = new int[ntri];
    load_obj(fp, vertices, colors, triangles, nver, ntri);

    delete[] vertices;
    delete[] colors;
    delete[] triangles;
}

void test_render() {
    // 1. loading obj
//    const char *fp = "/Users/gjz/gjzprojects/Sim3DR/data/vd005_mesh.obj";
    const char *fp = "/Users/gjz/gjzprojects/Sim3DR/data/face1.obj";
    int nver = 35709; //53215; //35709;
    int ntri = 70789; //105840;//70789;

    auto *vertices = new float[3 * nver];
    auto *colors = new float[3 * nver];
    auto *triangles = new int[3 * ntri];
    load_obj(fp, vertices, colors, triangles, nver, ntri);

    // 2. rendering
    int h = 224, w = 224, c = 3;

    // enlarging
    int scale = 4;
    h *= scale;
    w *= scale;
    for (int i = 0; i < nver * 3; ++i) vertices[i] *= scale;

    auto *image = new unsigned char[h * w * c]();
    auto *depth_buffer = new float[h * w]();

    for (int i = 0; i < h * w; ++i) depth_buffer[i] = -999999;

    clock_t t;
    t = clock();

    _rasterize(image, vertices, triangles, colors, depth_buffer, ntri, h, w, c, true);
    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    printf("Render took %f seconds to execute \n", time_taken);


//    auto *image_char = new u_char[h * w * c]();
//    for (int i = 0; i < h * w * c; ++i)
//        image_char[i] = u_char(255 * image[i]);
    write_ppm("res.ppm", image, h, w, c);

//    delete[] image_char;
    delete[] vertices;
    delete[] colors;
    delete[] triangles;
    delete[] image;
    delete[] depth_buffer;
}

void test_light() {
    // 1. loading obj
    const char *fp = "/Users/gjz/gjzprojects/Sim3DR/data/emma_input_0_noheader.ply";
    int nver = 53215; //35709;
    int ntri = 105840; //70789;

    auto *vertices = new float[3 * nver];
    auto *colors = new float[3 * nver];
    auto *triangles = new int[3 * ntri];
    load_ply(fp, vertices, triangles, nver, ntri);

    // 2. rendering
//    int h = 1901, w = 3913, c = 3;
    int h = 2000, w = 4000, c = 3;

    // enlarging
//    int scale = 1;
//    h *= scale;
//    w *= scale;
//    for (int i = 0; i < nver * 3; ++i) vertices[i] *= scale;

    auto *image = new unsigned char[h * w * c]();
    auto *depth_buffer = new float[h * w]();

    for (int i = 0; i < h * w; ++i) depth_buffer[i] = -999999;
    for (int i = 0; i < 3 * nver; ++i) colors[i] = 0.8;

    clock_t t;
    t = clock();

    _rasterize(image, vertices, triangles, colors, depth_buffer, ntri, h, w, c, true);
    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    printf("Render took %f seconds to execute \n", time_taken);


//    auto *image_char = new u_char[h * w * c]();
//    for (int i = 0; i < h * w * c; ++i)
//        image_char[i] = u_char(255 * image[i]);
    write_ppm("emma.ppm", image, h, w, c);

//    delete[] image_char;
    delete[] vertices;
    delete[] colors;
    delete[] triangles;
    delete[] image;
    delete[] depth_buffer;
}

int main(int argc, char *argv[]) {
//    std::cout << "Hello CMake!" << std::endl;

//    test_isPointInTri();
//    test_getPointWeight();
//    test_get_tri_normal();
//    test_load_obj();
//    test_render();
    test_light();
    return 0;
}