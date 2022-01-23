/*
 Author: Yao Feng
 Modified by Jianzhu Guo

 functions that can not be optimazed by vertorization in python.
 1. rasterization.(need process each triangle)
 2. normal of each vertex.(use one-ring, need process each vertex)
 3. write obj(seems that it can be verctorized? anyway, writing it in c++ is simple, so also add function here. --> however, why writting in c++ is still slow?)



*/

#include "rasterize.h"


/* Judge whether the Point is in the triangle
Method:
    http://blackpawn.com/texts/pointinpoly/
Args:
    Point: [x, y]
    tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
Returns:
    bool: true for in triangle
*/
bool is_point_in_tri(Point p, Point p0, Point p1, Point p2) {
    // vectors
    Point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if (dot00 * dot11 - dot01 * dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // check if Point in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}

void get_point_weight(float *weight, Point p, Point p0, Point p1, Point p2) {
    // vectors
    Point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if (dot00 * dot11 - dot01 * dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}

/*
 * Get normals of triangles.
 */
void _get_tri_normal(float *tri_normal, float *vertices, int *triangles, int ntri, bool norm_flg) {
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    float v1x, v1y, v1z, v2x, v2y, v2z;

    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        // counter clockwise order
        v1x = vertices[3 * tri_p1_ind] - vertices[3 * tri_p0_ind];
        v1y = vertices[3 * tri_p1_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v1z = vertices[3 * tri_p1_ind + 2] - vertices[3 * tri_p0_ind + 2];

        v2x = vertices[3 * tri_p2_ind] - vertices[3 * tri_p0_ind];
        v2y = vertices[3 * tri_p2_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v2z = vertices[3 * tri_p2_ind + 2] - vertices[3 * tri_p0_ind + 2];

        if (norm_flg) {
            float c1 = v1y * v2z - v1z * v2y;
            float c2 = v1z * v2x - v1x * v2z;
            float c3 = v1x * v2y - v1y * v2x;
            float det = sqrt(c1 * c1 + c2 * c2 + c3 * c3);
            if (det <= 0) det = 1e-6;
            tri_normal[3 * i] = c1 / det;
            tri_normal[3 * i + 1] = c2 / det;
            tri_normal[3 * i + 2] = c3 / det;
        } else {
            tri_normal[3 * i] = v1y * v2z - v1z * v2y;
            tri_normal[3 * i + 1] = v1z * v2x - v1x * v2z;
            tri_normal[3 * i + 2] = v1x * v2y - v1y * v2x;
        }
    }
}

/*
 * Get normal vector of vertices using triangle normals
 */
void _get_ver_normal(float *ver_normal, float *tri_normal, int *triangles, int nver, int ntri) {
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;

    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        for (int j = 0; j < 3; j++) {
            ver_normal[3 * tri_p0_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p1_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p2_ind + j] += tri_normal[3 * i + j];
        }
    }

    // normalizing
    float nx, ny, nz, det;
    for (int i = 0; i < nver; ++i) {
        nx = ver_normal[3 * i];
        ny = ver_normal[3 * i + 1];
        nz = ver_normal[3 * i + 2];

        det = sqrt(nx * nx + ny * ny + nz * nz);
        if (det <= 0) det = 1e-6;
        ver_normal[3 * i] = nx / det;
        ver_normal[3 * i + 1] = ny / det;
        ver_normal[3 * i + 2] = nz / det;
    }
}

/*
 * Directly get normal of vertices, which can be regraded as a combination of _get_tri_normal and _get_ver_normal
 */
void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri) {
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    float v1x, v1y, v1z, v2x, v2y, v2z;

    // get tri_normal
//    float tri_normal[3 * ntri];
    float *tri_normal;
    tri_normal = new float[3 * ntri];
    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        // counter clockwise order
        v1x = vertices[3 * tri_p1_ind] - vertices[3 * tri_p0_ind];
        v1y = vertices[3 * tri_p1_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v1z = vertices[3 * tri_p1_ind + 2] - vertices[3 * tri_p0_ind + 2];

        v2x = vertices[3 * tri_p2_ind] - vertices[3 * tri_p0_ind];
        v2y = vertices[3 * tri_p2_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v2z = vertices[3 * tri_p2_ind + 2] - vertices[3 * tri_p0_ind + 2];


        tri_normal[3 * i] = v1y * v2z - v1z * v2y;
        tri_normal[3 * i + 1] = v1z * v2x - v1x * v2z;
        tri_normal[3 * i + 2] = v1x * v2y - v1y * v2x;

    }

    // get ver_normal
    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        for (int j = 0; j < 3; j++) {
            ver_normal[3 * tri_p0_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p1_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p2_ind + j] += tri_normal[3 * i + j];
        }
    }

    // normalizing
    float nx, ny, nz, det;
    for (int i = 0; i < nver; ++i) {
        nx = ver_normal[3 * i];
        ny = ver_normal[3 * i + 1];
        nz = ver_normal[3 * i + 2];

        det = sqrt(nx * nx + ny * ny + nz * nz);
        if (det <= 0) det = 1e-6;
        ver_normal[3 * i] = nx / det;
        ver_normal[3 * i + 1] = ny / det;
        ver_normal[3 * i + 2] = nz / det;
    }

    delete[] tri_normal;
}

// rasterization by Z-Buffer with optimization
// Complexity: < ntri * h * w * c
void _rasterize(
        unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer,
        int ntri, int h, int w, int c, float alpha, bool reverse) {
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    Point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float p_color, p0_color, p1_color, p2_color;
    float weight[3];

    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[3 * tri_p0_ind];
        p0.y = vertices[3 * tri_p0_ind + 1];
        p0_depth = vertices[3 * tri_p0_ind + 2];
        p1.x = vertices[3 * tri_p1_ind];
        p1.y = vertices[3 * tri_p1_ind + 1];
        p1_depth = vertices[3 * tri_p1_ind + 2];
        p2.x = vertices[3 * tri_p2_ind];
        p2.y = vertices[3 * tri_p2_ind + 1];
        p2_depth = vertices[3 * tri_p2_ind + 2];

        x_min = max((int) ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int) floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int) ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int) floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if (x_max < x_min || y_max < y_min) {
            continue;
        }

        for (y = y_min; y <= y_max; y++) {
            for (x = x_min; x <= x_max; x++) {
                p.x = float(x);
                p.y = float(y);

                // call get_point_weight function once
                get_point_weight(weight, p, p0, p1, p2);

                // and judge is_point_in_tri by below line of code
                if (weight[2] > 0 && weight[1] > 0 && weight[0] > 0) {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if ((p_depth > depth_buffer[y * w + x])) {
                        for (k = 0; k < c; k++) {
                            p0_color = colors[c * tri_p0_ind + k];
                            p1_color = colors[c * tri_p1_ind + k];
                            p2_color = colors[c * tri_p2_ind + k];

                            p_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color;
                            if (reverse) {
                                image[(h - 1 - y) * w * c + x * c + k] = (unsigned char) (
                                        (1 - alpha) * image[(h - 1 - y) * w * c + x * c + k] + alpha * 255 * p_color);
//                                image[(h - 1 - y) * w * c + x * c + k] = (unsigned char) (255 * p_color);
                            } else {
                                image[y * w * c + x * c + k] = (unsigned char) (
                                        (1 - alpha) * image[y * w * c + x * c + k] + alpha * 255 * p_color);
//                                image[y * w * c + x * c + k] = (unsigned char) (255 * p_color);
                            }
                        }

                        depth_buffer[y * w + x] = p_depth;
                    }
                }
            }
        }
    }
}


void _rasterize_triangles(
        float *vertices, int *triangles, float *depth_buffer, int *triangle_buffer, float *barycentric_weight,
        int ntri, int h, int w) {
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    Point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float weight[3];

    for (i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[3 * tri_p0_ind];
        p0.y = vertices[3 * tri_p0_ind + 1];
        p0_depth = vertices[3 * tri_p0_ind + 2];
        p1.x = vertices[3 * tri_p1_ind];
        p1.y = vertices[3 * tri_p1_ind + 1];
        p1_depth = vertices[3 * tri_p1_ind + 2];
        p2.x = vertices[3 * tri_p2_ind];
        p2.y = vertices[3 * tri_p2_ind + 1];
        p2_depth = vertices[3 * tri_p2_ind + 2];

        x_min = max((int) ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int) floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int) ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int) floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if (x_max < x_min || y_max < y_min) {
            continue;
        }

        for (y = y_min; y <= y_max; y++) //h
        {
            for (x = x_min; x <= x_max; x++) //w
            {
                p.x = x;
                p.y = y;
//                if (p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3 || is_point_in_tri(p, p0, p1, p2)) {
                if (is_point_in_tri(p, p0, p1, p2)) {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if ((p_depth > depth_buffer[y * w + x])) {
                        depth_buffer[y * w + x] = p_depth;
                        triangle_buffer[y * w + x] = i;
                        for (k = 0; k < 3; k++) {
                            barycentric_weight[y * w * 3 + x * 3 + k] = weight[k];
                        }
                    }
                }
            }
        }
    }
}


// Depth-Buffer 算法
// https://blog.csdn.net/Jurbo/article/details/75007260
void _render_texture_core(
        float *image, float *vertices, int *triangles,
        float *texture, float *tex_coords, int *tex_triangles,
        float *depth_buffer,
        int nver, int tex_nver, int ntri,
        int h, int w, int c,
        int tex_h, int tex_w, int tex_c,
        int mapping_type) {
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    int tex_tri_p0_ind, tex_tri_p1_ind, tex_tri_p2_ind;
    Point p0, p1, p2, p;
    Point tex_p0, tex_p1, tex_p2, tex_p;
    int x_min, x_max, y_min, y_max;
    float weight[3];
    float p_depth, p0_depth, p1_depth, p2_depth;
    float xd, yd;
    float ul, ur, dl, dr;
    for (i = 0; i < ntri; i++) {
        // mesh
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[3 * tri_p0_ind];
        p0.y = vertices[3 * tri_p0_ind + 1];
        p0_depth = vertices[3 * tri_p0_ind + 2];
        p1.x = vertices[3 * tri_p1_ind];
        p1.y = vertices[3 * tri_p1_ind + 1];
        p1_depth = vertices[3 * tri_p1_ind + 2];
        p2.x = vertices[3 * tri_p2_ind];
        p2.y = vertices[3 * tri_p2_ind + 1];
        p2_depth = vertices[3 * tri_p2_ind + 2];

        // texture
        tex_tri_p0_ind = tex_triangles[3 * i];
        tex_tri_p1_ind = tex_triangles[3 * i + 1];
        tex_tri_p2_ind = tex_triangles[3 * i + 2];

        tex_p0.x = tex_coords[3 * tex_tri_p0_ind];
        tex_p0.y = tex_coords[3 * tri_p0_ind + 1];
        tex_p1.x = tex_coords[3 * tex_tri_p1_ind];
        tex_p1.y = tex_coords[3 * tri_p1_ind + 1];
        tex_p2.x = tex_coords[3 * tex_tri_p2_ind];
        tex_p2.y = tex_coords[3 * tri_p2_ind + 1];


        x_min = max((int) ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int) floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int) ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int) floor(max(p0.y, max(p1.y, p2.y))), h - 1);


        if (x_max < x_min || y_max < y_min) {
            continue;
        }

        for (y = y_min; y <= y_max; y++) //h
        {
            for (x = x_min; x <= x_max; x++) //w
            {
                p.x = x;
                p.y = y;
                if (p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3 || is_point_in_tri(p, p0, p1, p2)) {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if ((p_depth > depth_buffer[y * w + x])) {
                        // -- color from texture
                        // cal weight in mesh tri
                        get_point_weight(weight, p, p0, p1, p2);
                        // cal coord in texture
                        tex_p = tex_p0 * weight[0] + tex_p1 * weight[1] + tex_p2 * weight[2];
                        tex_p.x = max(min(tex_p.x, float(tex_w - 1)), float(0));
                        tex_p.y = max(min(tex_p.y, float(tex_h - 1)), float(0));

                        yd = tex_p.y - floor(tex_p.y);
                        xd = tex_p.x - floor(tex_p.x);
                        for (k = 0; k < c; k++) {
                            if (mapping_type == 0)// nearest
                            {
                                image[y * w * c + x * c + k] = texture[int(round(tex_p.y)) * tex_w * tex_c +
                                                                       int(round(tex_p.x)) * tex_c + k];
                            } else//bilinear interp
                            {
                                ul = texture[(int) floor(tex_p.y) * tex_w * tex_c + (int) floor(tex_p.x) * tex_c + k];
                                ur = texture[(int) floor(tex_p.y) * tex_w * tex_c + (int) ceil(tex_p.x) * tex_c + k];
                                dl = texture[(int) ceil(tex_p.y) * tex_w * tex_c + (int) floor(tex_p.x) * tex_c + k];
                                dr = texture[(int) ceil(tex_p.y) * tex_w * tex_c + (int) ceil(tex_p.x) * tex_c + k];

                                image[y * w * c + x * c + k] =
                                        ul * (1 - xd) * (1 - yd) + ur * xd * (1 - yd) + dl * (1 - xd) * yd +
                                        dr * xd * yd;
                            }

                        }

                        depth_buffer[y * w + x] = p_depth;
                    }
                }
            }
        }
    }
}


// ------------------------------------------------- write
// obj write
// Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/core/Mesh.hpp
void _write_obj_with_colors_texture(string filename, string mtl_name,
                                    float *vertices, int *triangles, float *colors, float *uv_coords,
                                    int nver, int ntri, int ntexver) {
    int i;

    ofstream obj_file(filename);

    // first line of the obj file: the mtl name
    obj_file << "mtllib " << mtl_name << endl;

    // write vertices
    for (i = 0; i < nver; ++i) {
        obj_file << "v " << vertices[3 * i] << " " << vertices[3 * i + 1] << " " << vertices[3 * i + 2] << colors[3 * i]
                 << " " << colors[3 * i + 1] << " " << colors[3 * i + 2] << endl;
    }

    // write uv coordinates
    for (i = 0; i < ntexver; ++i) {
        //obj_file << "vt " << uv_coords[2*i] << " " << (1 - uv_coords[2*i + 1]) << endl;
        obj_file << "vt " << uv_coords[2 * i] << " " << uv_coords[2 * i + 1] << endl;
    }

    obj_file << "usemtl FaceTexture" << endl;
    // write triangles
    for (i = 0; i < ntri; ++i) {
        // obj_file << "f " << triangles[3*i] << "/" << triangles[3*i] << " " << triangles[3*i + 1] << "/" << triangles[3*i + 1] << " " << triangles[3*i + 2] << "/" << triangles[3*i + 2] << endl;
        obj_file << "f " << triangles[3 * i + 2] << "/" << triangles[3 * i + 2] << " " << triangles[3 * i + 1] << "/"
                 << triangles[3 * i + 1] << " " << triangles[3 * i] << "/" << triangles[3 * i] << endl;
    }

}