#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define clip(_x, _min, _max) min(max(_x, _min), _max)

struct Tuple3D
{
    float x;
    float y;
    float z;
};

void _render(const int *triangles,
             const int ntri,
             const float *light,
             const float *directional,
             const float *ambient,
             const float *vertices,
             const int nver,
             unsigned char *image,
             const int h, const int w)
{
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    int color_index;
    float dot00, dot01, dot11, dot02, dot12;
    float cos_sum, det;

    struct Tuple3D p0, p1, p2;
    struct Tuple3D v0, v1, v2;
    struct Tuple3D p, start, end;

    struct Tuple3D ver_max = {-1.0e8, -1.0e8, -1.0e8};
    struct Tuple3D ver_min = {1.0e8, 1.0e8, 1.0e8};
    struct Tuple3D ver_mean = {0.0, 0.0, 0.0};

    float *ver_normal = (float *)calloc(3 * nver, sizeof(float));
    float *colors = (float *)malloc(3 * nver * sizeof(float));
    float *depth_buffer = (float *)calloc(h * w, sizeof(float));

    for (int i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        // counter clockwise order
        start.x = vertices[tri_p1_ind] - vertices[tri_p0_ind];
        start.y = vertices[tri_p1_ind + 1] - vertices[tri_p0_ind + 1];
        start.z = vertices[tri_p1_ind + 2] - vertices[tri_p0_ind + 2];

        end.x = vertices[tri_p2_ind] - vertices[tri_p0_ind];
        end.y = vertices[tri_p2_ind + 1] - vertices[tri_p0_ind + 1];
        end.z = vertices[tri_p2_ind + 2] - vertices[tri_p0_ind + 2];

        p.x = start.y * end.z - start.z * end.y;
        p.y = start.z * end.x - start.x * end.z;
        p.z = start.x * end.y - start.y * end.x;

        ver_normal[tri_p0_ind] += p.x;
        ver_normal[tri_p1_ind] += p.x;
        ver_normal[tri_p2_ind] += p.x;

        ver_normal[tri_p0_ind + 1] += p.y;
        ver_normal[tri_p1_ind + 1] += p.y;
        ver_normal[tri_p2_ind + 1] += p.y;

        ver_normal[tri_p0_ind + 2] += p.z;
        ver_normal[tri_p1_ind + 2] += p.z;
        ver_normal[tri_p2_ind + 2] += p.z;
    }

    for (int i = 0; i < nver; ++i)
    {
        p.x = ver_normal[3 * i];
        p.y = ver_normal[3 * i + 1];
        p.z = ver_normal[3 * i + 2];

        det = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (det <= 0)
            det = 1e-6;

        ver_normal[3 * i] /= det;
        ver_normal[3 * i + 1] /= det;
        ver_normal[3 * i + 2] /= det;

        ver_mean.x += p.x;
        ver_mean.y += p.y;
        ver_mean.z += p.z;

        ver_max.x = max(ver_max.x, p.x);
        ver_max.y = max(ver_max.y, p.y);
        ver_max.z = max(ver_max.z, p.z);

        ver_min.x = min(ver_min.x, p.x);
        ver_min.y = min(ver_min.y, p.y);
        ver_min.z = min(ver_min.z, p.z);
    }

    ver_mean.x /= nver;
    ver_mean.y /= nver;
    ver_mean.z /= nver;

    for (int i = 0; i < nver; ++i)
    {
        colors[3 * i] = vertices[3 * i];
        colors[3 * i + 1] = vertices[3 * i + 1];
        colors[3 * i + 2] = vertices[3 * i + 2];

        colors[3 * i] -= ver_mean.x;
        colors[3 * i] /= ver_max.x - ver_min.x;

        colors[3 * i + 1] -= ver_mean.y;
        colors[3 * i + 1] /= ver_max.y - ver_min.y;

        colors[3 * i + 2] -= ver_mean.z;
        colors[3 * i + 2] /= ver_max.z - ver_min.z;

        p.x = light[0] - colors[3 * i];
        p.y = light[1] - colors[3 * i + 1];
        p.z = light[2] - colors[3 * i + 2];

        det = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (det <= 0)
            det = 1e-6;

        colors[3 * i] = p.x / det;
        colors[3 * i + 1] = p.y / det;
        colors[3 * i + 2] = p.z / det;

        colors[3 * i] *= ver_normal[3 * i];
        colors[3 * i + 1] *= ver_normal[3 * i + 1];
        colors[3 * i + 2] *= ver_normal[3 * i + 2];

        cos_sum = colors[3 * i] + colors[3 * i + 1] + colors[3 * i + 2];

        colors[3 * i] = clip(cos_sum * directional[0] + ambient[0], 0, 1);
        colors[3 * i + 1] = clip(cos_sum * directional[1] + ambient[1], 0, 1);
        colors[3 * i + 2] = clip(cos_sum * directional[2] + ambient[2], 0, 1);
    }

    for (int i = 0; i < ntri; ++i)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[tri_p0_ind];
        p0.y = vertices[tri_p0_ind + 1];
        p0.z = vertices[tri_p0_ind + 2];

        p1.x = vertices[tri_p1_ind];
        p1.y = vertices[tri_p1_ind + 1];
        p1.z = vertices[tri_p1_ind + 2];

        p2.x = vertices[tri_p2_ind];
        p2.y = vertices[tri_p2_ind + 1];
        p2.z = vertices[tri_p2_ind + 2];

        start.x = max(ceil(min(p0.x, min(p1.x, p2.x))), 0);
        end.x = min(floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        start.y = max(ceil(min(p0.y, min(p1.y, p2.y))), 0);
        end.y = min(floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if (end.x < start.x || end.y < start.y)
            continue;

        v0.x = p2.x - p0.x;
        v0.y = p2.y - p0.y;
        v1.x = p1.x - p0.x;
        v1.y = p1.y - p0.y;

        // dot products np.dot(v0.T, v0)
        dot00 = v0.x * v0.x + v0.y * v0.y;
        dot01 = v0.x * v1.x + v0.y * v1.y;
        dot11 = v1.x * v1.x + v1.y * v1.y;

        // barycentric coordinates
        start.z = dot00 * dot11 - dot01 * dot01;
        if (start.z != 0)
            start.z = 1 / start.z;

        for (p.y = start.y; p.y <= end.y; p.y += 1.0)
        {
            for (p.x = start.x; p.x <= end.x; p.x += 1.0)
            {
                v2.x = p.x - p0.x;
                v2.y = p.y - p0.y;

                dot02 = v0.x * v2.x + v0.y * v2.y;
                dot12 = v1.x * v2.x + v1.y * v2.y;

                v2.z = (dot11 * dot02 - dot01 * dot12) * start.z;
                v1.z = (dot00 * dot12 - dot01 * dot02) * start.z;
                v0.z = 1 - v2.z - v1.z;

                // judge is_point_in_tri by below line of code
                if (v2.z >= 0 && v1.z >= 0 && v0.z > 0)
                {
                    p.z = v0.z * p0.z + v1.z * p1.z + v2.z * p2.z;
                    color_index = p.y * w + p.x;

                    if (p.z > depth_buffer[color_index])
                    {
                        end.z = v0.z * colors[tri_p0_ind];
                        end.z += v1.z * colors[tri_p1_ind];
                        end.z += v2.z * colors[tri_p2_ind];
                        image[3 * color_index] = end.z * 255;

                        end.z = v0.z * colors[tri_p0_ind + 1];
                        end.z += v1.z * colors[tri_p1_ind + 1];
                        end.z += v2.z * colors[tri_p2_ind + 1];
                        image[3 * color_index + 1] = end.z * 255;

                        end.z = v0.z * colors[tri_p0_ind + 2];
                        end.z += v1.z * colors[tri_p1_ind + 2];
                        end.z += v2.z * colors[tri_p2_ind + 2];
                        image[3 * color_index + 2] = end.z * 255;

                        depth_buffer[color_index] = p.z;
                    }
                }
            }
        }
    }

    free(depth_buffer);
    free(colors);
    free(ver_normal);
}
