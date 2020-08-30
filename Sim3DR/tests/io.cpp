#include "io.h"

//void load_obj(const string obj_fp, float* vertices, float* colors, float* triangles){
//    string line;
//    ifstream in(obj_fp);
//
//    if(in.is_open()){
//        while (getline(in, line)){
//            stringstream ss(line);
//
//            char t; // type: v, f
//            ss >> t;
//            if (t == 'v'){
//
//            }
//        }
//    }
//}

void load_obj(const char *obj_fp, float *vertices, float *colors, int *triangles, int nver, int ntri) {
    FILE *fp;
    fp = fopen(obj_fp, "r");

    char t; // type: v or f
    if (fp != nullptr) {
        for (int i = 0; i < nver; ++i) {
            fscanf(fp, "%c", &t);
            for (int j = 0; j < 3; ++j)
                fscanf(fp, " %f", &vertices[3 * i + j]);
            for (int j = 0; j < 3; ++j)
                fscanf(fp, " %f", &colors[3 * i + j]);
            fscanf(fp, "\n");
        }
//        fscanf(fp, "%c", &t);
        for (int i = 0; i < ntri; ++i) {
            fscanf(fp, "%c", &t);
            for (int j = 0; j < 3; ++j) {
                fscanf(fp, " %d", &triangles[3 * i + j]);
                triangles[3 * i + j] -= 1;
            }
            fscanf(fp, "\n");
        }

        fclose(fp);
    }
}

void load_ply(const char *ply_fp, float *vertices, int *triangles, int nver, int ntri) {
    FILE *fp;
    fp = fopen(ply_fp, "r");

//    char s[256];
    char t;
    if (fp != nullptr) {
//        for (int i = 0; i < 9; ++i)
//            fscanf(fp, "%s", s);
        for (int i = 0; i < nver; ++i)
            fscanf(fp, "%f %f %f\n", &vertices[3 * i], &vertices[3 * i + 1], &vertices[3 * i + 2]);

        for (int i = 0; i < ntri; ++i)
            fscanf(fp, "%c %d %d %d\n", &t, &triangles[3 * i], &triangles[3 * i + 1], &triangles[3 * i + 2]);

        fclose(fp);
    }
}

void write_ppm(const char *filename, unsigned char *img, int h, int w, int c) {
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //image size
    fprintf(fp, "%d %d\n", w, h);

    // rgb component depth
    fprintf(fp, "%d\n", MAX_PXL_VALUE);

    // pixel data
    fwrite(img, sizeof(unsigned char), size_t(h * w * c), fp);
    fclose(fp);
}