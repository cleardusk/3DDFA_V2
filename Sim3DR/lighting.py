# coding: utf-8

import numpy as np
from .Sim3DR import get_normal, rasterize

_norm = lambda arr: arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices


def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj


class RenderPipeline(object):
    def __init__(self, **kwargs):
        self.intensity_ambient = convert_type(kwargs.get('intensity_ambient', 0.3))
        self.intensity_directional = convert_type(kwargs.get('intensity_directional', 0.6))
        self.intensity_specular = convert_type(kwargs.get('intensity_specular', 0.1))
        self.specular_exp = kwargs.get('specular_exp', 5)
        self.color_ambient = convert_type(kwargs.get('color_ambient', (1, 1, 1)))
        self.color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
        self.light_pos = convert_type(kwargs.get('light_pos', (0, 0, 5)))
        self.view_pos = convert_type(kwargs.get('view_pos', (0, 0, 5)))

    def update_light_pos(self, light_pos):
        self.light_pos = convert_type(light_pos)

    def __call__(self, vertices, triangles, bg, texture=None):
        normal = get_normal(vertices, triangles)

        # 2. lighting
        light = np.zeros_like(vertices, dtype=np.float32)
        # ambient component
        if self.intensity_ambient > 0:
            light += self.intensity_ambient * self.color_ambient

        vertices_n = norm_vertices(vertices.copy())
        if self.intensity_directional > 0:
            # diffuse component
            direction = _norm(self.light_pos - vertices_n)
            cos = np.sum(normal * direction, axis=1)[:, None]
            # cos = np.clip(cos, 0, 1)
            #  todo: check below
            light += self.intensity_directional * (self.color_directional * np.clip(cos, 0, 1))

            # specular component
            if self.intensity_specular > 0:
                v2v = _norm(self.view_pos - vertices_n)
                reflection = 2 * cos * normal - direction
                spe = np.sum((v2v * reflection) ** self.specular_exp, axis=1)[:, None]
                spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
                light += self.intensity_specular * self.color_directional * np.clip(spe, 0, 1)
        light = np.clip(light, 0, 1)

        # 2. rasterization, [0, 1]
        if texture is None:
            render_img = rasterize(vertices, triangles, light, bg=bg)
            return render_img
        else:
            texture *= light
            render_img = rasterize(vertices, triangles, texture, bg=bg)
            return render_img


def main():
    pass


if __name__ == '__main__':
    main()
