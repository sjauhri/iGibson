# Functionality to convert from world to screen / raster space and back
# Author: Daljeet Nandha

import numpy as np


class GL_Transform():
    def __init__(self):

        self.H, self.W = 512, 512

        ## camera matrix (normally not needed)
        #self.K = [[256.,  0., 256.],
        # [  0., 256., 256.],
        # [  0.,  0.,   1.]]

        # projection matrix: camera -> screen
        self.P = [[ 1.,        0.,        0.,        0.      ],
         [ 0.,        1.,        0.,        0.      ],
         [ 0.,        0.,       -1.0002,   -1.      ],
         [ 0.,        0.,       -0.020002,  0.      ]]

        # view matrix: world -> camera
        self.V = [[ 0.17633685,  0.9843299,  -0.,         -0.09054275],
         [-0.72555274,  0.12997846,  0.6757802,  -0.41956753],
         [ 0.66519064, -0.11916495,  0.7371032,  -1.4083172 ],
         [ 0.        ,  0.,          0.,          1.        ]]
        #self.V = [[ 0.,  1., -0., -0.],
        # [ 0.,  0.,  1., -0.],
        # [ 1., -0., -0., -1.],
        # [ 0.,  0.,  0.,  1.]]

        # model matrix: identity
        self.M = np.eye(4)

        #self.K = np.asarray(self.K)
        self.P = np.asarray(self.P)
        self.V = np.asarray(self.V)
        self.M = np.asarray(self.M)

        # lower left corner x, y, width, height
        self.viewport = [0, self.H, self.W, self.H]

    # model view projection matrix
    def mvp(self):
        return self.P @ self.V @ self.M

    def inverse_mvp(self):
        return np.linalg.inv(self.P @ self.V @ self.M)

    def test_mvp(self):
        return self.mvp() @ self.inverse_mvp()  # must be identity

    def to_homo(self, vec):
        return np.concatenate([vec, np.array([1])])

    def to_coords(self, vec):
        coords = vec / vec[-1]
        return coords[:3]

    # project points from object space to window space
    # see: gluProject
    def project(self, vec):
        v1 = self.M @ vec
        v2 = self.V @ v1
        v_ = self.P @ v2

        v_[-1] = -v2[2]
        v_ /= v_[-1]

        (u, v) = self.viewport_transform(v_)
        d = 1/2 * (v_[2] + 1)

        return (u, v), d

    # unproject points from window space to object space
    # see: glUnProject
    def unproject(self, uv, d):
        (x, y) = self.inverse_viewport_transform(uv)
        z = 2 * d - 1

        X = self.inverse_mvp() @ np.array([x, y, z, 1])
        X /= X[-1]

        return X

    # from ndc to window cords
    # see: glViewport
    def viewport_transform(self, normalized_cords):
        window_cords_x = self.viewport[0] + 1/2 * self.viewport[2] * (normalized_cords[0] + 1)
        window_cords_y = self.viewport[1] + 1/2 * self.viewport[3] * (normalized_cords[1] + 1)

        return (window_cords_x, window_cords_y)

    # from window cords to ndc
    # see: glViewport
    def inverse_viewport_transform(self, window_cords):
        normalized_cords_x = 2 * (window_cords[0] - self.viewport[0]) / self.viewport[2] - 1
        normalized_cords_y = 2 * (window_cords[1] - self.viewport[1]) / self.viewport[3] - 1

        return (normalized_cords_x, normalized_cords_y)

    def world_to_cam(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(self.to_homo(vec))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v / v[-1]
        else:
            return None

    def cam_to_world(self, vec):
        coords = self.to_coords(np.linalg.inv(self.V) @ vec)
        return coords


if __name__ == "__main__":
    u, v = 202, 212
    depth = 2.235776901245117

    Xest = np.array([0.3, -0.1, 0.8])


    bla = GL_Transform()
    x = bla.world_to_cam(Xest)
    print(x)
