import numpy as np
from scipy.integrate import dblquad
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._compute_mesh()

    def _compute_mesh(self):
        # Parametric equations of a Möbius strip
        U, V = self.U, self.V
        R = self.R
        X = (R + V * np.cos(U / 2)) * np.cos(U)
        Y = (R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title("Möbius Strip")
        plt.show()

    def surface_area(self):
        # Surface area: numerical double integral over the surface
        def integrand(u, v):
            # Compute the magnitude of the cross product of partial derivatives
            du = 1e-5
            dv = 1e-5
            x1 = self._parametric_point(u, v)
            x2 = self._parametric_point(u + du, v)
            x3 = self._parametric_point(u, v + dv)
            du_vec = (x2 - x1) / du
            dv_vec = (x3 - x1) / dv
            return np.linalg.norm(np.cross(du_vec, dv_vec))

        area, _ = dblquad(integrand,
                          -self.w / 2, self.w / 2,
                          lambda v: 0,
                          lambda v: 2 * np.pi)
        return area

    def edge_length(self):
        # Approximate the edge length by summing distances along v = ±w/2
        edge_points = []
        for sign in [-1, 1]:
            v = sign * self.w / 2
            points = [self._parametric_point(u, v) for u in self.u]
            length = sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))
            edge_points.append(length)
        return sum(edge_points)

    def _parametric_point(self, u, v):
        R = self.R
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return np.array([x, y, z])

# Example usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.4, n=200)
    mobius.plot()
    print(f"Approximate Surface Area: {mobius.surface_area():.4f}")
    print(f"Approximate Edge Length: {mobius.edge_length():.4f}")