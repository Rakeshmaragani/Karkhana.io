import numpy as np
from scipy.integrate import dblquad
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    """
    Class to model a Möbius strip using parametric equations,
    compute 3D mesh, surface area, and edge length.
    """
    def __init__(self, R=1.0, w=0.2, n=100):
        """
        Initialize the Möbius strip parameters.

        Parameters:
        R : float
            Radius of the circular center line of the strip
        w : float
            Width of the strip
        n : int
            Resolution for mesh sampling
        """
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._compute_mesh()

    def _compute_mesh(self):
        """
        Generate the 3D mesh grid for the Möbius strip.
        """
        U, V = self.U, self.V
        R = self.R
        X = (R + V * np.cos(U / 2)) * np.cos(U)
        Y = (R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def plot(self):
        """
        Plot the Möbius strip in 3D using matplotlib.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', alpha=0.9)
        ax.set_title("Möbius Strip")
        plt.tight_layout()
        plt.show()

    def surface_area(self):
        """
        Approximate the surface area using numerical integration.
        """
        def integrand(v, u):
            # Finite difference approximation of partial derivatives
            du = 1e-5
            dv = 1e-5
            p = self._parametric_point(u, v)
            pu = self._parametric_point(u + du, v)
            pv = self._parametric_point(u, v + dv)
            tangent_u = (pu - p) / du
            tangent_v = (pv - p) / dv
            return np.linalg.norm(np.cross(tangent_u, tangent_v))

        area, _ = dblquad(integrand,
                          0, 2 * np.pi,
                          lambda u: -self.w / 2,
                          lambda u: self.w / 2)
        return area

    def edge_length(self):
        """
        Approximate the total edge length of the strip (two boundaries).
        """
        edge_length = 0
        for sign in [-1, 1]:  # edges at v = ±w/2
            v = sign * self.w / 2
            points = [self._parametric_point(u, v) for u in self.u]
            edge_length += sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))
        return edge_length

    def _parametric_point(self, u, v):
        """
        Compute a single point on the Möbius strip from parameters (u, v).
        """
        R = self.R
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return np.array([x, y, z])

# Example usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.4, n=200)
    mobius.plot()
    print(f"Approximate Surface Area: {mobius.surface_area():.5f}")
    print(f"Approximate Edge Length: {mobius.edge_length():.5f}")