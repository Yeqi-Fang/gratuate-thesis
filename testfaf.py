import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

def generate_radioactive_distribution(nx=128, ny=128):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    
    circle = (xx**2 + yy**2 <= 0.3**2).astype(float) * 5.0
    xc, yc = 0.3, -0.2
    a, b = 0.2, 0.35
    ellipse = (
        (((xx - xc)/a)**2 + ((yy - yc)/b)**2) <= 1.0
    ).astype(float) * 3.0
    
    activity = circle + ellipse
    return activity

def generate_detectors(num_detectors=180, radius=50.0):
    detectors = []
    for i in range(num_detectors):
        angle = 2.0 * np.pi * i / num_detectors
        x_i = radius * np.cos(angle)
        y_i = radius * np.sin(angle)
        detectors.append((x_i, y_i))
    return np.array(detectors)

def find_detector_for_photon(x, y, theta, detectors, radius=50.0):
    dx, dy = np.cos(theta), np.sin(theta)
    A = dx*dx + dy*dy
    B = 2*(x*dx + y*dy)
    C = x*x + y*y - radius*radius
    disc = B*B - 4*A*C
    # pick positive root
    t = (-B + np.sqrt(disc)) / (2*A)
    x_int = x + t*dx
    y_int = y + t*dy
    dists = np.sqrt((detectors[:,0] - x_int)**2 + (detectors[:,1] - y_int)**2)
    det_id = np.argmin(dists)
    return det_id

def simulate_events(activity_map, detectors, num_events=20000, radius=50.0):
    nx, ny = activity_map.shape
    flat = activity_map.ravel()
    prob = flat / np.sum(flat)
    
    xx = np.linspace(-1, 1, nx)
    yy = np.linspace(-1, 1, ny)
    def idx_to_xy(idx):
        ix = idx % nx
        iy = idx // nx
        return xx[ix], yy[iy]
    
    emission_points = []
    detector_pairs = []
    
    for _ in range(num_events):
        idx = np.random.choice(nx*ny, p=prob)
        x_c, y_c = idx_to_xy(idx)
        theta = 2.0*np.pi*np.random.rand()
        d1 = find_detector_for_photon(x_c, y_c, theta, detectors, radius)
        d2 = find_detector_for_photon(x_c, y_c, theta+np.pi, detectors, radius)
        emission_points.append((x_c, y_c))
        detector_pairs.append((d1, d2))
    
    return emission_points, detector_pairs

def points_to_image(emission_points, nx=128, ny=128):
    image = np.zeros((ny, nx))
    x_bins = np.linspace(-1, 1, nx+1)
    y_bins = np.linspace(-1, 1, ny+1)
    xs = np.array([p[0] for p in emission_points])
    ys = np.array([p[1] for p in emission_points])
    hist2d, _, _ = np.histogram2d(ys, xs, bins=[y_bins, x_bins])
    return hist2d

# Main
num_detectors = 180
R = 50.0
detectors = generate_detectors(num_detectors, R)
activity_map = generate_radioactive_distribution(128, 128)

num_events = 20000
emission_points, detector_pairs = simulate_events(activity_map, detectors, num_events, R)

measured_image = points_to_image(emission_points, 128, 128)
theta_vals = np.linspace(0., 180., max(measured_image.shape), endpoint=False)
sinogram = radon(measured_image, theta=theta_vals, circle=True)
reconstruction_fbp = iradon(sinogram, theta=theta_vals, circle=True)

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.title("True Activity")
plt.imshow(activity_map, cmap='hot', origin='lower')
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Sinogram (Radon Transform)")
plt.imshow(sinogram, cmap='gray', aspect='auto',
           extent=(theta_vals.min(), theta_vals.max(), 0, sinogram.shape[0]))
plt.xlabel("Angle (deg)")
plt.ylabel("Projection Bin")

plt.subplot(1,3,3)
plt.title("Reconstruction (FBP)")
plt.imshow(reconstruction_fbp, cmap='hot', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.show()
