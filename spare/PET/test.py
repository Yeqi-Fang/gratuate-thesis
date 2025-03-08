import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate multiple radioactive ellipses
def generate_material_ellipses(num_ellipses=5, max_radius=50, center=(0, 0), num_points=1000):
    material_points = []
    for _ in range(num_ellipses):
        # Randomly generate the ellipse parameters
        radius_x = np.random.uniform(10, max_radius)
        radius_y = np.random.uniform(10, max_radius)
        ellipse_center_x = np.random.uniform(-max_radius, max_radius)
        ellipse_center_y = np.random.uniform(-max_radius, max_radius)
        
        for _ in range(num_points):
            # Generate points within the ellipse
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1)
            x = radius_x * np.sqrt(r) * np.cos(angle) + ellipse_center_x
            y = radius_y * np.sqrt(r) * np.sin(angle) + ellipse_center_y
            material_points.append((x, y))
    
    return np.array(material_points)

# Step 2: Generate the detector ring
def generate_detector_ring(num_detectors=360, radius=150):
    angles = np.linspace(0, 2 * np.pi, num_detectors, endpoint=False)
    detectors = np.array([(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles])
    return detectors

# Step 3: Simulate photon emission and detector reception
def simulate_detection(detectors, material_points, num_photons=1000):
    detector_responses = np.zeros(len(detectors))
    for _ in range(num_photons):
        # Randomly choose a point in the material to emit a photon
        point = material_points[np.random.randint(len(material_points))]
        
        # Simulate photon pair emission in opposite directions
        angle = np.random.uniform(0, 2 * np.pi)
        x1, y1 = point
        x2 = x1 + np.cos(angle) * 100  # Move in the direction of photon 1
        y2 = y1 + np.sin(angle) * 100
        
        # Check which detectors would receive the photons
        for i, (dx, dy) in enumerate(detectors):
            dist1 = np.sqrt((dx - x1)**2 + (dy - y1)**2)
            dist2 = np.sqrt((dx - x2)**2 + (dy - y2)**2)
            if dist1 < 10 or dist2 < 10:  # Photon hits the detector if close enough
                detector_responses[i] += 1
    return detector_responses

# Step 4: Form the image (projection from detector data)
def form_image(detector_responses, material_size=(300, 300), detector_ring_radius=150):
    image = np.zeros(material_size)
    for i, response in enumerate(detector_responses):
        angle = i * 2 * np.pi / len(detector_responses)
        x = int(material_size[0] / 2 + detector_ring_radius * np.cos(angle))
        y = int(material_size[1] / 2 + detector_ring_radius * np.sin(angle))
        x = min(max(x, 0), material_size[0] - 1)
        y = min(max(y, 0), material_size[1] - 1)
        image[x, y] = response
    return image

# Step 5: Reconstruct the image (Filtered Back Projection)
def reconstruct_image(projection, num_iterations=50):
    reconstructed_image = np.zeros_like(projection)
    for _ in range(num_iterations):
        reconstructed_image += projection
    return reconstructed_image

# Visualizing the results
def visualize(detectors, material_points, detector_responses, reconstructed_image):
    # Plot the material and detectors
    plt.figure(figsize=(10, 10))
    plt.scatter(material_points[:, 0], material_points[:, 1], c='red', s=1, alpha=0.5, label='Material')
    plt.scatter(detectors[:, 0], detectors[:, 1], c=detector_responses, cmap='hot', label='Detectors')
    plt.title('PET Simulation with Multiple Ellipses')
    plt.legend()
    plt.show()

    # Plot the reconstructed image
    plt.figure(figsize=(8, 8))
    plt.imshow(reconstructed_image, cmap='hot', interpolation='nearest')
    plt.title('Reconstructed Image')
    plt.colorbar()
    plt.show()

# Main function
def run_pet_simulation():
    # Parameters
    num_ellipses = 5  # Number of ellipses in the material
    num_detectors = 360  # Number of detectors
    num_photons = 10000  # Number of photon pairs emitted
    material_points = generate_material_ellipses(num_ellipses)
    detectors = generate_detector_ring(num_detectors)
    
    # Simulate detection
    detector_responses = simulate_detection(detectors, material_points, num_photons)
    
    # Form the projection image
    projection = form_image(detector_responses)
    
    # Reconstruct the image
    reconstructed_image = reconstruct_image(projection)
    
    # Visualize the results
    visualize(detectors, material_points, detector_responses, reconstructed_image)

# Run the simulation
run_pet_simulation()
