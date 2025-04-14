import numpy as np
import matplotlib.pyplot as plt

def create_image(size=(100, 100)):
    image = np.zeros(size)
    image[30:70, 30:70] = 255
    return image

def add_noise(image, noise_stdeviation = 25):
    noisy_image = image.copy()
    noise = np.random.randn(*image.shape) * noise_stdeviation
    noisy_image = np.clip(noisy_image + noise, 0, 255)
    # some shade of black and white
    return noisy_image

def gibbs_sampling(image, noise_stdeviation = 25, iterations = 50000):
    denoised_image = image.copy()
    rows, cols = denoised_image.shape
    prior_variance = noise_stdeviation ** 2
    likelihood_variance = 1


    for _ in range(iterations):
        i, j = np.random.randint(0, rows), np.random.randint(0, cols)

        neighbors = []
        if i > 0: neighbors.append(denoised_image[i - 1, j])
        if i < rows - 1: neighbors.append(denoised_image[i + 1, j])
        if j > 0: neighbors.append(denoised_image[i, j - 1])
        if j < cols - 1: neighbors.append(denoised_image[i, j + 1])

        mean_neighbors = 0
        if neighbors:
            mean_neighbors = np.mean(neighbors)

        # using bayes theorem we combine prior and likelihood to get posterior
        # p(x) = prior , p(y|x) is likelihood
        # posterior = p(x|y) - more details on this can be found on the solution description

        posterior_variance = 1 / (1/prior_variance + 1/likelihood_variance)
        posterior_mean = ((mean_neighbors * prior_variance + denoised_image[i, j] * likelihood_variance)
                          / ( prior_variance + likelihood_variance ))

        new_pixel_value = np.random.normal(posterior_mean, np.sqrt(posterior_variance))

        denoised_image[i, j] = np.clip(new_pixel_value, 0, 255)

    return denoised_image


original_image = create_image(size=(100, 100))
noisy_image = add_noise(original_image, noise_stdeviation=25)
denoised_image = gibbs_sampling(noisy_image, noise_stdeviation=25, iterations = 50000)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title("Noisy Image")
axes[1].axis('off')

axes[2].imshow(denoised_image, cmap='gray')
axes[2].set_title("Denoised Image (Gibbs Sampling)")
axes[2].axis('off')

plt.savefig("plot.png")
plt.close()
