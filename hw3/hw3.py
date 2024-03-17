from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# 5.1
def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    mean = np.mean(x, axis = 0)
    centered_dataset = x - mean

    return centered_dataset.astype(float)

# 5.2
def get_covariance(dataset):
    # Your implementation goes here!
    # n = dataset.shape[1]
    # print(n) debug line
    # x = (1/(n-1)) * np.dot(dataset, np.transpose(dataset))
    # x = (1/(n-1)) * np.dot(np.transpose(dataset), dataset)
    x = np.cov(dataset, rowvar=False)
    return x

# 5.3
# Adapted argsort from ChatGPT, I figure it out how the rest works through debugging and printing values many times
def get_eig(S, m):
    # Perform eigendecomposition
    eigenvalues, eigenvectors = eigh(S)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    #print(eigenvalues)

    # Select the top m eigenvalues and corresponding eigenvectors
    top_eigenvalues = eigenvalues[:m]
    top_eigenvectors = eigenvectors[:, :m]

    # Create a diagonal matrix with the top eigenvalues
    diag_matrix = np.diag(top_eigenvalues)

    return diag_matrix, top_eigenvectors

# 5.4
# Adapted argsort from ChatGPT, I figure it out how the rest works through debugging and printing values many times
def get_eig_prop(S, prop):
    # Perform eigendecomposition
    eigenvalues, eigenvectors = eigh(S)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the total variance
    total_variance = np.sum(eigenvalues)
    #print("total_variance value:")
    #print(total_variance)

    # Calculate the cumulative explained variance
    cumulative_variance = eigenvalues / total_variance
    #print("cumulative_variance value:")
    #print(cumulative_variance)

    # Determine how many eigenvalues to keep
    num_eigenvalues_to_keep = np.sum(cumulative_variance >= prop)
    #print("num_eigenvalues_to_keep:")
    #print(num_eigenvalues_to_keep)
    # Select the eigenvalues and eigenvectors that explain more than prop of the variance

    top_eigenvalues = eigenvalues[:num_eigenvalues_to_keep]
    top_eigenvectors = eigenvectors[:, :num_eigenvalues_to_keep]

    # Create a diagonal matrix with the top eigenvalues
    diag_matrix = np.diag(top_eigenvalues)

    return diag_matrix, top_eigenvectors

# 5.5
def project_image(image, U):
    # Your implementation goes here!
    # Expanding dimensions because vector is (1024, ), needs to be (1024, 1)
    # Adapted from https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    image = np.expand_dims(image, axis=1)

    # Compute the projection coefficients (alpha)
    alpha = np.dot(U.T, image)

    # Reconstruct the image in the reduced-dimensional space
    reconstructed_image = np.dot(U, alpha)

    return reconstructed_image.flatten()

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    orig = np.expand_dims(orig, axis=1)
    proj = np.expand_dims(proj, axis=1)
    orig = orig.reshape(32, 32)
    proj = proj.reshape(32, 32)


    # Rotate image 90 degrees to the right
    # Adapted from https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
    orig = np.rot90(orig, k=-1)
    proj = np.rot90(proj, k=-1)

    # Set titles for the subplots
    ax1.set_title('Original')
    ax2.set_title('Projection')

    # Create colorbars for each image
    fig.colorbar(ax1.imshow(orig, aspect='auto'), ax=ax1)
    fig.colorbar(ax2.imshow(proj, aspect='auto'), ax=ax2)

    return fig, ax1, ax2

# many debug and testing/printing lines below

#print("5.1 test")
#dataset = load_and_center_dataset('YaleB_32x32.npy')
#print(dataset.shape)

#print("\n5.2 test")
#S = get_covariance(dataset)
#print(S.shape)


x = load_and_center_dataset('YaleB_32x32.npy')
#print(x.shape)
#print(len(x[0]))
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
#Lambda2, U2 = get_eig_prop(S, 0.07)
#print("Lambda2 value:")
#print(Lambda2)
#print("U2 value:")
#print(U)
#print("Lambda values:")
#print(Lambda)
#print("----------------")
#print(U)
#print(len(U))
#print(len(U[0]))
#print("U values:")
#print(U)
#print[U.shape]
#print("x[0] values")
##print(x[0])
projection = project_image(x[0], U)
#print("projection value:")
#print(projection)
#print(projection.shape)
fig, ax1, ax2 = display_image(x[0], projection)
#print("fig value:")
#print(fig)
plt.show()