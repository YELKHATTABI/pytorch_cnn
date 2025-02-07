import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def apply_convolution(image, kernel):
    """Apply a 2D convolution to the given grayscale image."""
    return cv2.filter2D(image, -1, kernel)


def visualize_kernel(kernel):
    """Visualize the kernel with numbers."""
    fig, ax = plt.subplots()
    ax.matshow(kernel, cmap="viridis")

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax.text(j, i, f"{kernel[i, j]:.2f}", va="center", ha="center", color="white")

    plt.title("Kernel Visualization")
    plt.axis("off")
    st.pyplot(fig)


def main():
    st.title("2D Convolution Visualizer")

    # Upload an image
    uploaded_file = st.file_uploader("Upload a black-and-white image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image as grayscale
        image = Image.open(uploaded_file).convert("L")
        image = np.array(image)

        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_column_width=True, clamp=True)

        # Define default filters
        filters = {
            "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "Blur": np.ones((3, 3), dtype=np.float32) / 9,
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        }

        # Filter selection
        filter_name = st.selectbox("Choose a filter", list(filters.keys()))
        kernel = filters[filter_name]

        # Allow custom kernel input
        if st.checkbox("Customize Kernel"):
            kernel_text = st.text_area(
                "Enter a custom kernel (as a numpy list, e.g., [[1, 0, -1], [1, 0, -1], [1, 0, -1]])"
            )
            if kernel_text:
                try:
                    kernel = np.array(eval(kernel_text))
                    if kernel.ndim != 2:
                        st.error("Kernel must be a 2D array.")
                        return
                except Exception as e:
                    st.error(f"Invalid kernel format: {e}")
                    return

        st.subheader("Selected Kernel")
        visualize_kernel(kernel)

        # Apply convolution
        convolved_image = apply_convolution(image, kernel)

        # Display results
        st.subheader("Convolved Image")
        st.image(convolved_image, caption="Convolved Image", use_column_width=True, clamp=True)


if __name__ == "__main__":
    main()
