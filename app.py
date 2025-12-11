import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import simpson  # Using simpson for potentially better accuracy than trapezoid, or match notebook
import os

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Complex Fourier Transform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Analytic Continuation of the Fourier Transform")
st.markdown("""
This app visualizes the **magnitude, phase, and spectral deformation** of the Fourier Transform extended to the complex plane:
$$
F(z) = \\int_{-\\infty}^{\\infty} f(x) e^{-izx} \\, dx, \\quad z \\in \\mathbb{C}
$$
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")

# Function Selection (Extensible)
func_option = st.sidebar.selectbox(
    "Test Function f(x)",
    ("Gaussian: exp(-x^2)", "BoxCar: rect(x/2)", "Lorentzian: 1/(1+x^2)")
)

# Frequency Grid Range
st.sidebar.subheader("Complex Plane Domain")
k_min = st.sidebar.slider("Re(z) Min", -10.0, -1.0, -5.0)
k_max = st.sidebar.slider("Re(z) Max", 1.0, 10.0, 5.0)
Im_min = st.sidebar.slider("Im(z) Min", -5.0, -1.0, -3.0)
Im_max = st.sidebar.slider("Im(z) Max", 1.0, 5.0, 3.0)
resolution = st.sidebar.slider("Grid Resolution", 50, 300, 100)

# Integration limits
x_limit = st.sidebar.number_input("Integration Limit (±x)", value=20.0)
x_points = st.sidebar.number_input("Integration Points", value=2000)

# --- Core Logic ---
@st.cache_data
def get_f_x(func_name, x):
    if "Gaussian" in func_name:
        return np.exp(-x**2)
    elif "BoxCar" in func_name:
        return np.where(np.abs(x) <= 1, 1.0, 0.0)
    elif "Lorentzian" in func_name:
        return 1 / (1 + x**2)
    return np.zeros_like(x)

@st.cache_data
def compute_fourier_grid(func_name, x_lim, x_pts, k_range, im_range, res):
    # Definition of x domain
    x = np.linspace(-x_lim, x_lim, int(x_pts))
    
    # Definition of z domain
    re_vals = np.linspace(k_range[0], k_range[1], res)
    im_vals = np.linspace(im_range[0], im_range[1], res)
    RE, IM = np.meshgrid(re_vals, im_vals)
    Z = RE + 1j * IM
    
    # Get f(x)
    fx = get_f_x(func_name, x)
    
    # Compute F(z) - Vectorized broadcast is too memory heavy for simple method loop
    # We'll calculate it efficiently or use a loop for clarity/memory safety if grid is large
    # Let's try semi-vectorized: iterate over flattened Z? or just rows
    
    F_z = np.zeros_like(Z, dtype=complex)
    
    # To display progress
    progress_bar = st.progress(0)
    
    # Loop over rows (Im values) to save memory compared to full 3D broadcast
    for i in range(res):
        # z_row is (res,) complex array for this row
        z_row = Z[i, :] 
        # We need sum(f(x) * exp(-i * z * x) * dx)
        # Broadcasting: x is (N,), z_row is (M,)
        # exp term: exp(-i * z_row[:, None] * x[None, :]) -> (M, N) matrix
        
        # Optimization: -i * (RE + jIM) * x = (-i RE + IM) * x = IM*x - i*RE*x
        # exp(IM*x) * exp(-i*RE*x)
        
        # NOTE: Be careful with overflow for large Im(z) and large x.
        # But this is the point of the simulation (instability).
        
        kernel = np.exp(-1j * z_row[:, np.newaxis] * x[np.newaxis, :])
        integrand = fx[np.newaxis, :] * kernel
        
        # Trapezoidal rule over x (-1 axis)
        # Using np.trapz or just sum * dx
        dx = x[1] - x[0]
        row_integral = np.sum(integrand, axis=1) * dx
        
        F_z[i, :] = row_integral
        
        progress_bar.progress((i + 1) / res)
        
    return RE, IM, F_z

# Run Calculation
RE, IM, FZ = compute_fourier_grid(
    func_option, 
    x_limit, 
    x_points, 
    (k_min, k_max), 
    (Im_min, Im_max), 
    resolution
)

# --- Visualization ---

tab1, tab2, tab3 = st.tabs(["2D Heatmaps", "3D Surface", "Animations"])

with tab1:
    st.subheader("2D Heatmaps")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Magnitude |F(z)|**")
        fig_mag, ax_mag = plt.subplots()
        c_mag = ax_mag.pcolormesh(RE, IM, np.abs(FZ), shading='auto', cmap='inferno')
        ax_mag.set_xlabel("Re(z)")
        ax_mag.set_ylabel("Im(z)")
        fig_mag.colorbar(c_mag, ax=ax_mag)
        st.pyplot(fig_mag)
        
    with col2:
        st.markdown("**Phase arg(F(z))**")
        fig_arg, ax_arg = plt.subplots()
        c_arg = ax_arg.pcolormesh(RE, IM, np.angle(FZ), shading='auto', cmap='twilight')
        ax_arg.set_xlabel("Re(z)")
        ax_arg.set_ylabel("Im(z)")
        fig_arg.colorbar(c_arg, ax=ax_arg)
        st.pyplot(fig_arg)

with tab2:
    st.subheader("3D Surface Plot")
    
    # Plotly 3D Surface
    fig_3d = go.Figure(data=[go.Surface(
        z=np.abs(FZ), 
        x=RE, 
        y=IM,
        colorscale='Viridis'
    )])
    
    fig_3d.update_layout(
        title='Magnitude |F(z)|',
        scene=dict(
            xaxis_title='Re(z)',
            yaxis_title='Im(z)',
            zaxis_title='|F(z)|'
        ),
        autosize=False,
        width=800,
        height=600,
    )
    st.plotly_chart(fig_3d)

with tab3:
    st.subheader("Animations")
    st.markdown("These animations show spectral sweeps, demonstrating stability and analyticity.")
    
    anim_col1, anim_col2 = st.columns(2)
    
    with anim_col1:
        st.markdown("### Creating Real Sweep")
        video_path_real = "animations/real_sweep.mp4"
        if os.path.exists(video_path_real):
            st.video(video_path_real)
        else:
            st.error(f"Video not found: {video_path_real}")
            
    with anim_col2:
        st.markdown("### Creating Imaginary Sweep")
        video_path_imag = "animations/imaginary_sweep.mp4"
        if os.path.exists(video_path_imag):
            st.video(video_path_imag)
        else:
            st.error(f"Video not found: {video_path_imag}")

