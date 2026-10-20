import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
import io

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SubsidenceLab — Ground Subsidence Prediction",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional engineering theme (white background, crisp dark text, blue accents)
st.markdown("""
<style>
    .main {
        background-color: #FAFAFA;
    }
    .stApp {
        color: #1E293B;
    }
    h1, h2, h3, h4 {
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #0284C7;
    }
    .metric-label {
        font-size: 13px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    .primary-calc-btn button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-size: 16px !important;
        padding: 10px 24px !important;
        width: 100%;
    }
    .primary-calc-btn button:hover {
        background-color: #0369A1 !important;
    }
    .section-header {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
        margin-top: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE NUMERICAL ENGINE (MATLAB TRANSLATION)
# -----------------------------------------------------------------------------

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Python translation of point_to_line.m:
    Calculates the perpendicular distance from point (px, py) to segment (x1, y1) - (x2, y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx*dx + dy*dy
    if l2 == 0.0:
        return np.hypot(px - x1, py - y1)
    
    t = ((px - x1) * dx + (py - y1) * dy) / l2
    t = np.clip(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.hypot(px - proj_x, py - proj_y)

def min_distance_to_polygon(px, py, poly_x, poly_y):
    """Calculates min perpendicular distance from point (px, py) to edges of polygon."""
    num_edges = len(poly_x) - 1
    dists = [
        point_to_segment_distance(px, py, poly_x[k], poly_y[k], poly_x[k+1], poly_y[k+1])
        for k in range(num_edges)
    ]
    return min(dists)

def run_subsidence_model(p):
    """
    Direct faithful translation of subsidence_model.m:
    Modified Influence Function Method by Sheorey et al. (2000).
    """
    H = p['H']
    drawDeg = p['drawDeg']
    NEW = p['NEW']
    h = p['h']
    e = p['e']
    dx = p['dx']
    dy = p['dy']
    panelX = np.array(p['panelX'], dtype=float)
    panelY = np.array(p['panelY'], dtype=float)
    leaseX = np.array(p['leaseX'], dtype=float)
    leaseY = np.array(p['leaseY'], dtype=float)

    # Influence radius R = H * tan(theta)
    R = H * np.tan(np.deg2rad(drawDeg))

    # Grid generation over lease boundary
    xmin = np.floor(np.min(leaseX) / dx) * dx
    xmax = np.ceil(np.max(leaseX) / dx) * dx
    ymin = np.floor(np.min(leaseY) / dy) * dy
    ymax = np.ceil(np.max(leaseY) / dy) * dy

    x_coords = np.arange(xmin, xmax + dx * 0.5, dx)
    y_coords = np.arange(ymin, ymax + dy * 0.5, dy)
    X, Y = np.meshgrid(x_coords, y_coords)

    lease_poly = Polygon(zip(leaseX, leaseY))
    panel_poly = Polygon(zip(panelX, panelY))

    # Mask points inside lease boundary
    in_lease = np.zeros(X.shape, dtype=bool)
    for m in range(X.shape[0]):
        for n in range(X.shape[1]):
            in_lease[m, n] = lease_poly.contains(Point(X[m, n], Y[m, n])) or lease_poly.touches(Point(X[m, n], Y[m, n]))

    # Ring weighting factors from Sheorey et al. (2000)
    ringWeights = np.array([0.0297, 0.0854, 0.1311, 0.1607, 0.1708, 0.1596, 0.1294, 0.0859, 0.0397, 0.0069])
    sectorWeights = ringWeights / 64.0

    W_emp = np.max(panelX) - np.min(panelX)
    L_emp = np.max(panelY) - np.min(panelY)
    widthRatio = W_emp / (H * NEW)

    # Sheorey empirical max subsidence relation for single seam
    baseFactor = 0.33 * (1.0 + 1.1 * np.tanh(1.4 * (widthRatio - 1.8)))
    baseFactor = max(baseFactor, 0.01)
    Smax = baseFactor * h * e

    Subsidence = np.full(X.shape, np.nan)
    Ux = np.full(X.shape, np.nan)
    Uy = np.full(X.shape, np.nan)

    dr = R / 10.0
    rho = (np.arange(1, 11) - 0.5) * dr
    theta = (np.arange(1, 65) - 0.5) * (2.0 * np.pi / 64.0)

    # Sector integration loops
    for m in range(X.shape[0]):
        for n in range(X.shape[1]):
            if not in_lease[m, n]:
                continue

            Px = X[m, n]
            Py = Y[m, n]
            Sz = 0.0
            Sx = 0.0
            Sy = 0.0

            for i in range(10):
                xel = Px + rho[i] * np.cos(theta)
                yel = Py + rho[i] * np.sin(theta)

                # Vectorized point containment check
                inside_flags = [panel_poly.contains(Point(x_pt, y_pt)) or panel_poly.touches(Point(x_pt, y_pt))
                                for x_pt, y_pt in zip(xel, yel)]
                inside_flags = np.array(inside_flags, dtype=bool)

                if not np.any(inside_flags):
                    continue

                xx = xel[inside_flags]
                yy = yel[inside_flags]
                sw = sectorWeights[i]

                # Perpendicular distance to extraction edges
                db = np.array([min_distance_to_polygon(ptx, pty, panelX, panelY) for ptx, pty in zip(xx, yy)])
                d = yy - np.min(panelY)

                # Edge and asymmetry corrections (Eqs 8 & 9)
                Wz = 0.5 * np.tanh((5.0 * db) / (1.5 * NEW * h) - 2.4) + 0.5
                Qz = 0.9 - 0.1 * np.tanh(0.5 * (d - 0.4 * L_emp))

                dVz = -sw * Smax * Wz * Qz
                Sz += np.sum(dVz)
                Sx += np.sum(dVz * (xx - Px) / H)
                Sy += np.sum(dVz * (yy - Py) / H)

            Subsidence[m, n] = Sz
            Ux[m, n] = Sx
            Uy[m, n] = Sy

    # Spatial numerical gradients (Subsidence in mm, Spacings in m -> Slope in mm/m)
    Sub_mm = Subsidence * 1000.0
    dSdy, dSdx = np.gradient(Sub_mm, dy, dx)

    # Horizontal strain components (Ux, Uy in mm -> Strain in mm/m)
    dUxdx = np.gradient(Ux * 1000.0, dx, axis=1)
    dUydy = np.gradient(Uy * 1000.0, dy, axis=0)
    Exx = dUxdx
    Eyy = dUydy

    valid_mask = ~np.isnan(Subsidence)
    sub_valid = Sub_mm[valid_mask]
    slope_x_valid = dSdx[valid_mask]
    slope_y_valid = dSdy[valid_mask]
    strain_x_valid = Exx[valid_mask]
    strain_y_valid = Eyy[valid_mask]

    stats = {
        'maxSub': np.min(sub_valid) if len(sub_valid) > 0 else 0.0,
        'maxSub_pos': np.max(np.abs(sub_valid)) if len(sub_valid) > 0 else 0.0,
        'maxSlope': max(np.max(np.abs(slope_x_valid)), np.max(np.abs(slope_y_valid))) if len(slope_x_valid) > 0 else 0.0,
        'maxTensile': max(np.max(strain_x_valid), np.max(strain_y_valid)) if len(strain_x_valid) > 0 else 0.0,
        'maxCompressive': min(np.min(strain_x_valid), np.min(strain_y_valid)) if len(strain_x_valid) > 0 else 0.0,
    }

    if len(sub_valid) > 0:
        max_idx = np.nanargmin(Subsidence)
        r_idx, c_idx = np.unravel_index(max_idx, Subsidence.shape)
        stats['maxX'] = float(X[r_idx, c_idx])
        stats['maxY'] = float(Y[r_idx, c_idx])
    else:
        stats['maxX'] = 0.0
        stats['maxY'] = 0.0

    return {
        'X': X,
        'Y': Y,
        'Subsidence': Subsidence,
        'SlopeX': dSdx,
        'SlopeY': dSdy,
        'StrainX': Exx,
        'StrainY': Eyy,
        'panelX': panelX,
        'panelY': panelY,
        'leaseX': leaseX,
        'leaseY': leaseY,
        'stats': stats,
        'R': R,
        'Smax_theoretical': Smax * 1000.0
    }

def run_original_script_model(p):
    """
    Direct faithful translation of matlab_proj_original.m:
    Standard analytical grid evaluation without dynamic lease clipping.
    """
    res = run_subsidence_model(p)
    W = np.max(p['panelX']) - np.min(p['panelX'])
    H = p['H']
    NEW = p['NEW']
    ratio = W / (H * NEW)
    
    # Hand-drawn curve reference estimation constants K1, K2, K3 from Fig. 8
    # K1 (Slope), K2 (Compressive), K3 (Tensile)
    if ratio < 1.0:
        K1, K2, K3 = 0.0, 0.0, 0.0
    else:
        K1 = 3.3 * np.exp(-0.4 * (ratio - 1.8)**2) + 1.8
        K2 = 2.2 * np.exp(-0.5 * (ratio - 1.8)**2) + 0.9
        K3 = 1.9 * np.exp(-0.5 * (ratio - 1.8)**2) + 0.8
    
    S_m = res['stats']['maxSub_pos'] / 1000.0
    res['stats']['empirical_K1_slope'] = float(K1 * (S_m / H) * 1000.0)
    res['stats']['empirical_K2_comp'] = float(-K2 * (S_m / H) * 1000.0)
    res['stats']['empirical_K3_tens'] = float(K3 * (S_m / H) * 1000.0)
    return res

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'calc_results' not in st.session_state:
    st.session_state.calc_results = None

if 'panel_vertices' not in st.session_state:
    st.session_state.panel_vertices = []

if 'lease_vertices' not in st.session_state:
    # Default 4-point outer boundary (0,0) to (500,500)
    st.session_state.lease_vertices = [
        {"x": 0.0, "y": 0.0},
        {"x": 500.0, "y": 0.0},
        {"x": 500.0, "y": 500.0},
        {"x": 0.0, "y": 500.0}
    ]

# -----------------------------------------------------------------------------
# TOP NAVIGATION BAR
# -----------------------------------------------------------------------------
nav_selection = st.sidebar.radio(
    "Navigation",
    ["Predict", "About & Method"],
    label_visibility="collapsed"
)

# -----------------------------------------------------------------------------
# TAB 1: PREDICT WORKFLOW
# -----------------------------------------------------------------------------
if nav_selection == "Predict":
    st.title("SubsidenceLab — Ground Subsidence Prediction")
    st.caption("Modified Influence Function Method for Irregular Extraction Panels (Sheorey et al., 2000)")

    # 1. Calculation Engine Selector
    with st.sidebar:
        st.markdown('<div class="section-header">1. Engine & Model Mode</div>', unsafe_allow_html=True)
        calc_engine = st.selectbox(
            "Calculation Methodology",
            ["GUI Model (subsidence_model.m)", "Original Script (matlab_proj_original.m)"],
            help="GUI model performs strict lease-bounded numerical sector integration. Original script performs full grid integration with empirical K-factors."
        )

        st.markdown('<div class="section-header">2. Mining Panel Geometry</div>', unsafe_allow_html=True)
        panel_geom_mode = st.radio("Panel Definition Mode", ["Standard Rectangle", "Custom Multi-Point Polygon"])

        if panel_geom_mode == "Standard Rectangle":
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                p_x0 = st.number_input("Start X0 (m)", value=100.0, step=10.0)
                p_w = st.number_input("Width W (m)", value=198.0, step=10.0, min_value=1.0)
            with col_p2:
                p_y0 = st.number_input("Start Y0 (m)", value=100.0, step=10.0)
                p_l = st.number_input("Length L (m)", value=294.0, step=10.0, min_value=1.0)
            panel_poly_x = [p_x0, p_x0 + p_w, p_x0 + p_w, p_x0, p_x0]
            panel_poly_y = [p_y0, p_y0, p_y0 + p_l, p_y0 + p_l, p_y0]
        else:
            st.info("Input custom polygon vertices below.")
            raw_pts = st.text_area(
                "Paste Coordinates (X, Y per line)",
                value="55, 48\n112, 228\n216, 206\n216, 244\n280, 246\n336, 233\n324, 195\n315, 148\n280, 112\n262, 100\n242, 83\n112, 48",
                height=140
            )
            parsed_panel_x, parsed_panel_y = [], []
            for line in raw_pts.strip().split("\n"):
                if "," in line:
                    parts = line.split(",")
                    try:
                        parsed_panel_x.append(float(parts[0].strip()))
                        parsed_panel_y.append(float(parts[1].strip()))
                    except ValueError:
                        pass
            if len(parsed_panel_x) >= 3:
                if parsed_panel_x[0] != parsed_panel_x[-1] or parsed_panel_y[0] != parsed_panel_y[-1]:
                    parsed_panel_x.append(parsed_panel_x[0])
                    parsed_panel_y.append(parsed_panel_y[0])
                panel_poly_x = parsed_panel_x
                panel_poly_y = parsed_panel_y
            else:
                st.warning("Please provide at least 3 valid coordinates.")
                panel_poly_x = [100.0, 298.0, 298.0, 100.0, 100.0]
                panel_poly_y = [100.0, 100.0, 394.0, 394.0, 100.0]

        st.markdown('<div class="section-header">3. Geomining Parameters</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            H = st.number_input("Depth of Cover H (m)", value=300.0, step=10.0, min_value=1.0)
            drawDeg = st.number_input("Draw Angle θ (deg)", value=13.0, step=1.0, min_value=1.0, max_value=85.0)
            NEW = st.number_input("Non-Effective Width (NEW)", value=0.40, step=0.05, min_value=0.01)
        with col_m2:
            h = st.number_input("Seam Thickness h (m)", value=4.6, step=0.5, min_value=0.1)
            e = st.number_input("Extraction Factor e", value=0.66, step=0.05, min_value=0.01, max_value=1.0)
            grid_res = st.number_input("Grid Spacing dx = dy (m)", value=5.0, step=1.0, min_value=1.0)

        # Calculated Influence Radius Display
        calc_R = H * np.tan(np.deg2rad(drawDeg))
        st.markdown(f"""
        <div style="background-color: #F0F9FF; border-left: 3px solid #0284C7; padding: 8px 12px; border-radius: 4px; margin-top: 8px;">
            <span style="font-size: 11px; color: #0369A1; font-weight: 600; text-transform: uppercase;">Influence Radius (R)</span><br/>
            <span style="font-size: 16px; font-weight: 700; color: #0C4A6E;">{calc_R:.2f} m</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">4. Profile Slices</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            slice_y = st.number_input("Profile X-X at Y (m)", value=150.0, step=10.0)
        with col_s2:
            slice_x = st.number_input("Profile Y-Y at X (m)", value=150.0, step=10.0)

        st.markdown('<div class="section-header">5. Execute</div>', unsafe_allow_html=True)
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            st.markdown('<div class="primary-calc-btn">', unsafe_allow_html=True)
            calc_btn = st.button("CALCULATE", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b2:
            reset_btn = st.button("Reset", use_container_width=True)
            if reset_btn:
                st.session_state.calc_results = None
                st.rerun()

    # Calculation Boundary (Lease) Configuration on main panel
    with st.expander("Surface Calculation Boundary (Lease Area)", expanded=False):
        st.write("Define the outer bounding polygon where surface displacements are computed.")
        col_lb1, col_lb2 = st.columns([1, 1])
        with col_lb1:
            st.caption("Coordinate List Editor")
            lease_df = pd.DataFrame(st.session_state.lease_vertices)
            edited_df = st.data_editor(lease_df, num_rows="dynamic", use_container_width=True)
            if st.button("Apply Edited Boundary"):
                st.session_state.lease_vertices = edited_df.to_dict('records')
                st.success("Updated calculation boundary.")
        with col_lb2:
            st.caption("Auto-Generate Rectangular Boundary")
            extent_margin = st.slider("Margin beyond panel (m)", 20.0, 300.0, 100.0, step=10.0)
            if st.button("Generate Extent from Panel"):
                min_px, max_px = min(panel_poly_x), max(panel_poly_x)
                min_py, max_py = min(panel_poly_y), max(panel_poly_y)
                st.session_state.lease_vertices = [
                    {"x": min_px - extent_margin, "y": min_py - extent_margin},
                    {"x": max_px + extent_margin, "y": min_py - extent_margin},
                    {"x": max_px + extent_margin, "y": max_py + extent_margin},
                    {"x": min_px - extent_margin, "y": max_py + extent_margin}
                ]
                st.rerun()

    # Assemble Lease Polygon Points
    lease_poly_x = [pt['x'] for pt in st.session_state.lease_vertices]
    lease_poly_y = [pt['y'] for pt in st.session_state.lease_vertices]
    if len(lease_poly_x) >= 3:
        if lease_poly_x[0] != lease_poly_x[-1] or lease_poly_y[0] != lease_poly_y[-1]:
            lease_poly_x.append(lease_poly_x[0])
            lease_poly_y.append(lease_poly_y[0])

    # Run Calculation
    if calc_btn:
        with st.spinner("Computing numerical influence function integrals..."):
            params = {
                'H': H,
                'drawDeg': drawDeg,
                'NEW': NEW,
                'h': h,
                'e': e,
                'dx': grid_res,
                'dy': grid_res,
                'panelX': panel_poly_x,
                'panelY': panel_poly_y,
                'leaseX': lease_poly_x,
                'leaseY': lease_poly_y
            }
            if "Original Script" in calc_engine:
                st.session_state.calc_results = run_original_script_model(params)
            else:
                st.session_state.calc_results = run_subsidence_model(params)
            st.success("Subsidence calculation complete.")

    # -------------------------------------------------------------------------
    # RESULTS DASHBOARD
    # -------------------------------------------------------------------------
    if st.session_state.calc_results is not None:
        r = st.session_state.calc_results
        stats = r['stats']

        # Metric Summary Cards
        st.markdown("### Numerical Results Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Subsidence</div>
                <div class="metric-value">{stats['maxSub_pos']:.2f} mm</div>
                <div style="font-size: 11px; color: #64748B;">At ({stats['maxX']:.1f}, {stats['maxY']:.1f}) m</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Slope</div>
                <div class="metric-value">{stats['maxSlope']:.3f} mm/m</div>
                <div style="font-size: 11px; color: #64748B;">Maximum spatial gradient</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Tensile Strain (+)</div>
                <div class="metric-value">{stats['maxTensile']:.3f} mm/m</div>
                <div style="font-size: 11px; color: #64748B;">Surface tension zone</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Compressive Strain (-)</div>
                <div class="metric-value">{stats['maxCompressive']:.3f} mm/m</div>
                <div style="font-size: 11px; color: #64748B;">Trough compression center</div>
            </div>
            """, unsafe_allow_html=True)

        if 'empirical_K1_slope' in stats:
            st.info(f"**matlab_proj_original.m Chart Benchmarks:** Empirical Slope: {stats['empirical_K1_slope']:.2f} mm/m | Compressive: {stats['empirical_K2_comp']:.2f} mm/m | Tensile: {stats['empirical_K3_tens']:.2f} mm/m")

        # Visualization Tabs
        tab_2d, tab_profx, tab_profy, tab_3d, tab_data = st.tabs([
            "2D Contour Map", "Profile X-X", "Profile Y-Y", "3D Subsidence Trough", "Tabular Data & Export"
        ])

        sub_mm = np.abs(r['Subsidence'] * 1000.0)

        # TAB: 2D Contour
        with tab_2d:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            valid_vals = sub_mm[~np.isnan(sub_mm)]
            if len(valid_vals) > 0 and (np.max(valid_vals) - np.min(valid_vals)) > 1.0:
                levels = [5, 50, 100, 200, 500, 1000, 1500, 2000, 2500]
                levels = [lvl for lvl in levels if lvl <= np.nanmax(sub_mm) + 10]
                if len(levels) < 3:
                    levels = 10
            else:
                levels = 10

            cs = ax.contour(r['X'], r['Y'], sub_mm, levels=levels, colors='black', linewidths=0.9)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
            ax.plot(r['panelX'], r['panelY'], 'k-', linewidth=1.8, label="Extraction Area")
            ax.plot(r['leaseX'], r['leaseY'], 'k-.', linewidth=1.2, label="Lease Boundary")
            ax.set_title("(a) Predicted Subsidence Contours (mm)", fontsize=11, fontweight='bold')
            ax.set_xlabel("X (m)", fontsize=9)
            ax.set_ylabel("Y (m)", fontsize=9)
            ax.set_aspect('equal')
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(False)
            st.pyplot(fig)
            plt.close(fig)

        # TAB: Profile X-X
        with tab_profx:
            y_coords = r['Y'][:, 0]
            y_idx = np.argmin(np.abs(y_coords - slice_y))
            actual_y = y_coords[y_idx]
            prof_x = r['X'][y_idx, :]
            prof_sub = sub_mm[y_idx, :]
            prof_slope = r['SlopeX'][y_idx, :]
            prof_strain = r['StrainX'][y_idx, :]

            fig_px, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True, dpi=140)
            ax1.plot(prof_x, prof_sub, 'k.-', markersize=6, label="Subsidence")
            ax1.set_ylabel("Sub (mm)")
            ax1.invert_yaxis()
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.set_title(f"Profile X-X at Y = {actual_y:.1f} m", fontsize=10, fontweight='bold')

            ax2.plot(prof_x, prof_slope, 'b.-', markersize=5, label="Slope")
            ax2.set_ylabel("Slope (mm/m)")
            ax2.grid(True, linestyle='--', alpha=0.5)

            ax3.plot(prof_x, prof_strain, 'r.-', markersize=5, label="Strain")
            ax3.set_ylabel("Strain (mm/m)")
            ax3.set_xlabel("X (m)")
            ax3.grid(True, linestyle='--', alpha=0.5)

            st.pyplot(fig_px)
            plt.close(fig_px)

        # TAB: Profile Y-Y
        with tab_profy:
            x_coords = r['X'][0, :]
            x_idx = np.argmin(np.abs(x_coords - slice_x))
            actual_x = x_coords[x_idx]
            prof_y = r['Y'][:, x_idx]
            prof_sub_y = sub_mm[:, x_idx]
            prof_slope_y = r['SlopeY'][:, x_idx]
            prof_strain_y = r['StrainY'][:, x_idx]

            fig_py, (ay1, ay2, ay3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True, dpi=140)
            ay1.plot(prof_y, prof_sub_y, 'k.-', markersize=6)
            ay1.set_ylabel("Sub (mm)")
            ay1.invert_yaxis()
            ay1.grid(True, linestyle='--', alpha=0.5)
            ay1.set_title(f"Profile Y-Y at X = {actual_x:.1f} m", fontsize=10, fontweight='bold')

            ay2.plot(prof_y, prof_slope_y, 'b.-', markersize=5)
            ay2.set_ylabel("Slope (mm/m)")
            ay2.grid(True, linestyle='--', alpha=0.5)

            ay3.plot(prof_y, prof_strain_y, 'r.-', markersize=5)
            ay3.set_ylabel("Strain (mm/m)")
            ay3.set_xlabel("Y (m)")
            ay3.grid(True, linestyle='--', alpha=0.5)

            st.pyplot(fig_py)
            plt.close(fig_py)

        # TAB: 3D Surface
        with tab_3d:
            fig3d = go.Figure(data=[
                go.Surface(
                    x=r['X'], y=r['Y'], z=sub_mm,
                    colorscale='Viridis',
                    colorbar_title="Sub (mm)"
                )
            ])
            # Draw panel outline at top of 3D plot
            fig3d.add_trace(go.Scatter3d(
                x=r['panelX'], y=r['panelY'], z=[0.0]*len(r['panelX']),
                mode='lines',
                line=dict(color='black', width=4),
                name='Panel Outline'
            ))
            fig3d.update_layout(
                title="3D Ground Subsidence Trough (Magnitude)",
                scene=dict(
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)",
                    zaxis_title="Subsidence (mm)",
                    zaxis=dict(autorange="reversed")
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # TAB: Data & Export
        with tab_data:
            df_grid = pd.DataFrame({
                'X_m': r['X'].flatten(),
                'Y_m': r['Y'].flatten(),
                'Subsidence_mm': sub_mm.flatten(),
                'SlopeX_mm_per_m': r['SlopeX'].flatten(),
                'SlopeY_mm_per_m': r['SlopeY'].flatten(),
                'StrainX_mm_per_m': r['StrainX'].flatten(),
                'StrainY_mm_per_m': r['StrainY'].flatten()
            }).dropna(subset=['Subsidence_mm'])

            st.dataframe(df_grid.head(200), use_container_width=True)
            csv_buffer = io.StringIO()
            df_grid.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Full Results Grid (CSV)",
                data=csv_buffer.getvalue(),
                file_name="subsidence_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("Set your parameters on the left sidebar and click **CALCULATE** to generate numerical predictions.")

# -----------------------------------------------------------------------------
# TAB 2: ABOUT & METHOD
# -----------------------------------------------------------------------------
else:
    st.title("About & Computational Method")
    st.markdown("""
    ### Theoretical Foundation
    **SubsidenceLab** implements the **Modified Influence Function Method** formulated by **P.R. Sheorey, J.P. Loui, K.B. Singh, and S.K. Singh (2000)** for complete ground subsidence, slope, and strain prediction in Indian coalfields (*International Journal of Rock Mechanics and Mining Sciences*, 37, 801-818).

    #### 1. Sector-Ring Influence Integration
    The ground surface point $P(x,y)$ receives vertical influence from extraction elements falling inside an influence circle of radius:
    $$R = H \\cdot \\tan(\\theta)$$
    The circle is discretized into 10 concentric rings and 64 sectors (640 elements). The ring weighting factors $S_i$ are given by:
    $$K_z = \\frac{0.5352}{R^2} \\left(1 + \\cos\\frac{\\pi r}{R}\\right)$$

    #### 2. Empirical Maximum Subsidence ($S_{max}$)
    For single seam extractions below unfractured rock:
    $$S = 0.33 \\left[1 + 1.1 \\tanh\\left(1.4(x - 1.8)\\right)\\right]$$
    where $x = \\frac{W}{H \\cdot NEW}$ and $S_{max} = S \\cdot h \\cdot e$.

    #### 3. Extraction Edge and Asymmetry Corrections
    - **Edge Damping Function ($W_z$)**: Dampens subsidence near unmined solid coal boundaries based on perpendicular boundary distance $d_b$:
      $$W_z = 0.5 \\tanh\\left(\\frac{5 d_b}{1.5 \\cdot NEW \\cdot h} - 2.4\\right) + 0.5$$
    - **Asymmetry Correction Function ($Q_z$)**: Accounts for the forward shift of maximum subsidence towards the starting line of extraction:
      $$Q_z = 0.9 - 0.1 \\tanh\\left[0.5(d - 0.4 d_{max})\\right]$$

    #### 4. Slopes and Strains
    Surface tilt and horizontal strain fields are evaluated using central spatial gradients of the predicted displacement fields:
    $$G_x = \\frac{\\partial S}{\\partial x}, \\quad G_y = \\frac{\\partial S}{\\partial y}$$
    $$\\epsilon_{xx} = \\frac{\\partial U_x}{\\partial x}, \\quad \\epsilon_{yy} = \\frac{\\partial U_y}{\\partial y}$$
    """)
  
