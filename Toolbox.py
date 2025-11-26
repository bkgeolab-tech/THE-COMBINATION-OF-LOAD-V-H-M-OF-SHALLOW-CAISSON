import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

def build_ann_plotly_figure(df):

    df = df[["V/V0", "beta", "H/suTCA", "M/suTCAD", "V/suTCA"]].dropna()

    if df.empty:
        return go.Figure()

    df = df.sort_values(["beta", "V/V0"])

    v_vals = sorted(df["V/V0"].unique())
    beta_vals = sorted(df["beta"].unique())

    Z = df.pivot(index="beta", columns="V/V0", values="V/suTCA").values
    X = df.pivot(index="beta", columns="V/V0", values="H/suTCA").values
    Y = df.pivot(index="beta", columns="V/V0", values="M/suTCAD").values

    fig = go.Figure()

    fig.add_surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[[0, "#d0ecff"], [1, "#2a76d2"]],
        opacity=0.9,
        showscale=False,
    )

    for i in range(len(beta_vals)):
        fig.add_trace(
            go.Scatter3d(
                x=X[i, :],
                y=Y[i, :],
                z=Z[i, :],
                mode="lines",
                line=dict(color="rgba(0, 60, 150, 0.8)", width=2),
                showlegend=False,
            )
        )

    for j in range(len(v_vals)):
        fig.add_trace(
            go.Scatter3d(
                x=X[:, j],
                y=Y[:, j],
                z=Z[:, j],
                mode="lines",
                line=dict(color="rgba(0,60,150,0.85)", width=1.5),
                showlegend=False,
            )
        )

    fig.update_layout(
        width = 900,
        height = 700,
        
        scene=dict(
            xaxis_title="H/suTCA",
            yaxis_title="M/suTCAD",
            zaxis_title="V/suTCA",
    
            # White background
            xaxis=dict(
                backgroundcolor="#ffbebd",
                gridcolor="lightgray"
            ),
            yaxis=dict(
                backgroundcolor="#ffbebd",
                gridcolor="lightgray"
            ),
            zaxis=dict(
                backgroundcolor="#ffbebd",
                gridcolor="lightgray"
            ),
    
            aspectmode="manual",

            aspectratio=dict(x=1.4, y=1, z=0.8)
        ),
    
        # White figure background
        paper_bgcolor="white",
        plot_bgcolor="white",
    
        margin=dict(l=0, r=0, b=0, t=30),
        template="none"
    )

    fig.update_scenes(camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)))
    
    # Add manual point if available
    if "manual_point" in st.session_state:
        p = st.session_state.manual_point
    
        # Normalized coordinates
        Hn = p["H_norm"]
        Mn = p["M_norm"]
        Vn = p["V_norm"]
    
        # Add point to 3D plot
        fig.add_trace(
            go.Scatter3d(
                x=[Hn],
                y=[Mn],
                z=[Vn],
                mode="markers+text",
                marker=dict(size=8, color="red", opacity=1.0),
                text=["Manual Load"],
                textposition="top center",
                name="Manual Point"
            )
        )
        # Add Excel load cases (if available)
        if "load_points" in st.session_state:
            dfp = st.session_state.load_points
            fig.add_trace(go.Scatter3d(
                x=dfp["H_norm"],
                y=dfp["M_norm"],
                z=dfp["V_norm"],
                mode="markers",
                marker=dict(size=6, color="red", opacity=0.9),
                name="Load Cases"
            ))
    return fig  
        
# ================== LOAD MODEL ==================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Bestmodeltraining.pkl")

@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)

bundle  = load_bundle(MODEL_PATH)
model   = bundle["model"]
scalerX = bundle.get("scaler_X", None)
scalerY = bundle.get("scaler_y", None)

# ================== PAGE & THEME ==================
st.set_page_config(page_title="Bearing Capacity Predictor", page_icon="🧮", layout="centered")

st.markdown("""
<style>

div[data-testid="stFileUploader"] {
    border: none !important;
    padding: 0 !important;
    background: none !important;
}

/* DRAG & DROP BOX (inner box) */
div[data-testid="stFileDropzone"] {
    background-color: #ffffff !important;
    border: 1px dashed #999999 !important;
    border-radius: 6px !important;
    padding: 12px !important;
}

/* Text inside */
div[data-testid="stFileDropzone"] * {
    color: #000000 !important;
}

/* Icon */
div[data-testid="stFileDropzone"] svg {
    fill: #000000 !important;
}

/* Browse files button */
div[data-testid="stFileUploader"] button {
    background-color:#ffffff !important;
    color:#000000 !important;
    border:1px solid #cccccc !important;
    border-radius:6px !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.stApp { background:#ffffff !important; color:#000000 !important; }
header[data-testid="stHeader"] { display:none; }
.white-container { padding-top: 1rem; }

/* inputs & buttons */
.stNumberInput input { background:#fff !important; color:#000 !important; border:1px solid #00000040 !important; border-radius:6px !important; }
.stButton > button, .stDownloadButton > button { background:#fff !important; color:#000 !important; border:1px solid #000 !important; border-radius:6px !important; }

/* image helpers */
.duo-img, .footer-img {
  max-width: 100%;
  height: auto;
  object-fit: contain;
  display: white;
  margin: 0 auto;
}
</style>

""", unsafe_allow_html=True)

st.title("Problem Definition")


# ================== SIMPLE TOP-BOTTOM IMAGE DISPLAY ==================

root_dir = Path(__file__).parent

top_path    = (root_dir / "Problem_definition_1.svg").resolve()
bottom_path = (root_dir / "Problem_definition_2.svg").resolve()


# ========== IMAGE HELPERS ==========
def _data_url(path: Path):
    if not path.exists():
        return None, None
    ext = path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext)
    if not mime:
        return None, None
    b64 = base64.b64encode(path.read_bytes()).decode()
    return mime, b64


def _img_html(path: Path):
    mime, b64 = _data_url(path)
    if not mime:
        return ""
    return f"""
        <img src="data:{mime};base64,{b64}"
             style="
                width:100%;
                height:auto;
                display:block;
                margin:20px auto;
             ">
    """


# ========== RENDER IMAGES (TOP → BOTTOM) ==========
def render_top_banner(top: Path, bottom: Path):

    top_html    = _img_html(top)
    bottom_html = _img_html(bottom)

    if top_html:
        st.markdown(top_html, unsafe_allow_html=True)

    if bottom_html:
        st.markdown(bottom_html, unsafe_allow_html=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #eee;margin:20px 0;'>",
        unsafe_allow_html=True
    )

render_top_banner(top_path, bottom_path)


# ================== INPUT FORM ==================
st.markdown("### Input Parameters")

st.markdown("### Warning: the input parameters should be in the range of the trained model. Otherwise, the results might not be reliable. The range below:")

st.markdown("### L/D: 0 → 1, rₑ: 0.5 → 1")

c1 = st.columns([1])[0]

with c1:

    st.latex("L \ (m)")
    L = st.number_input("L", label_visibility="collapsed", min_value=0.0, format="%.4f", key="L_input")
    
    st.latex("D \ (m)")
    D = st.number_input("D", label_visibility="collapsed", min_value=0.0, format="%.4f", key="D_input")

    st.latex(r"s_{uc} \ (kPa)")
    suc = st.number_input("suc", label_visibility="collapsed", min_value=0.0, format="%.4f", key="suc")

    st.latex(r"s_{ue} \ (kPa)")
    sue = st.number_input("sue", label_visibility="collapsed", min_value=0.0, format="%.4f", key="sue")

        # Auto compute L/D
    if D > 0:
        L_over_D = L / D
    else:
        L_over_D = 0.0
    
    st.latex(r"\frac{L}{D}")
    st.text_input("L/D (auto = L / D)", value=f"{L_over_D:.4f}", disabled=True,label_visibility="collapsed")

    # r_e auto
    r_e = sue / suc if suc > 0 else 0.0

    # Display readonly field
    st.latex(r"r_e = \frac{s_{ue}}{s_{uc}}")
    st.text_input("r_e (auto = sue / suc)", value=f"{r_e:.4f}", disabled=True,label_visibility="collapsed")
    
    # ===================== MANUAL USER INPUT =====================
    st.markdown("### Manual Applied Loads")
    
    # Manual H, M, V
    st.latex(r"H \ (kN)")
    H_manual = st.number_input("H_manual", label_visibility="collapsed", min_value=0.0, format="%.4f", key="H_manual")

    st.latex(r"M \ (kN.m)")
    M_manual = st.number_input("M_manual", label_visibility="collapsed",min_value=0.0, format="%.4f", key="M_manual")
  
    st.latex(r"V \ (kN)")
    V_manual = st.number_input("V_manual", label_visibility="collapsed",min_value=0.0, format="%.4f", key="V_manual")
    
    # Compute area & suTCA
    A = 3.141592653589793 * (D ** 2) / 4.0
    suTCA = A * suc
    suTCAD = suTCA * D
    
    # Normalized manual values
    Hn = H_manual / suTCA if suTCA > 0 else 0.0
    Mn = M_manual / suTCAD if suTCAD > 0 else 0.0
    Vn = V_manual / suTCA if suTCA > 0 else 0.0
    
    # Display normalized results
    st.markdown("#### Normalized Manual Loads")
    st.latex(rf"H/s_{{uTCA}} :\ {Hn:.4f}")
    st.latex(rf"M/s_{{uTCAD}} :\ {Mn:.4f}")
    st.latex(rf"V/s_{{uTCA}} :\ {Vn:.4f}")
    
    # Save manual point for 3D plotting later
    st.session_state.manual_point = {
        "H": H_manual, "M": M_manual, "V": V_manual,
        "H_norm": Hn, "M_norm": Mn, "V_norm": Vn,
        "A": A, "suTCA": suTCA, "suTCAD": suTCAD,
        "L": L, "D": D, "suc": suc, "sue": sue, "r_e": r_e,
        "L_over_D": L_over_D
    }

    st.markdown("### Upload Load Cases (Excel)")
    
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        df_loads = pd.read_excel(uploaded_file)
    
        # Required columns
        expected_cols = {"H", "M", "V"}
    
        if not expected_cols.issubset(df_loads.columns):
            st.error("Excel must contain columns: H, M, V")
        else:
            st.success(f"Loaded {len(df_loads)} load cases successfully.")
    
            # Normalize all rows
            df_loads["H_norm"] = df_loads["H"] / suTCA
            df_loads["M_norm"] = df_loads["M"] / suTCAD
            df_loads["V_norm"] = df_loads["V"] / suTCA
    
            st.session_state.load_points = df_loads
    
            st.markdown("#### Normalized Load Cases")
            styled_df = df_loads.style.set_properties(**{
                'background-color': 'white',
                'color': 'black',
                'border-color': '#00000040'
            })
            st.dataframe(styled_df, use_container_width=True)   

# ================== SESSION STATE ==================
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame(
        columns=["L/D","suc","sue","r_e","V/V0","beta",
                 "H/suTCA","M/suTCAD","V/suTCA"]
    )

# ================== PREDICT ==================
if st.button("🔮 Predict"):
    try:

        # 1. Reset table
        st.session_state.results = pd.DataFrame()

        # 2. Generate V/V0 list (0 → 1 step 0.1)
        V_list = [i/10 for i in range(11)]

        # 3. Generate beta list (0 → 360 step 15)
        beta_full = list(range(0, 361, 15))

        rows = []

        for v in V_list:

           for beta in beta_full:

               # Determine effective beta + sign symmetry

               beta_eff = beta % 180
                
               if beta < 180:
                    sign = 1
               elif beta == 180:
                    sign = -1
               else:
                    sign = -1  

               if beta == 360:
                    sign = 1

               # Build ML input
               dfX = pd.DataFrame(
                   [[L_over_D, r_e, v, beta_eff]],
                   columns=["L/D", "r_e", "V/V0", "beta"]
               )

               # Predict
               X_scaled = scalerX.transform(dfX) if scalerX is not None else dfX.values
               y_pred_norm = model.predict(X_scaled)
               y_pred = scalerY.inverse_transform(y_pred_norm)[0] if scalerY else y_pred_norm[0]

               # Symmetry for β > 180
               H = y_pred[0] * sign
               M = y_pred[1] * sign
               V = y_pred[2]

               # Append
               rows.append({
                   "L/D": L_over_D,
                   "s_uc": suc,
                   "s_ue": sue,
                   "r_e": r_e,
                   "V/V0": v,
                   "beta": beta,
                   "H/suTCA": H,
                   "M/suTCAD": M,
                   "V/suTCA": V
               })

        # Store full table
        st.session_state.results = pd.DataFrame(rows)
        
    except Exception as e:
        st.error(f"Errors: {e}")


# ================== RESULTS TABLE ==================
def render_results_table_white(df: pd.DataFrame):
    rename_map = {"L/D":"L/D",
                "s_uc": "s<sub>uc</sub>",
                "s_ue": "s<sub>ue</sub>",
                "r_e": "r<sub>e</sub>",
                "beta": "&beta;",
                "V/V0": "V/V<sub>0</sub>",
                "H/suTCA": "H/s<sub>uTCA</sub>",
                "M/suTCAD": "M/s<sub>uTCAD</sub>",
                "V/suTCA": "V/s<sub>uTCA</sub>",}
    df_show = df.rename(columns=rename_map).copy()
    df_fmt = df_show.copy()
    for col in ["H/suTCA","M/suTCAD","V/suTCA"]:
        if col in df_fmt.columns:
            df_fmt[col] = pd.to_numeric(df_fmt[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for col in ["L/D","rₑ","V/V₀","β"]:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda v: (f"{v:.10g}" if isinstance(v,(int,float,np.floating)) else str(v)))

    styled = (df_fmt.style.hide(axis="index")
              .set_table_styles([
                  {"selector":"table","props":"border-collapse:collapse;width:100%;background:#fff;color:#000;font-size:15px;"},
                  {"selector":"th","props":"background:#fff;color:#000;text-align:center;font-weight:700;font-size:16px;padding:10px 12px;border-bottom:2px solid rgba(0,0,0,.25);"},
                  {"selector":"td","props":"background:#fff;color:#000;text-align:center;padding:10px 12px;border-bottom:1px solid rgba(0,0,0,.12);"},
              ]))
    st.markdown(styled.to_html(), unsafe_allow_html=True)

if not st.session_state.results.empty:
    st.markdown("### Table of Results")
    render_results_table_white(st.session_state.results)
    csv_bytes = st.session_state.results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_bytes, "prediction_results.csv", "text/csv")

# ================== DRAWING THE DIAGRAM ==================
st.markdown("### 📊 3D Visualization (Interaction Diagrams)")

if st.button("🎨 Plot 3D Surface"):
    df_plot = st.session_state.results.copy()
    fig3d = build_ann_plotly_figure(df_plot)
    st.plotly_chart(fig3d, use_container_width=True)

    
