import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import warnings
from typing import Callable, Dict, Tuple

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# ---- Mạng nơ-ron (tùy chọn, không bắt buộc phải có sklearn) ----
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.exceptions import ConvergenceWarning
    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


# =========================
# 1. HÀM TIỆN ÍCH CHUNG
# =========================

def parse_number_list(text: str) -> np.ndarray:
    """
    Mỗi dòng là một giá trị số.
    Ví dụ:
        1.2
        3.4
        5
    """
    if not text:
        return np.array([])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    return np.array([float(ln) for ln in lines], dtype=float)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính hệ số xác định R² (bỏ qua NaN nếu có).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def create_sorted_grid(x: np.ndarray, num_points: int = 400) -> np.ndarray:
    """
    Tạo lưới x mịn để vẽ đường cong (từ min đến max của x).
    """
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    return np.linspace(x_min, x_max, num_points)


# =========================
# 2. CÁC MÔ HÌNH HÀM SỐ CƠ BẢN
# =========================

def fit_linear(x: np.ndarray, y: np.ndarray):
    coef = np.polyfit(x, y, 1)
    a, b = coef

    def f(x_new: np.ndarray) -> np.ndarray:
        return a * x_new + b

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = f"y = {a:.6g}·x + {b:.6g}"
    return f, eq, r2


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int):
    """
    Fit đa thức bậc 'degree': y = a_n x^n + ... + a_0
    """
    coef = np.polyfit(x, y, degree)
    poly = np.poly1d(coef)

    def f(x_new: np.ndarray) -> np.ndarray:
        return poly(x_new)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)

    # Xây dựng chuỗi phương trình
    terms = []
    deg = degree
    for c in coef:
        if abs(c) < 1e-12:
            deg -= 1
            continue
        if deg > 1:
            terms.append(f"{c:.6g}·x^{deg}")
        elif deg == 1:
            terms.append(f"{c:.6g}·x")
        else:
            terms.append(f"{c:.6g}")
        deg -= 1

    eq = "y = " + " + ".join(terms).replace("+ -", "- ")
    return f, eq, r2


def fit_exponential(x: np.ndarray, y: np.ndarray):
    """
    y = a * exp(bx), yêu cầu y > 0
    """
    if np.any(y <= 0):
        raise ValueError("Mô hình hàm mũ yêu cầu mọi giá trị y > 0.")
    ln_y = np.log(y)
    b, ln_a = np.polyfit(x, ln_y, 1)
    a = np.exp(ln_a)

    def f(x_new: np.ndarray) -> np.ndarray:
        return a * np.exp(b * x_new)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = f"y = {a:.6g}·e^({b:.6g}·x)"
    return f, eq, r2


def fit_logarithmic(x: np.ndarray, y: np.ndarray):
    """
    y = a*ln(x) + b, yêu cầu x > 0.
    """
    if np.any(x <= 0):
        raise ValueError("Mô hình logarit yêu cầu mọi giá trị x > 0.")
    ln_x = np.log(x)
    a, b = np.polyfit(ln_x, y, 1)

    def f(x_new: np.ndarray) -> np.ndarray:
        return a * np.log(x_new) + b

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = f"y = {a:.6g}·ln(x) + {b:.6g}"
    return f, eq, r2


def fit_power(x: np.ndarray, y: np.ndarray):
    """
    y = a * x^b, yêu cầu x > 0, y > 0.
    """
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Mô hình lũy thừa yêu cầu mọi giá trị x > 0 và y > 0.")
    ln_x = np.log(x)
    ln_y = np.log(y)
    b, ln_a = np.polyfit(ln_x, ln_y, 1)
    a = np.exp(ln_a)

    def f(x_new: np.ndarray) -> np.ndarray:
        return a * (x_new ** b)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = f"y = {a:.6g}·x^{b:.6g}"
    return f, eq, r2


# =========================
# 3. LOGARIT ĐA THỨC TỔNG QUÁT
# =========================

def fit_log_poly_base(x: np.ndarray, y: np.ndarray, base: float, degree: int):
    """
    Fit mô hình: y = log_base(P_n(x))
    trong đó P_n(x) = a_n x^n + ... + a_0 (đa thức bậc 'degree').

    Dùng biến đổi: base^y ≈ P_n(x),
    rồi giải least squares để tìm các hệ số a_n...a_0.
    """
    if base <= 0 or np.isclose(base, 1.0):
        raise ValueError("Cơ số log phải > 0 và khác 1.")

    t = base ** y  # luôn dương

    # Ma trận thiết kế cho đa thức bậc 'degree'
    powers = [x ** k for k in range(degree, -1, -1)]  # x^degree, ..., x^0
    M = np.column_stack(powers)

    coef, *_ = np.linalg.lstsq(M, t, rcond=None)  # a_n ... a_0

    def f(x_new: np.ndarray) -> np.ndarray:
        powers_new = [x_new ** k for k in range(degree, -1, -1)]
        inner = np.zeros_like(x_new, dtype=float)
        for c, p in zip(coef, powers_new):
            inner += c * p

        eps = 1e-12
        inner = np.where(inner > eps, inner, np.nan)
        return np.log(inner) / np.log(base)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)

    # Xây dựng phương trình
    terms = []
    deg = degree
    for c in coef:
        if abs(c) < 1e-12:
            deg -= 1
            continue
        if deg > 1:
            terms.append(f"{c:.6g}·x^{deg}")
        elif deg == 1:
            terms.append(f"{c:.6g}·x")
        else:
            terms.append(f"{c:.6g}")
        deg -= 1

    poly_str = " + ".join(terms).replace("+ -", "- ")
    eq = f"y = log_{base:g}({poly_str})"
    return f, eq, r2


# =========================
# 4. CÁC MÔ HÌNH NÂNG CAO
# =========================

def fit_trig(x: np.ndarray, y: np.ndarray):
    """
    y = A*sin(ωx) + B*cos(ωx) + C
    """
    def trig_func(x_, A, B, C, omega):
        return A * np.sin(omega * x_) + B * np.cos(omega * x_) + C

    A0 = (np.max(y) - np.min(y)) / 2 if len(y) > 0 else 1.0
    C0 = np.mean(y)
    omega0 = 2 * np.pi / (x.max() - x.min()) if x.max() > x.min() else 1.0
    p0 = [A0, 0.0, C0, omega0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        popt, _ = curve_fit(trig_func, x, y, p0=p0, maxfev=10000)
    A, B, C, omega = popt

    def f(x_new: np.ndarray) -> np.ndarray:
        return trig_func(x_new, A, B, C, omega)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = (
        f"y = {A:.6g}·sin({omega:.6g}·x) + "
        f"{B:.6g}·cos({omega:.6g}·x) + {C:.6g}"
    )
    return f, eq, r2


def fit_logistic(x: np.ndarray, y: np.ndarray):
    """
    y = L / (1 + exp(-k(x - x0))) + b
    """
    def logistic_func(x_, L, x0, k, b):
        z = -k * (x_ - x0)
        z = np.clip(z, -500, 500)
        return L / (1 + np.exp(z)) + b

    L0 = np.max(y) - np.min(y)
    x0_0 = (np.max(x) + np.min(x)) / 2
    k0 = 1.0
    b0 = np.min(y)
    p0 = [L0, x0_0, k0, b0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        popt, _ = curve_fit(logistic_func, x, y, p0=p0, maxfev=10000)
    L, x0, k, b = popt

    def f(x_new: np.ndarray) -> np.ndarray:
        return logistic_func(x_new, L, x0, k, b)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = (
        "y = L / (1 + exp(-k·(x - x0))) + b  "
        f"(L={L:.6g}, x0={x0:.6g}, k={k:.6g}, b={b:.6g})"
    )
    return f, eq, r2


def fit_rational(x: np.ndarray, y: np.ndarray):
    """
    y = (a1·x + a0)/(b1·x + b0)
    """
    def rat_func(x_, a1, a0, b1, b0):
        denom = b1 * x_ + b0
        denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        return (a1 * x_ + a0) / denom

    a1_0 = 1.0
    a0_0 = 0.0
    b1_0 = 0.0
    b0_0 = 1.0
    p0 = [a1_0, a0_0, b1_0, b0_0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        popt, _ = curve_fit(rat_func, x, y, p0=p0, maxfev=10000)
    a1, a0, b1, b0 = popt

    def f(x_new: np.ndarray) -> np.ndarray:
        return rat_func(x_new, a1, a0, b1, b0)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = (
        "y = (a1·x + a0) / (b1·x + b0)  "
        f"(a1={a1:.6g}, a0={a0:.6g}, b1={b1:.6g}, b0={b0:.6g})"
    )
    return f, eq, r2


def fit_spline(x: np.ndarray, y: np.ndarray):
    """
    Spline bậc 3 nội suy (UnivariateSpline, s=0).
    """
    spline = UnivariateSpline(x, y, s=0, k=3)

    def f(x_new: np.ndarray) -> np.ndarray:
        return spline(x_new)

    y_pred = f(x)
    r2 = r2_score(y, y_pred)
    eq = f"Cubic spline (UnivariateSpline, {len(x)} điểm nút)"
    return f, eq, r2


if HAS_SKLEARN:
    def fit_nn(x: np.ndarray, y: np.ndarray):
        """
        Mạng nơ-ron đơn giản: MLPRegressor với 2 hidden layers.
        """
        X = x.reshape(-1, 1)
        mlp = MLPRegressor(
            hidden_layer_sizes=(20, 20),
            activation="relu",
            max_iter=5000,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            mlp.fit(X, y)

        def f(x_new: np.ndarray) -> np.ndarray:
            X_new = x_new.reshape(-1, 1)
            return mlp.predict(X_new)

        y_pred = f(x)
        r2 = r2_score(y, y_pred)
        eq = "MLPRegressor (20-20-1, activation=ReLU)"
        return f, eq, r2
else:
    # Nếu chưa cài sklearn mà vẫn cố gọi thì báo lỗi rõ ràng
    def fit_nn(*args, **kwargs):
        raise ImportError("Mạng nơ-ron (MLP) yêu cầu cài thêm thư viện scikit-learn.")


# =========================
# 5. ỨNG DỤNG STREAMLIT
# =========================

def main():
    st.set_page_config(page_title="Fitting hàm số từ dữ liệu x–y", layout="wide")
    st.title("🔢 Fitting hàm số từ dữ liệu x – y")
    st.write(
        "Nhập các giá trị **x** và **y** tương ứng. **Lưu ý: Nhập mỗi giá trị trên một dòng mới.** "
        "Ứng dụng sẽ thử nhiều dạng hàm số, hiển thị **phương trình**, **R²** và **biểu đồ tương tác**."
    )

    # --- Sidebar: cấu hình mô hình ---
    st.sidebar.header("⚙️ Cài đặt mô hình cơ bản")

    use_linear = st.sidebar.checkbox("Hàm tuyến tính (y = a·x + b)", value=False)

    # Đa thức: tự động chạy từ bậc min -> max
    use_poly = st.sidebar.checkbox("Hàm đa thức (y = aₙxⁿ + … + a₀)", value=False)
    if use_poly:
        st.sidebar.markdown("**Khoảng bậc đa thức**")
        poly_min_deg = st.sidebar.number_input(
            "Bậc thấp nhất", min_value=1, max_value=20, value=2, step=1
        )
        poly_max_deg = st.sidebar.number_input(
            "Bậc cao nhất", min_value=poly_min_deg, max_value=20, value=10, step=1
        )
    else:
        poly_min_deg, poly_max_deg = 2, 2

    use_exp = st.sidebar.checkbox("Hàm mũ (y = a·e^{b·x})", value=False)
    use_log = st.sidebar.checkbox("Hàm logarit (y = a·ln(x) + b)", value=False)
    use_power = st.sidebar.checkbox("Hàm lũy thừa (y = a·x^b)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("✨ Logarit đa thức tổng quát")
    use_log_poly = st.sidebar.checkbox(
        "Hàm logarit đa thức: y = log₍base₎(Pₙ(x))", value=False
    )
    if use_log_poly:
        log_poly_base = st.sidebar.number_input(
            "Cơ số (base) của log", min_value=1.00001, max_value=100.0, value=3.0, step=0.5
        )
        st.sidebar.markdown("**Khoảng bậc đa thức cho Pₙ(x)**")
        log_poly_min_deg = st.sidebar.number_input(
            "Bậc thấp nhất (log-poly)", min_value=1, max_value=20, value=1, step=1
        )
        log_poly_max_deg = st.sidebar.number_input(
            "Bậc cao nhất (log-poly)",
            min_value=log_poly_min_deg,
            max_value=20,
            value=4,
            step=1,
        )
    else:
        log_poly_base = 3.0
        log_poly_min_deg, log_poly_max_deg = 1, 1

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Mô hình nâng cao")
    use_trig = st.sidebar.checkbox("Hàm sin/cos", value=False)
    use_logistic = st.sidebar.checkbox("Hàm logistic", value=False)
    use_rational = st.sidebar.checkbox("Hàm phân thức hữu tỉ", value=False)
    use_spline = st.sidebar.checkbox("Spline bậc 3", value=False)
    if HAS_SKLEARN:
        use_nn = st.sidebar.checkbox("Mạng nơ-ron (MLP)", value=False)
    else:
        st.sidebar.markdown(
            "⚠️ Mạng nơ-ron (MLP) cần thư viện `scikit-learn`.\n\n"
            "Nếu muốn dùng, hãy cài thêm:\n"
            "`pip install scikit-learn`"
        )
        use_nn = False



    # --- Nhập dữ liệu ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Giá trị x")
        x_text = st.text_area(
            "Danh sách x",
            "",
            height=160,
        )
    with col2:
        st.subheader("Giá trị y")
        y_text = st.text_area(
            "Danh sách y tương ứng",
            "",
            height=160,
        )

    st.markdown("---")

    if st.button("🚀 Thực hiện fitting", type="primary"):
        # --- Xử lý dữ liệu ---
        x = parse_number_list(x_text)
        y = parse_number_list(y_text)

        if x.size == 0 or y.size == 0:
            st.error("Vui lòng nhập đầy đủ danh sách x và y.")
            return
        if x.size != y.size:
            st.error(f"Số lượng phần tử x ({x.size}) và y ({y.size}) không bằng nhau.")
            return
        if x.size < 3:
            st.error("Cần ít nhất 3 điểm dữ liệu để fitting các mô hình.")
            return

        # Sắp xếp theo x
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        st.success(f"Đã nhận {x.size} cặp dữ liệu hợp lệ.")

        # --- Fit các mô hình ---
        models: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], str, float]] = {}

        if use_linear:
            f, eq, r2 = fit_linear(x, y)
            models["Hàm tuyến tính"] = (f, eq, r2)

        if use_poly:
            for deg in range(int(poly_min_deg), int(poly_max_deg) + 1):
                try:
                    f, eq, r2 = fit_polynomial(x, y, deg)
                    models[f"Hàm đa thức bậc {deg}"] = (f, eq, r2)
                except np.linalg.LinAlgError:
                    st.warning(f"Không thể fit đa thức bậc {deg}: ma trận suy biến.")

        if use_exp:
            try:
                f, eq, r2 = fit_exponential(x, y)
                models["Hàm mũ"] = (f, eq, r2)
            except ValueError as e:
                st.warning(str(e))

        if use_log:
            try:
                f, eq, r2 = fit_logarithmic(x, y)
                models["Hàm logarit"] = (f, eq, r2)
            except ValueError as e:
                st.warning(str(e))

        if use_power:
            try:
                f, eq, r2 = fit_power(x, y)
                models["Hàm lũy thừa"] = (f, eq, r2)
            except ValueError as e:
                st.warning(str(e))

        if use_log_poly:
            for deg in range(int(log_poly_min_deg), int(log_poly_max_deg) + 1):
                try:
                    f, eq, r2 = fit_log_poly_base(
                        x, y, base=log_poly_base, degree=deg
                    )
                    models[
                        f"Hàm logarit đa thức bậc {deg} (base={log_poly_base:g})"
                    ] = (f, eq, r2)
                except ValueError as e:
                    st.warning(f"Log-poly bậc {deg}: {e}")

        # Mô hình nâng cao
        if use_trig:
            try:
                f, eq, r2 = fit_trig(x, y)
                models["Hàm sin/cos"] = (f, eq, r2)
            except Exception as e:
                st.warning(f"Không fit được hàm sin/cos: {e}")

        if use_logistic:
            try:
                f, eq, r2 = fit_logistic(x, y)
                models["Hàm logistic"] = (f, eq, r2)
            except Exception as e:
                st.warning(f"Không fit được hàm logistic: {e}")

        if use_rational:
            try:
                f, eq, r2 = fit_rational(x, y)
                models["Hàm phân thức hữu tỉ"] = (f, eq, r2)
            except Exception as e:
                st.warning(f"Không fit được hàm phân thức hữu tỉ: {e}")

        if use_spline:
            try:
                f, eq, r2 = fit_spline(x, y)
                models["Spline bậc 3"] = (f, eq, r2)
            except Exception as e:
                st.warning(f"Không fit được spline: {e}")

        if use_nn and HAS_SKLEARN:
            try:
                f, eq, r2 = fit_nn(x, y)
                models["Mạng nơ-ron (MLP)"] = (f, eq, r2)
            except Exception as e:
                st.warning(f"Không fit được MLP: {e}")

        if not models:
            st.error("Không có mô hình nào được fit thành công. Hãy kiểm tra lại dữ liệu và lựa chọn.")
            return

        # --- Lưu models vào session_state ---
        st.session_state.models = models
        st.session_state.x = x
        st.session_state.y = y

    # =========================================================
    # PHẦN HIỂN THỊ KẾT QUẢ (Lấy từ Session State để không bị mất khi reload)
    # =========================================================
    if "models" in st.session_state and st.session_state.models:
        models = st.session_state.models
        x = st.session_state.x
        y = st.session_state.y
        
        # Tạo lại lưới x để vẽ
        x_grid = create_sorted_grid(x)

        # --- Bảng tổng hợp & mô hình tốt nhất ---
        st.subheader("📋 Tổng hợp các mô hình và độ phù hợp (R²)")
        rows = []
        for name, (f, eq, r2) in models.items():
            rows.append({"Mô hình": name, "Phương trình": eq, "R²": r2})
        df_summary = pd.DataFrame(rows)

        df_sorted = df_summary.sort_values("R²", ascending=False).reset_index(drop=True)
        best = df_sorted.iloc[0]
        st.markdown(
            f"✅ **Mô hình phù hợp nhất (R² lớn nhất):** "
            f"**{best['Mô hình']}** với `R² = {best['R²']:.6f}`"
        )
        st.dataframe(df_sorted, use_container_width=True)

        # --- Biểu đồ tương tác (Plotly) ---
        st.subheader("📈 Biểu đồ dữ liệu và các đường fitting (tương tác)")
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Dữ liệu gốc",
                marker=dict(size=9, symbol="circle"),
                hovertemplate="x=%{x}<br>y=%{y}<extra>Dữ liệu</extra>",
            )
        )

        for name, (f, eq, r2) in models.items():
            y_grid = f(x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_grid,
                    mode="lines",
                    name=f"{name} (R²={r2:.4f})",
                    hovertemplate="x=%{x}<br>y=%{y}<extra>" + name + "</extra>",
                )
            )

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            hovermode="x unified",
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            template="plotly_white",
            height=550,
        )
        fig.update_xaxes(showgrid=True, zeroline=True)
        fig.update_yaxes(showgrid=True, zeroline=True)

        st.plotly_chart(fig, use_container_width=True)

        # --- Chi tiết từng mô hình ---
        st.subheader("🔍 Chi tiết từng mô hình")
        model_names = list(models.keys())
        selected_name = st.selectbox("Chọn mô hình để xem chi tiết", model_names)

        f_sel, eq_sel, r2_sel = models[selected_name]
        st.markdown(f"### {selected_name}")
        st.markdown("**Phương trình:**")
        st.code(eq_sel)
        st.markdown(f"**R²:** `{r2_sel:.6f}`")

        fig_m = go.Figure()
        fig_m.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Dữ liệu gốc",
                marker=dict(size=9),
            )
        )
        y_grid = f_sel(x_grid)
        fig_m.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name=selected_name,
            )
        )
        fig_m.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig_m, use_container_width=True)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            padding: 20px 10px;
            margin-top: 40px;
            color: var(--text-color);
        }
        .footer p {
            margin: 5px 0;
        }
        .footer a {
            text-decoration: none;
            margin: 0 10px;
            font-weight: 600;
        }
        .footer a.facebook { color: #1877F2; }
        .footer a.youtube { color: #FF0000; }
        .footer a:hover { opacity: 0.8; }
        </style>
        <div class="footer">
            <p>© 2025 <b>Văn Quân Bùi</b>. All rights reserved.</p>
            <p>
                <a class="facebook" href="https://www.facebook.com/Thewind1104" target="_blank">Facebook</a>
                <a class="youtube" href="https://www.youtube.com/@thewind2002" target="_blank">Youtube</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
