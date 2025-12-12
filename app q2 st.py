import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import re
import io
import zipfile
from scipy.optimize import linprog

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Solver PL - Graphic Method", page_icon="📐", layout="wide")

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 150
})

# =============================================================================
# 1. PARSER (Input Reader)
# =============================================================================
def parse_problem(text):
    text = re.sub(r'(?i)(Subject to|Sujeito a\s*[:]?)\s*', r'\1\n', text)
    text = re.sub(r'(?i)(Such that|Tal que\s*[:]?)\s*', r'\n\1\n', text)
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines: raise ValueError("Empty text.")

    opt_type = None
    c, A, b, signs = [], [], [], []
    term_pattern = re.compile(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)')
    reading_constraints = False
    var_indices = set()

    for line in lines:
        line_lower = line.lower()
        
        # Objective Function
        if 'max' in line_lower or 'min' in line_lower:
            opt_type = 'min' if 'min' in line_lower else 'max'
            if '=' not in line: raise ValueError("Objective must contain '='.")
            eq = line.split('=')[1].strip()
            matches = term_pattern.findall(eq)
            if not matches: raise ValueError("Variables not found in objective.")
            
            temp_c = {}
            for coeff, idx in matches:
                idx = int(idx) - 1
                var_indices.add(idx)
                coeff = coeff.replace(' ', '')
                val = 1.0 if coeff in ['', '+'] else -1.0 if coeff == '-' else float(coeff)
                temp_c[idx] = val
            n_vars = max(var_indices) + 1 if var_indices else 0
            c = np.zeros(n_vars)
            for i, v in temp_c.items(): c[i] = v
            continue

        if 'sujeito a' in line_lower or 'subject' in line_lower: reading_constraints = True; continue
        if 'tal que' in line_lower or 'such that' in line_lower: reading_constraints = False; continue

        if reading_constraints:
            sign, sep = None, None
            if '<=' in line: sep, sign = '<=', '<='
            elif '>=' in line: sep, sign = '>=', '>='
            elif '=' in line: sep, sign = '=', '='
            
            if sep:
                parts = line.split(sep)
                try:
                    b_val = float(re.split(r'\s', parts[1].strip())[0])
                    matches = term_pattern.findall(parts[0])
                    if not matches: continue
                    
                    row = np.zeros(len(c)) if len(c)>0 else np.zeros(2)
                    for coeff, idx in matches:
                        idx = int(idx) - 1
                        if idx >= len(row): 
                            new = np.zeros(idx+1); new[:len(row)] = row; row = new
                        coeff = coeff.replace(' ', '')
                        k = 1.0 if coeff in ['', '+'] else -1.0 if coeff == '-' else float(coeff)
                        row[idx] = k
                    A.append(row); b.append(b_val); signs.append(sign)
                except: continue

    if opt_type is None: raise ValueError("Objective not found.")
    if not A: raise ValueError("No constraints found.")
    
    # Dimension adjustment
    max_len = max(len(c), max([len(r) for r in A]) if A else 0)
    if len(c) < max_len: 
        new = np.zeros(max_len); new[:len(c)] = c; c = new
    A_fin = [np.pad(r, (0, max_len-len(r))) if len(r)<max_len else r for r in A]
    
    return np.array(c), np.array(A_fin), np.array(b), signs, opt_type

# =============================================================================
# 2. ROBUST GRAPHICAL ENGINE (Grid Sampling & Exact Vertices)
# =============================================================================
def get_plot_limits(vertices, A, b):
    """Calculates auto-zoom limits based on vertices and constraints."""
    if len(vertices) == 0: return 0, 10, 0, 10 # Default
    
    min_x, max_x = np.min(vertices[:,0]), np.max(vertices[:,0])
    min_y, max_y = np.min(vertices[:,1]), np.max(vertices[:,1])
    
    margin_x = (max_x - min_x) * 0.2 if max_x > min_x else 1.0
    margin_y = (max_y - min_y) * 0.2 if max_y > min_y else 1.0
    
    return max(0, min_x - margin_x), max_x + margin_x, max(0, min_y - margin_y), max_y + margin_y

def is_feasible(point, A, b, signs):
    """Checks if a point satisfies all constraints."""
    if point[0] < -1e-7 or point[1] < -1e-7: return False
    for i, row in enumerate(A):
        val = np.dot(row, point)
        if signs[i] == '<=' and val > b[i] + 1e-7: return False
        if signs[i] == '>=' and val < b[i] - 1e-7: return False
        if signs[i] == '=' and abs(val - b[i]) > 1e-7: return False
    return True

def solve_exact_vertices(c, A, b, signs):
    """Calculates all exact vertices of the feasible region."""
    # Add axes as constraints for intersection calculation
    A_aug = np.vstack([A, np.eye(2)])
    b_aug = np.concatenate([b, [0, 0]])
    
    vertices = []
    for i, j in itertools.combinations(range(len(b_aug)), 2):
        mat = np.array([A_aug[i], A_aug[j]])
        rhs = np.array([b_aug[i], b_aug[j]])
        
        if abs(np.linalg.det(mat)) > 1e-10:
            try:
                p = np.linalg.solve(mat, rhs)
                if is_feasible(p, A, b, signs):
                    # Check duplicate
                    if not any(np.linalg.norm(p - v) < 1e-7 for v in vertices):
                        vertices.append(p)
            except: pass
            
    return np.array(vertices)

def generate_robust_plot(c, A, b, signs, opt_type, title):
    """
    Generates the plot using Grid Sampling for the region and Analytical calc for vertices.
    """
    # 1. Exact Vertices
    vertices = solve_exact_vertices(c, A, b, signs)
    
    # 2. Auto-Zoom Limits
    x_min, x_max, y_min, y_max = get_plot_limits(vertices, A, b)
    
    # Expand slightly for visibility
    x_max = max(x_max, 10); y_max = max(y_max, 10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 3. Grid Sampling (The "Robust" Filling)
    # Create a grid of points to test feasibility
    res = 300 # Resolution
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    
    # Create mask: True where feasible
    mask = (X >= 0) & (Y >= 0) # Non-negativity
    for i, row in enumerate(A):
        val = row[0]*X + row[1]*Y
        if signs[i] == '<=': mask &= (val <= b[i] + 1e-5)
        elif signs[i] == '>=': mask &= (val >= b[i] - 1e-5)
        elif signs[i] == '=': mask &= (np.abs(val - b[i]) < 0.1) # Tolerance for equality visual
        
    # Plot Region using contourf or imshow
    if np.any(mask):
        ax.contourf(X, Y, mask, levels=[0.5, 1.5], colors=['mediumseagreen'], alpha=0.3)
        # Check Unboundedness: if mask touches the edges (top or right)
        if np.any(mask[-1, :]) or np.any(mask[:, -1]):
             ax.text(x_max*0.9, y_max*0.9, "PROVAVELMENTE ILIMITADA", 
                     color='darkgreen', ha='right', fontweight='bold', backgroundcolor='white')
    else:
        ax.text(x_max/2, y_max/2, "REGIÃO VAZIA (INVIÁVEL)", 
                color='red', ha='center', fontweight='bold')

    # 4. Plot Constraint Lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(A)))
    for i, (row, val) in enumerate(zip(A, b)):
        if abs(row[1]) > 1e-6: # Not vertical
            y_line = (val - row[0]*x) / row[1]
            # Clip to view
            valid = (y_line >= y_min) & (y_line <= y_max)
            ax.plot(x[valid], y_line[valid], label=f'R{i+1}', color=colors[i], linewidth=2)
        else: # Vertical
            x_line = val / row[0]
            if x_min <= x_line <= x_max:
                ax.vlines(x_line, y_min, y_max, label=f'R{i+1}', color=colors[i], linewidth=2)

    # 5. Plot Vertices & Optimal
    z_opt_val = None
    opt_point = None
    
    if len(vertices) > 0:
        z_vals = np.dot(vertices, c)
        best_idx = np.argmin(z_vals) if opt_type == 'min' else np.argmax(z_vals)
        z_opt_val = z_vals[best_idx]
        opt_point = vertices[best_idx]
        
        ax.scatter(vertices[:,0], vertices[:,1], c='black', s=40, zorder=5, label='Vértices')
        ax.scatter(opt_point[0], opt_point[1], c='red', s=150, marker='*', zorder=6, label='Ótimo')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    return fig, vertices, z_opt_val, opt_point

# =============================================================================
# 3. TEXT FORMATTER
# =============================================================================
def format_model_text(c, A, b, signs, type):
    sense = "Minimizar" if type == 'min' else "Maximizar"
    terms = []
    for i, v in enumerate(c):
        if abs(v) > 1e-9:
            s = "+" if v >= 0 else "-"
            if i==0 and v>=0: s=""
            terms.append(f"{s} {abs(v):.2f}x{i+1}")
    z_str = "".join(terms).strip()
    
    txt = f"{sense} : Z = {z_str}\nSujeito a :\n"
    for i, (row, val) in enumerate(zip(A, b)):
        row_t = []
        for j, k in enumerate(row):
            if abs(k) > 1e-9:
                s = "+" if k>=0 else "-"
                if len(row_t)==0 and k>=0: s=""
                row_t.append(f"{s} {abs(k):.2f}x{j+1}")
        lhs = "".join(row_t).strip()
        if not lhs: lhs="0"
        op = "<=" if signs[i]=='<=' else ">=" if signs[i]=='>=' else "="
        txt += f"  {lhs} {op} {val}\n"
    txt += "Tal que : x1, x2 >= 0"
    return txt

# =============================================================================
# 4. INTERFACE
# =============================================================================
st.title("📊 Solver PL - Método Gráfico (Robust)")
st.markdown("Ferramenta visual para PL com 2 variáveis. Suporta regiões ilimitadas e não-convexas.")

exercises = {
    "Personalizado": "",
    "Ex 17: Energéticos": "Minimizar : Z = 0.06x1 + 0.08x2\nSujeito a :\n8x1 + 6x2 >= 48\n1x1 + 2x2 >= 12\n1x1 + 2x2 <= 20\nTal que : x1, x2 >= 0",
    "Ex 18: Quinquilharias": "Maximizar : Z = 2x1 + 1x2\nSujeito a :\n6x1 + 3x2 <= 480\n2x1 + 4x2 <= 480\nTal que : x1, x2 >= 0",
    "Ex 19: Infinito e Além": "Maximizar : Z = 60x1 + 40x2\nSujeito a :\n10x1 + 10x2 <= 100\n3x1 + 7x2 <= 42\nTal que : x1, x2 >= 0",
    "Ex 20: Nutrição": "Minimizar : Z = 10x1 + 16x2\nSujeito a :\nx1 + 2x2 >= 40\n2x1 + 5x2 >= 50\nTal que : x1, x2 >= 0",
    "Ex 21: Janelas": "Maximizar : Z = 60x1 + 30x2\nSujeito a :\n6x1 + 8x2 <= 48\nx1 <= 6\nx2 <= 4\nTal que : x1, x2 >= 0",
    "Ex 22: Notebooks": "Maximizar : Z = 120x1 + 80x2\nSujeito a :\n20x1 + 10x2 <= 500\nx1 <= 40\nx2 <= 10\nTal que : x1, x2 >= 0",
    "Ex 23: Lazer": "Minimizar : Z = 14x1 + 20x2\nSujeito a :\nx1 + 2x2 >= 4\n7x1 + 6x2 >= 20\nTal que : x1, x2 >= 0",
    "Ex 24: Bijuterias": "Maximizar : Z = 50x1 + 75x2\nSujeito a :\n4x1 <= 12\n4x1 + 8x2 <= 20\n8x1 + 20x2 <= 50\nTal que : x1, x2 >= 0"
}

with st.sidebar:
    st.header("Entrada")
    sel_ex = st.selectbox("📚 Exercício:", list(exercises.keys()))
    val_ini = exercises[sel_ex] if sel_ex != "Personalizado" else "Maximizar : Z = 3x1 + 5x2\nSujeito a :\nx1 <= 4\n2x2 <= 12\nTal que : x1, x2 >= 0"
    texto_input = st.text_area("Modelo:", value=val_ini, height=300)
    btn_run = st.button("🚀 Resolver", type="primary")
    
    st.info("**Sintaxe:**\nMaximizar : Z = ...\nSujeito a :\n... <= ...\nTal que : x1, x2 >= 0")

if btn_run and texto_input:
    try:
        c, A, b, signs, opt_type = parse_problem(texto_input)
        if len(c) != 2: 
            st.error("❌ O método gráfico requer 2 variáveis.")
        else:
            # Generate Plot and Results
            fig, verts, z_opt, p_opt = generate_robust_plot(c, A, b, signs, opt_type, sel_ex)
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("1. Modelo Interpretado")
                mod_txt = format_model_text(c, A, b, signs, opt_type)
                st.code(mod_txt, language="text")
                
                if p_opt is not None:
                    st.subheader("2. Resultados")
                    st.metric("Valor Ótimo (Z*)", f"{z_opt:.4f}")
                    
                    df_v = [{"x1": f"{v[0]:.2f}", "x2": f"{v[1]:.2f}", "Z": f"{np.dot(v, c):.2f}"} for v in verts]
                    st.dataframe(df_v, use_container_width=True)
                else:
                    st.error("⚠️ Problema Inviável ou Ilimitado (Sem vértices finitos ótimos).")

            with col2:
                st.subheader("3. Gráfico")
                st.pyplot(fig)
            
            # --- DOWNLOAD ---
            st.markdown("---")
            nome_base = "Modelo_Personalizado"
            if sel_ex != "Personalizado":
                m = re.search(r'Ex (\d+)', sel_ex)
                if m: nome_base = f"Questao_{m.group(1)}"
            
            zip_name = f"{nome_base}.zip"
            txt_content = f"RELATORIO DE SOLUCAO - {nome_base}\n{'='*40}\n\n"
            txt_content += "1. MODELO MATEMATICO:\n" + mod_txt + "\n\n"
            txt_content += "2. VERTICES:\n"
            if len(verts) > 0:
                for v in verts:
                    z = np.dot(v, c)
                    mark = " (*)" if abs(z - z_opt) < 1e-5 else ""
                    txt_content += f"x=({v[0]:.2f}, {v[1]:.2f}) -> Z={z:.4f}{mark}\n"
                txt_content += f"\n3. SOLUCAO OTIMA:\nx* = ({p_opt[0]:.4f}, {p_opt[1]:.4f})\nZ* = {z_opt:.4f}"
            else:
                txt_content += "Nenhum vértice factível encontrado (Inviável)."

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            
            z_buf = io.BytesIO()
            with zipfile.ZipFile(z_buf, "w") as zf:
                zf.writestr(f"Relatorio_{nome_base}.txt", txt_content)
                zf.writestr(f"Grafico_{nome_base}.png", img_buf.getvalue())
            z_buf.seek(0)
            
            st.download_button(f"📦 Baixar ({zip_name})", data=z_buf, file_name=zip_name, mime="application/zip")

    except Exception as e: st.error(f"Erro: {str(e)}")
else:
    if not texto_input: st.info("👈 Selecione um exercício.")
