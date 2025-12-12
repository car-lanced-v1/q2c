import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import re
import io
import zipfile
from matplotlib.patches import Polygon

# =============================================================================
# CONFIGURAÇÃO DO AMBIENTE
# =============================================================================
st.set_page_config(
    page_title="Solver PL - Método Gráfico",
    page_icon="📐",
    layout="wide"
)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

# =============================================================================
# 1. PARSER (MANTIDO)
# =============================================================================
def ler_problema_texto(texto_entrada):
    texto_entrada = re.sub(r'(?i)(Sujeito a\s*[:]?)\s*', r'\1\n', texto_entrada)
    texto_entrada = re.sub(r'(?i)(Tal que\s*[:]?)\s*', r'\n\1\n', texto_entrada)
    
    lines = [l.strip() for l in texto_entrada.split('\n') if l.strip()]
    tipo_opt = None
    c, A, b, sinais = [], [], [], []
    
    lendo_restricoes = False
    term_pattern = re.compile(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)')
    variaveis_indices = set()

    for line in lines:
        line_lower = line.lower()
        
        if 'maximizar' in line_lower or 'minimizar' in line_lower:
            tipo_opt = 'min' if 'minimizar' in line_lower else 'max'
            if '=' in line:
                eq_part = line.split('=')[1].strip()
                matches = term_pattern.findall(eq_part)
                if not matches: raise ValueError("Formato da função objetivo inválido.")
                temp_c = {}
                for coeff_str, idx_str in matches:
                    idx = int(idx_str) - 1
                    variaveis_indices.add(idx)
                    coeff_str = coeff_str.replace(' ', '')
                    if coeff_str in ['', '+']: coeff = 1.0
                    elif coeff_str == '-': coeff = -1.0
                    else: coeff = float(coeff_str)
                    temp_c[idx] = coeff
                n_vars = max(variaveis_indices) + 1 if variaveis_indices else 0
                c = np.zeros(n_vars)
                for idx, val in temp_c.items(): c[idx] = val
            continue

        if 'sujeito a' in line_lower:
            lendo_restricoes = True; continue
        if 'tal que' in line_lower:
            lendo_restricoes = False; continue

        if lendo_restricoes:
            sinal, sep = None, None
            if '<=' in line: sep, sinal = '<=', '<='
            elif '>=' in line: sep, sinal = '>=', '>='
            elif '=' in line: sep, sinal = '=', '='
            
            if sep:
                parts = line.split(sep)
                try:
                    b_val = float(re.split(r'\s', parts[1].strip())[0])
                    b.append(b_val); sinais.append(sinal)
                    row = np.zeros(len(c)) if len(c) > 0 else np.zeros(2)
                    for coeff_str, idx_str in term_pattern.findall(parts[0]):
                        idx = int(idx_str) - 1
                        if idx >= len(row):
                            nova = np.zeros(idx + 1); nova[:len(row)] = row; row = nova
                        coeff_str = coeff_str.replace(' ', '')
                        if coeff_str in ['', '+']: k = 1.0
                        elif coeff_str == '-': k = -1.0
                        else: k = float(coeff_str)
                        row[idx] = k
                    A.append(row)
                except: continue

    if tipo_opt is None: raise ValueError("Objetivo não encontrado.")
    if len(A) == 0: raise ValueError("Sem restrições.")

    max_len = max(len(c), max([len(row) for row in A]) if A else 0)
    if len(c) < max_len:
        new_c = np.zeros(max_len); new_c[:len(c)] = c; c = new_c
    A_final = [np.pad(row, (0, max_len - len(row))) if len(row) < max_len else row for row in A]
            
    return np.array(c), np.array(A_final), np.array(b), sinais, tipo_opt

# =============================================================================
# 2. CÁLCULO (MANTIDO)
# =============================================================================
def verificar_limitacao_region(A, sinais):
    normais = []
    for i, row in enumerate(A):
        if sinais[i] == '<=': normais.append(row)
        elif sinais[i] == '>=': normais.append(-row)
        else: normais.append(row); normais.append(-row)
    normais.append([-1.0, 0.0]); normais.append([0.0, -1.0])
    
    angulos = []
    for v in normais:
        if np.linalg.norm(v) > 1e-9:
            ang = math.degrees(math.atan2(v[1], v[0])) % 360
            angulos.append(ang)
    if not angulos: return False 
    angulos.sort()
    max_gap = max([(angulos[(i+1)%len(angulos)] - angulos[i]) % 360 for i in range(len(angulos))])
    return max_gap < (180.0 - 1e-4)

def resolver_sistema_grafico(c, A, b, sinais, tipo):
    A_ext = np.vstack([A, np.eye(2)]) 
    b_ext = np.concatenate([b, [0, 0]])
    pontos = []
    for i, j in itertools.combinations(range(len(b_ext)), 2):
        m = np.array([A_ext[i], A_ext[j]])
        rhs = np.array([b_ext[i], b_ext[j]])
        if np.abs(np.linalg.det(m)) > 1e-10:
            try: pontos.append(np.linalg.solve(m, rhs))
            except: pass
            
    factiveis = []
    for p in pontos:
        if p[0] < -1e-7 or p[1] < -1e-7: continue
        viavel = True
        for k, row in enumerate(A):
            val = np.dot(row, p)
            if sinais[k] == '<=' and val > b[k] + 1e-7: viavel = False
            elif sinais[k] == '>=' and val < b[k] - 1e-7: viavel = False
            elif sinais[k] == '=' and abs(val - b[k]) > 1e-7: viavel = False
            if not viavel: break
        if viavel and not any(np.linalg.norm(p - v) < 1e-7 for v in factiveis):
            factiveis.append(p)
    
    if not factiveis: return {"status": "inviavel"}

    vertices = np.array(factiveis)
    vertices = vertices[np.lexsort((vertices[:,1], vertices[:,0]))]
    z_vals = np.dot(vertices, c)
    z_opt = np.min(z_vals) if tipo == 'min' else np.max(z_vals)
    indices_opt = np.where(np.abs(z_vals - z_opt) < 1e-7)[0]
    
    return {
        "status": "otimo",
        "vertices": vertices,
        "z_vals": z_vals,
        "otimos": vertices[indices_opt],
        "z_opt": z_opt,
        "eh_ilimitada": not verificar_limitacao_region(A, sinais),
        "multiplos_otimos": len(indices_opt) > 1
    }

# =============================================================================
# 3. VISUALIZAÇÃO
# =============================================================================
def gerar_texto_latex(c, A, b, sinais, tipo):
    sinal_z = "Min" if tipo == 'min' else "Max"
    termos = [f"{'+' if v>=0 else '-'} {abs(v):.2f}x_{{{i+1}}}" for i, v in enumerate(c) if abs(v)>1e-9]
    z_eq = " ".join(termos).strip().lstrip("+")
    latex = f"\\text{{{sinal_z}}} \\ Z = {z_eq} \\\\ \\text{Sujeito a:} \\\\ \\begin{{cases}} "
    for i, (row, val) in enumerate(zip(A, b)):
        lhs = " ".join([f"{'+' if k>=0 else '-'} {abs(k):.2f}x_{{{j+1}}}" for j, k in enumerate(row) if abs(k)>1e-9]).strip().lstrip("+")
        op = "\\le" if sinais[i] == '<=' else "\\ge" if sinais[i] == '>=' else "="
        latex += f"{lhs} {op} {val} \\\\"
    return latex + "x_1, x_2 \\ge 0 \\end{cases}"

def gerar_grafico(res, A, b, titulo):
    vertices, otimos = res["vertices"], res["otimos"]
    limit = max(np.max(vertices) * 1.5, 10) if len(vertices) > 0 else 10
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(vertices) > 2:
        c = np.mean(vertices, axis=0)
        ang = np.arctan2(vertices[:,1] - c[1], vertices[:,0] - c[0])
        ax.add_patch(Polygon(vertices[np.argsort(ang)], closed=True, alpha=0.3, color='mediumseagreen', label='Região Factível'))

    ax.scatter(vertices[:,0], vertices[:,1], color='black', s=30, zorder=5)
    ax.scatter(otimos[:,0], otimos[:,1], color='red', s=120, marker='*', zorder=6, label='Ótimo')
    
    x = np.linspace(0, limit, 200)
    colors = plt.cm.tab10(np.linspace(0, 1, len(A)))
    for i, (row, val) in enumerate(zip(A, b)):
        if abs(row[1]) > 1e-6:
            y = (val - row[0]*x) / row[1]
            ax.plot(x, y, label=f'R{i+1}', color=colors[i])
        else:
            ax.vlines(val/row[0], 0, limit, label=f'R{i+1}', color=colors[i])

    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(titulo); ax.grid(True, alpha=0.3); ax.legend()
    return fig

# =============================================================================
# 4. INTERFACE
# =============================================================================
st.title("📊 Solver de Programação Linear (Método Gráfico)")
st.markdown("Ferramenta para resolução e análise gráfica dos Exercícios 17-24.")

exercicios = {
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
    st.header("Entrada de Dados")
    sel_ex = st.selectbox("📚 Carregar Exercício:", list(exercicios.keys()))
    texto_input = st.text_area("Modelo:", value=exercicios[sel_ex] if sel_ex != "Personalizado" else """Maximizar : Z = 3x1 + 5x2\nSujeito a :\nx1 <= 4\n2x2 <= 12\nTal que : x1, x2 >= 0""", height=300)
    btn_resolver = st.button("🚀 Resolver", type="primary")

if btn_resolver and texto_input:
    try:
        c, A, b, sinais, tipo = ler_problema_texto(texto_input)
        if len(c) != 2: st.error("❌ O método gráfico requer exatamente 2 variáveis.")
        else:
            res = resolver_sistema_grafico(c, A, b, sinais, tipo)
            if res["status"] == "inviavel": st.warning("⚠️ Problema Inviável.")
            else:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.subheader("1. Resultados")
                    st.latex(gerar_texto_latex(c, A, b, sinais, tipo))
                    st.dataframe([{"x1": f"{v[0]:.2f}", "x2": f"{v[1]:.2f}", "Z": f"{z:.2f}"} for v, z in zip(res['vertices'], res['z_vals'])], use_container_width=True)
                    st.success(f"**Z* = {res['z_opt']:.4f}**")
                
                with col2:
                    st.subheader("2. Gráfico")
                    fig = gerar_grafico(res, A, b, sel_ex if sel_ex != "Personalizado" else "Solução Gráfica")
                    st.pyplot(fig)
                
                # --- LÓGICA DE DOWNLOAD PERSONALIZADO ---
                st.markdown("---")
                
                # 1. Definir Nomes de Arquivos com base na seleção
                nome_base = "Modelo_Personalizado"
                match = re.search(r'Ex (\d+)', sel_ex)
                if match:
                    num = match.group(1)
                    nome_base = f"Questao_{num}"
                
                zip_filename = f"{nome_base}.zip"
                txt_filename = f"Relatorio_{nome_base}.txt"
                img_filename = f"Grafico_{nome_base}.png"

                # 2. Gerar Conteúdo do Relatório em Texto
                relatorio_txt = f"RELATÓRIO DE SOLUÇÃO - {sel_ex}\n"
                relatorio_txt += "="*50 + "\n\n"
                relatorio_txt += "1. DEFINIÇÃO DO PROBLEMA\n"
                relatorio_txt += texto_input + "\n\n"
                relatorio_txt += "2. VÉRTICES ENCONTRADOS\n"
                relatorio_txt += f"{'x1':<10} | {'x2':<10} | {'Valor Z':<15}\n"
                relatorio_txt += "-"*40 + "\n"
                for v, z in zip(res['vertices'], res['z_vals']):
                    mark = " (*)" if np.abs(z - res['z_opt']) < 1e-7 else ""
                    relatorio_txt += f"{v[0]:<10.2f} | {v[1]:<10.2f} | {z:<15.4f}{mark}\n"
                relatorio_txt += "\n3. SOLUÇÃO ÓTIMA\n"
                v_opt = res['otimos'][0]
                relatorio_txt += f"x* = ({v_opt[0]:.4f}, {v_opt[1]:.4f})\n"
                relatorio_txt += f"Z* = {res['z_opt']:.4f}\n"

                # 3. Preparar Buffers
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    zf.writestr(txt_filename, relatorio_txt)
                    zf.writestr(img_filename, img_buf.getvalue())
                zip_buf.seek(0)
                
                st.download_button(
                    label=f"📦 Baixar Resultados ({zip_filename})",
                    data=zip_buf,
                    file_name=zip_filename,
                    mime="application/zip"
                )

    except Exception as e: st.error(f"Erro: {str(e)}")
else:
    if not texto_input: st.info("👈 Selecione um exercício para começar.")
