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
# CONFIGURAÇÃO
# =============================================================================
st.set_page_config(page_title="Solver PL - Gráfico", page_icon="📐", layout="wide")

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 150
})

# =============================================================================
# 1. PARSER
# =============================================================================
def ler_problema_texto(texto_entrada):
    texto_entrada = re.sub(r'(?i)(Sujeito a\s*[:]?)\s*', r'\1\n', texto_entrada)
    texto_entrada = re.sub(r'(?i)(Tal que\s*[:]?)\s*', r'\n\1\n', texto_entrada)
    
    lines = [l.strip() for l in texto_entrada.split('\n') if l.strip()]
    if not lines: raise ValueError("Texto vazio.")

    tipo_opt = None
    c, A, b, sinais = [], [], [], []
    term_pattern = re.compile(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)')
    lendo_restricoes = False
    variaveis_indices = set()

    for line in lines:
        line_lower = line.lower()
        
        # Objetivo
        if 'maximizar' in line_lower or 'minimizar' in line_lower:
            tipo_opt = 'min' if 'minimizar' in line_lower else 'max'
            if '=' not in line: raise ValueError("Objetivo deve ter '='.")
            eq = line.split('=')[1].strip()
            matches = term_pattern.findall(eq)
            if not matches: raise ValueError("Variáveis não encontradas no objetivo.")
            
            temp_c = {}
            for coeff, idx in matches:
                idx = int(idx) - 1
                variaveis_indices.add(idx)
                coeff = coeff.replace(' ', '')
                if coeff in ['', '+']: val = 1.0
                elif coeff == '-': val = -1.0
                else: val = float(coeff)
                temp_c[idx] = val
            n_vars = max(variaveis_indices) + 1 if variaveis_indices else 0
            c = np.zeros(n_vars)
            for i, v in temp_c.items(): c[i] = v
            continue

        if 'sujeito a' in line_lower: lendo_restricoes = True; continue
        if 'tal que' in line_lower: lendo_restricoes = False; continue

        if lendo_restricoes:
            sinal, sep = None, None
            if '<=' in line: sep, sinal = '<=', '<='
            elif '>=' in line: sep, sinal = '>=', '>='
            elif '=' in line: sep, sinal = '=', '='
            
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
                        if coeff in ['', '+']: k = 1.0
                        elif coeff == '-': k = -1.0
                        else: k = float(coeff)
                        row[idx] = k
                    A.append(row); b.append(b_val); sinais.append(sinal)
                except: continue

    if tipo_opt is None: raise ValueError("Objetivo não encontrado.")
    if not A: raise ValueError("Sem restrições.")
    
    # Ajuste dimensional
    max_len = max(len(c), max([len(r) for r in A]) if A else 0)
    if len(c) < max_len: 
        new = np.zeros(max_len); new[:len(c)] = c; c = new
    A_fin = [np.pad(r, (0, max_len-len(r))) if len(r)<max_len else r for r in A]
    
    return np.array(c), np.array(A_fin), np.array(b), sinais, tipo_opt

# =============================================================================
# 2. CÁLCULO
# =============================================================================
def verificar_limitacao(A, sinais):
    normais = []
    for i, row in enumerate(A):
        if sinais[i] == '<=': normais.append(row)
        elif sinais[i] == '>=': normais.append(-row)
        else: normais.append(row); normais.append(-row)
    normais.append([-1,0]); normais.append([0,-1])
    
    angs = []
    for v in normais:
        if np.linalg.norm(v) > 1e-9:
            ang = math.degrees(math.atan2(v[1], v[0])) % 360
            angs.append(ang)
    if not angs: return False
    angs.sort()
    gap = max([(angs[(i+1)%len(angs)] - angs[i]) % 360 for i in range(len(angs))])
    return gap < (180.0 - 1e-4)

def resolver_grafico(c, A, b, sinais, tipo):
    A_ext = np.vstack([A, np.eye(2)])
    b_ext = np.concatenate([b, [0,0]])
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
            if sinais[k]=='<=' and val > b[k]+1e-7: viavel=False
            elif sinais[k]=='>=' and val < b[k]-1e-7: viavel=False
            elif sinais[k]=='=' and abs(val-b[k]) > 1e-7: viavel=False
            if not viavel: break
        if viavel and not any(np.linalg.norm(p-v)<1e-7 for v in factiveis):
            factiveis.append(p)
            
    if not factiveis: return {"status": "inviavel"}
    
    verts = np.array(factiveis)
    # Ordena vértices para polígono
    if len(verts) > 2:
        centro = np.mean(verts, axis=0)
        ang = np.arctan2(verts[:,1]-centro[1], verts[:,0]-centro[0])
        verts = verts[np.argsort(ang)]
        
    z_vals = np.dot(verts, c)
    z_opt = np.min(z_vals) if tipo == 'min' else np.max(z_vals)
    idxs = np.where(np.abs(z_vals - z_opt) < 1e-7)[0]
    
    return {
        "status": "otimo",
        "vertices": verts,
        "z_vals": z_vals,
        "otimos": verts[idxs],
        "z_opt": z_opt,
        "limitada": verificar_limitacao(A, sinais)
    }

# =============================================================================
# 3. FORMATADOR DE TEXTO (SUBSTITUI LATEX)
# =============================================================================
def formatar_modelo_texto(c, A, b, sinais, tipo):
    sense = "Minimizar" if tipo == 'min' else "Maximizar"
    
    # Função Objetivo
    terms = []
    for i, v in enumerate(c):
        if abs(v) > 1e-9:
            sign = "+" if v >= 0 else "-"
            if i==0 and v>=0: sign = ""
            terms.append(f"{sign} {abs(v):.2f}x{i+1}")
    z_str = "".join(terms).strip()
    
    txt = f"{sense} : Z = {z_str}\nSujeito a :\n"
    
    # Restrições
    for i, (row, val) in enumerate(zip(A, b)):
        row_terms = []
        for j, k in enumerate(row):
            if abs(k) > 1e-9:
                sign = "+" if k>=0 else "-"
                if len(row_terms)==0 and k>=0: sign=""
                row_terms.append(f"{sign} {abs(k):.2f}x{j+1}")
        lhs = "".join(row_terms).strip()
        if not lhs: lhs="0"
        
        op = "<=" if sinais[i]=='<=' else ">=" if sinais[i]=='>=' else "="
        txt += f"  {lhs} {op} {val}\n"
        
    txt += "Tal que : x1, x2 >= 0"
    return txt

def gerar_grafico(res, A, b, titulo):
    verts, otimos = res["vertices"], res["otimos"]
    limit = max(np.max(verts)*1.5, 10) if len(verts)>0 else 10
    
    fig, ax = plt.subplots(figsize=(8,6))
    if len(verts) > 2 and res["limitada"]:
        ax.add_patch(Polygon(verts, closed=True, alpha=0.3, color='mediumseagreen', label='Região Factível'))
    elif len(verts) > 0 and not res["limitada"]:
        # Visual simples para ilimitada
        ax.fill_between([0, limit], [0,0], [limit, limit], color='mediumseagreen', alpha=0.1, label='Região Ilimitada')

    ax.scatter(verts[:,0], verts[:,1], c='k', s=30, zorder=5)
    ax.scatter(otimos[:,0], otimos[:,1], c='r', s=120, marker='*', zorder=6, label='Ótimo')
    
    x = np.linspace(0, limit, 200)
    colors = plt.cm.tab10(np.linspace(0,1,len(A)))
    for i, (row, val) in enumerate(zip(A, b)):
        if abs(row[1]) > 1e-6:
            y = (val - row[0]*x)/row[1]
            ax.plot(x, y, label=f'R{i+1}', color=colors[i])
        else:
            ax.vlines(val/row[0], 0, limit, label=f'R{i+1}', color=colors[i])
            
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(titulo); ax.grid(True, alpha=0.3); ax.legend()
    return fig

# =============================================================================
# 4. INTERFACE
# =============================================================================
st.title("📊 Solver PL - Método Gráfico")
st.markdown("Resolução de problemas de PL com 2 variáveis (Questões 17-24).")

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
    st.header("Entrada")
    sel_ex = st.selectbox("📚 Exercício:", list(exercicios.keys()))
    val_ini = exercicios[sel_ex] if sel_ex != "Personalizado" else "Maximizar : Z = 3x1 + 5x2\nSujeito a :\nx1 <= 4\n2x2 <= 12\nTal que : x1, x2 >= 0"
    texto_input = st.text_area("Modelo:", value=val_ini, height=300)
    btn_run = st.button("🚀 Resolver", type="primary")
    
    st.info("**Sintaxe:**\nMaximizar : Z = ...\nSujeito a :\n... <= ...\nTal que : x1, x2 >= 0")

if btn_run and texto_input:
    try:
        c, A, b, sinais, tipo = ler_problema_texto(texto_input)
        if len(c) != 2: 
            st.error("❌ O método gráfico requer 2 variáveis.")
        else:
            res = resolver_grafico(c, A, b, sinais, tipo)
            
            if res["status"] == "inviavel":
                st.warning("⚠️ Problema Inviável.")
            else:
                col1, col2 = st.columns([1, 1.5])
                
                # --- RESULTADOS EM TEXTO PURO ---
                with col1:
                    st.subheader("1. Modelo Interpretado")
                    # Exibe o modelo formatado como código (texto puro)
                    modelo_fmt = formatar_modelo_texto(c, A, b, sinais, tipo)
                    st.code(modelo_fmt, language="text")
                    
                    st.subheader("2. Resultados")
                    st.metric("Valor Ótimo (Z*)", f"{res['z_opt']:.4f}")
                    
                    # Tabela Simples
                    df_v = [{"x1": f"{v[0]:.2f}", "x2": f"{v[1]:.2f}", "Z": f"{z:.2f}"} for v, z in zip(res['vertices'], res['z_vals'])]
                    st.dataframe(df_v, use_container_width=True)

                # --- GRÁFICO ---
                with col2:
                    st.subheader("3. Gráfico")
                    fig = gerar_grafico(res, A, b, sel_ex)
                    st.pyplot(fig)
                
                # --- DOWNLOAD ---
                st.markdown("---")
                nome_base = "Modelo_Personalizado"
                if sel_ex != "Personalizado":
                    m = re.search(r'Ex (\d+)', sel_ex)
                    if m: nome_base = f"Questao_{m.group(1)}"
                
                zip_name = f"{nome_base}.zip"
                txt_content = f"RELATORIO DE SOLUCAO - {nome_base}\n{'='*40}\n\n"
                txt_content += "1. MODELO MATEMATICO:\n" + modelo_fmt + "\n\n"
                txt_content += "2. VERTICES:\n"
                for v, z in zip(res['vertices'], res['z_vals']):
                    mark = " (*)" if np.abs(z - res['z_opt']) < 1e-7 else ""
                    txt_content += f"x=({v[0]:.2f}, {v[1]:.2f}) -> Z={z:.4f}{mark}\n"
                
                v_opt = res['otimos'][0]
                txt_content += f"\n3. SOLUCAO OTIMA:\nx* = ({v_opt[0]:.4f}, {v_opt[1]:.4f})\nZ* = {res['z_opt']:.4f}"

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
