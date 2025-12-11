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
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Solver Gráfico - Trabalho Final FPO",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo Matplotlib Profissional
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

# =============================================================================
# 1. PARSING ROBUSTO (ACEITA COPIAR E COLAR DO GABARITO)
# =============================================================================

def ler_problema_texto(texto_entrada):
    """
    Interpreta o texto do problema.
    Projetado para aceitar o formato exato do gabarito/lista de exercícios.
    """
    # 1. Limpeza Prévia: Garante que 'Sujeito a' e 'Tal que' tenham quebra de linha
    # Isso resolve o problema de colar "Sujeito a : x1..." tudo na mesma linha
    texto_entrada = re.sub(r'(?i)(Sujeito a\s*[:]?)\s*', r'\1\n', texto_entrada)
    texto_entrada = re.sub(r'(?i)(Tal que\s*[:]?)\s*', r'\n\1\n', texto_entrada)
    
    lines = [l.strip() for l in texto_entrada.split('\n') if l.strip()]
    
    tipo_opt = None
    c = []
    A = []
    b = []
    sinais = []
    
    lendo_restricoes = False
    # Regex flexível para capturar coeficientes (ex: "x1", "-x2", "2.5x1", "+ 3x2")
    term_pattern = re.compile(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)')
    variaveis_indices = set()

    for line in lines:
        line_lower = line.lower()
        
        # --- Identificação do Objetivo (Maximizar/Minimizar) ---
        if 'maximizar' in line_lower or 'minimizar' in line_lower:
            tipo_opt = 'min' if 'minimizar' in line_lower else 'max'
            
            if '=' not in line:
                # Se o usuário digitou "Maximizar 2x1+..." sem o "Z ="
                raise ValueError("A função objetivo deve conter 'Z =' ou similar (ex: Maximizar : Z = ...)")
            
            # Pega o que está depois do igual
            eq_part = line.split('=')[1].strip()
            matches = term_pattern.findall(eq_part)
            
            if not matches:
                raise ValueError("Não foi possível identificar as variáveis na função objetivo (ex: 10x1 + 16x2).")

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
            for idx, val in temp_c.items():
                c[idx] = val
            continue

        # --- Controle de Seções ---
        if 'sujeito a' in line_lower:
            lendo_restricoes = True
            continue
        if 'tal que' in line_lower:
            lendo_restricoes = False
            continue

        # --- Leitura das Restrições ---
        if lendo_restricoes:
            # Tenta identificar o operador lógico
            sinal = None
            sep = None
            
            if '<=' in line: sep, sinal = '<=', '<='
            elif '>=' in line: sep, sinal = '>=', '>='
            elif '≤' in line: sep, sinal = '≤', '<='  # Suporte a caractere especial
            elif '≥' in line: sep, sinal = '≥', '>='  # Suporte a caractere especial
            elif '=' in line: sep, sinal = '=', '='
            
            if sep:
                parts = line.split(sep)
                lhs = parts[0]
                rhs = parts[1]
                
                try:
                    # Limpa comentários ou textos após o número (ex: "40 (Restrição 1)")
                    rhs_clean = re.split(r'\s', rhs.strip())[0]
                    b_val = float(rhs_clean)
                except ValueError:
                    continue # Ignora linha se não tiver um número válido à direita
                    
                b.append(b_val)
                sinais.append(sinal)
                
                matches = term_pattern.findall(lhs)
                row = np.zeros(len(c)) if len(c) > 0 else np.zeros(2) # Fallback seguro
                
                for coeff_str, idx_str in matches:
                    idx = int(idx_str) - 1
                    # Expande vetor se encontrar variável nova (ex: x3 apareceu só na restrição)
                    if idx >= len(row):
                        nova_col = np.zeros(idx + 1)
                        nova_col[:len(row)] = row
                        row = nova_col
                        
                    coeff_str = coeff_str.replace(' ', '')
                    if coeff_str in ['', '+']: coeff = 1.0
                    elif coeff_str == '-': coeff = -1.0
                    else: coeff = float(coeff_str)
                    row[idx] = coeff
                A.append(row)

    # Validações Finais
    if tipo_opt is None:
        raise ValueError("Não encontrei 'Maximizar' ou 'Minimizar'. Verifique o cabeçalho.")
    if len(A) == 0:
        raise ValueError("Nenhuma restrição foi encontrada. Verifique se escreveu 'Sujeito a :'.")

    # Ajuste de dimensões (caso vetor c seja menor que A)
    max_len = max(len(c), max([len(row) for row in A]) if A else 0)
    if len(c) < max_len:
        c_new = np.zeros(max_len)
        c_new[:len(c)] = c
        c = c_new
    
    A_final = []
    for row in A:
        if len(row) < max_len:
            row_new = np.zeros(max_len)
            row_new[:len(row)] = row
            A_final.append(row_new)
        else:
            A_final.append(row)
            
    return np.array(c), np.array(A_final), np.array(b), sinais, tipo_opt

# =============================================================================
# 2. LÓGICA MATEMÁTICA E CÁLCULO
# =============================================================================

def verificar_limitacao_region(A, sinais):
    """Verifica geometricamente se a região é fechada."""
    normais = []
    for i, row in enumerate(A):
        if sinais[i] == '<=': normais.append(row)
        elif sinais[i] == '>=': normais.append(-row)
        elif sinais[i] == '=':
            normais.append(row)
            normais.append(-row)
            
    normais.append(np.array([-1.0, 0.0])) 
    normais.append(np.array([0.0, -1.0]))
    
    angulos = []
    for v in normais:
        if np.linalg.norm(v) > 1e-9:
            ang = math.atan2(v[1], v[0])
            ang_deg = math.degrees(ang) % 360
            angulos.append(ang_deg)
            
    if not angulos: return False 
    angulos.sort()
    max_gap = 0
    for i in range(len(angulos)):
        atual = angulos[i]
        proximo = angulos[(i + 1) % len(angulos)]
        gap = (proximo - atual) % 360
        if gap > max_gap: max_gap = gap
            
    return max_gap < (180.0 - 1e-4)

def resolver_sistema_grafico(c, A, b, sinais, tipo):
    """Resolve o PL encontrando vértices e otimizando."""
    A_ext = np.vstack([A, np.eye(2)]) 
    b_ext = np.concatenate([b, [0, 0]])
    pontos = []
    indices = range(len(b_ext))
    
    for i, j in itertools.combinations(indices, 2):
        m = np.array([A_ext[i], A_ext[j]])
        rhs = np.array([b_ext[i], b_ext[j]])
        if np.abs(np.linalg.det(m)) > 1e-10:
            try: pontos.append(np.linalg.solve(m, rhs))
            except: pass
            
    vertices_factiveis = []
    for p in pontos:
        if p[0] < -1e-7 or p[1] < -1e-7: continue
        viavel = True
        for k, row in enumerate(A):
            val = np.dot(row, p)
            if sinais[k] == '<=' and val > b[k] + 1e-7: viavel = False
            elif sinais[k] == '>=' and val < b[k] - 1e-7: viavel = False
            elif sinais[k] == '=' and abs(val - b[k]) > 1e-7: viavel = False
            if not viavel: break
        
        if viavel:
            if not any(np.linalg.norm(p - v) < 1e-7 for v in vertices_factiveis):
                vertices_factiveis.append(p)
    
    if not vertices_factiveis:
        return {"status": "inviavel"}

    vertices = np.array(vertices_factiveis)
    vertices = vertices[np.lexsort((vertices[:,1], vertices[:,0]))]
    z_vals = np.dot(vertices, c)
    
    if tipo == 'min':
        z_opt = np.min(z_vals)
        indices_opt = np.where(np.abs(z_vals - z_opt) < 1e-7)[0]
    else:
        z_opt = np.max(z_vals)
        indices_opt = np.where(np.abs(z_vals - z_opt) < 1e-7)[0]
    
    otimos = vertices[indices_opt]
    regiao_limitada = verificar_limitacao_region(A, sinais)
    
    return {
        "status": "otimo",
        "vertices": vertices,
        "z_vals": z_vals,
        "otimos": otimos,
        "z_opt": z_opt,
        "eh_ilimitada": not regiao_limitada,
        "multiplos_otimos": len(otimos) > 1
    }

# =============================================================================
# 3. VISUALIZAÇÃO E RELATÓRIOS
# =============================================================================

def gerar_texto_latex(c, A, b, sinais, tipo):
    sinal_z = "Min" if tipo == 'min' else "Max"
    termos = []
    for i, val in enumerate(c):
        if abs(val) > 1e-9:
            op = "+" if val >= 0 else "-"
            v = abs(val)
            v_str = f"{v:.2f}" if abs(v - 1.0) > 1e-9 else ""
            termos.append(f"{op} {v_str}x_{{{i+1}}}")
    z_eq = " ".join(termos).strip().lstrip("+")
    
    latex = f"\\text{{{sinal_z}}} \\ Z = {z_eq} \\\\"
    latex += "\\text{Sujeito a:} \\\\"
    latex += "\\begin{cases} "
    for i, (row, val) in enumerate(zip(A, b)):
        t_row = []
        for j, coeff in enumerate(row):
            if abs(coeff) > 1e-9:
                op = "+" if coeff >= 0 else "-"
                v = abs(coeff)
                v_str = f"{v:.2f}" if abs(v - 1.0) > 1e-9 else ""
                t_row.append(f"{op} {v_str}x_{{{j+1}}}")
        lhs = " ".join(t_row).strip().lstrip("+")
        if not lhs: lhs = "0"
        s_lat = "\\le" if sinais[i] == '<=' else "\\ge" if sinais[i] == '>=' else "="
        latex += f"{lhs} {s_lat} {val} \\\\"
    latex += "x_1, x_2 \\ge 0"
    latex += "\\end{cases}"
    return latex

def gerar_grafico(res, A, b, tipo, titulo):
    vertices = res["vertices"]
    otimos = res["otimos"]
    eh_ilimitada = res["eh_ilimitada"]
    
    max_x = np.max(vertices[:, 0]) if len(vertices) > 0 else 0
    max_y = np.max(vertices[:, 1]) if len(vertices) > 0 else 0
    
    int_x = [b[i]/row[0] for i, row in enumerate(A) if abs(row[0]) > 1e-6]
    int_y = [b[i]/row[1] for i, row in enumerate(A) if abs(row[1]) > 1e-6]
    limite_filtro = max(max_x, max_y) * 4 + 20
    int_x = [x for x in int_x if 0 < x < limite_filtro]
    int_y = [y for y in int_y if 0 < y < limite_filtro]
    max_int = max(max(int_x) if int_x else 0, max(int_y) if int_y else 0)
    
    limite_zoom = max(max_x, max_y, max_int) * 1.2
    if limite_zoom < 10: limite_zoom = 10
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pontos_para_poligono = vertices.copy()
    if eh_ilimitada:
        pontos_fundo = [[limite_zoom, 0], [limite_zoom, limite_zoom], [0, limite_zoom]]
        todos_pontos = np.vstack([vertices, pontos_fundo])
        centro = np.mean(todos_pontos, axis=0)
        angulos = np.arctan2(todos_pontos[:,1] - centro[1], todos_pontos[:,0] - centro[0])
        pontos_para_poligono = todos_pontos[np.argsort(angulos)]
        cor_fundo = 'mediumseagreen'
        lbl = 'Região Factível (Ilimitada)'
        extra_txt = "ILIMITADA ↗"
    else:
        if len(vertices) > 2:
            centro = np.mean(vertices, axis=0)
            angulos = np.arctan2(vertices[:,1] - centro[1], vertices[:,0] - centro[0])
            pontos_para_poligono = vertices[np.argsort(angulos)]
        cor_fundo = 'mediumseagreen'
        lbl = 'Região Factível'
        extra_txt = None

    poly = Polygon(pontos_para_poligono, closed=True, alpha=0.3, color=cor_fundo, label=lbl)
    ax.add_patch(poly)
    
    if extra_txt:
        ax.text(limite_zoom*0.95, limite_zoom*0.95, extra_txt, 
                ha='right', va='top', color='darkgreen', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.scatter(vertices[:,0], vertices[:,1], color='black', s=30, zorder=5)
    ax.scatter(otimos[:,0], otimos[:,1], color='red', s=120, marker='*', zorder=6, label='Ótimo Global')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(A)))
    vals = np.linspace(0, limite_zoom, 200)
    for i, (row, val) in enumerate(zip(A, b)):
        color = colors[i]
        if abs(row[1]) > 1e-6:
            y_vals = (val - row[0]*vals) / row[1]
            mask = (y_vals >= -limite_zoom*0.1) & (y_vals <= limite_zoom*1.2)
            if np.any(mask):
                ax.plot(vals[mask], y_vals[mask], label=f'R{i+1}', color=color)
        else:
            x_v = val / row[0]
            if 0 <= x_v <= limite_zoom:
                ax.vlines(x_v, 0, limite_zoom, label=f'R{i+1}', color=color)

    ax.set_xlim(0, limite_zoom)
    ax.set_ylim(0, limite_zoom)
    ax.set_xlabel('Variável $x_1$')
    ax.set_ylabel('Variável $x_2$')
    ax.set_title(f"{titulo}\nOtimização: {tipo.upper()}", pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), borderaxespad=0)
    
    return fig

def gerar_relatorio_markdown(texto_original, res, tipo):
    md = "# Relatório de Solução - Método Gráfico\n\n"
    md += "## 1. Definição do Problema\n"
    md += "```text\n" + texto_original + "\n```\n\n"
    md += "## 2. Análise dos Vértices\n"
    md += "| Vértice ($x_1, x_2$) | Valor de Z | Status |\n| :--- | :--- | :--- |\n"
    z_opt = res['z_opt']
    for v, z in zip(res['vertices'], res['z_vals']):
        tag = "⭐ ÓTIMO" if np.abs(z - z_opt) < 1e-7 else ""
        md += f"| ({v[0]:.2f}, {v[1]:.2f}) | {z:.4f} | {tag} |\n"
    md += "\n## 3. Conclusão\n"
    if res['multiplos_otimos']:
        md += "**Obs:** Identificada multiplicidade de soluções (segmento de reta).\n\n"
    v_opt = res['otimos'][0]
    md += f"Solução ótima em **x1 = {v_opt[0]:.4f}**, **x2 = {v_opt[1]:.4f}**\n"
    md += f"Valor objetivo **Z = {z_opt:.4f}**."
    return md

# =============================================================================
# 4. INTERFACE GRÁFICA DO APP
# =============================================================================

st.title("📊 Solver Gráfico de Programação Linear")
st.markdown("""
Ferramenta para resolução de problemas de PL com **2 variáveis**.
Insira o modelo matemático abaixo seguindo o padrão da disciplina.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Entrada de Dados")
    
    # Exemplo padrão que aparece ao carregar (formato correto)
    exemplo = """Minimizar : Z = 14x1 + 20x2
Sujeito a : 
x1 + 2x2 >= 4
7x1 + 6x2 >= 20
Tal que : 
x1, x2 >= 0"""
    
    texto_input = st.text_area("Formulação do Modelo:", value=exemplo, height=300)
    btn_resolver = st.button("🚀 Resolver", type="primary")
    
    st.info("""
    **Formato Aceito:**
    - Use `Maximizar :` ou `Minimizar :`
    - Função Z deve ter `Z =`
    - Restrições abaixo de `Sujeito a :`
    - Use `x1` e `x2` como variáveis
    """)

# --- Lógica de Resolução ---
if btn_resolver:
    # 1. Validação de Campo Vazio
    if not texto_input.strip():
        st.warning("⚠️ A caixa de texto está vazia. Por favor, insira o problema.")
    else:
        try:
            # 2. Parsing e Validação
            c, A, b, sinais, tipo = ler_problema_texto(texto_input)
            
            if len(c) != 2:
                st.error("❌ Erro de Dimensão: O método gráfico suporta **estritamente 2 variáveis** ($x_1$ e $x_2$).")
            else:
                # 3. Resolução
                res = resolver_sistema_grafico(c, A, b, sinais, tipo)
                
                if res["status"] == "inviavel":
                    st.warning("⚠️ **Problema Inviável:** Não existe região factível que atenda a todas as restrições.")
                else:
                    # Layout de Colunas
                    col1, col2 = st.columns([1, 1.5])
                    
                    with col1:
                        st.subheader("1. Definição Matemática")
                        st.latex(gerar_texto_latex(c, A, b, sinais, tipo))
                        
                        st.subheader("2. Vértices Encontrados")
                        data_v = []
                        for v, z in zip(res['vertices'], res['z_vals']):
                            mark = "⭐" if np.abs(z - res['z_opt']) < 1e-7 else ""
                            data_v.append({"x1": f"{v[0]:.2f}", "x2": f"{v[1]:.2f}", "Z": f"{z:.2f}", "Status": mark})
                        st.dataframe(data_v, use_container_width=True)
                        
                        st.success(f"**Resultado Ótimo: Z* = {res['z_opt']:.4f}**")
                        if res['multiplos_otimos']:
                            st.info("ℹ️ Nota: Existem múltiplas soluções ótimas.")

                    with col2:
                        st.subheader("3. Análise Gráfica")
                        fig = gerar_grafico(res, A, b, tipo, "Região Factível")
                        st.pyplot(fig)
                    
                    # --- Exportação ---
                    st.markdown("---")
                    st.subheader("📥 Download dos Resultados")
                    
                    relatorio = gerar_relatorio_markdown(texto_input, res, tipo)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight')
                    img_buf.seek(0)
                    
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        zf.writestr("relatorio_solucao.md", relatorio)
                        zf.writestr("grafico_solucao.png", img_buf.getvalue())
                    zip_buf.seek(0)
                    
                    st.download_button(
                        label="📦 Baixar Relatório + Gráfico (.zip)",
                        data=zip_buf,
                        file_name="solucao_fpo.zip",
                        mime="application/zip"
                    )

        except ValueError as ve:
            # Erros específicos de formatação (ex: faltou "Z =")
            st.error(f"❌ Erro de Formatação: {str(ve)}")
            st.markdown("""
            **Exemplo Correto:**
            ```text
            Maximizar : Z = 3x1 + 5x2
            Sujeito a :
            x1 <= 4
            x2 <= 6
            Tal que :
            x1, x2 >= 0
            ```
            """)
        except Exception as e:
            # Erros genéricos de execução
            st.error(f"❌ Erro inesperado: {str(e)}")
else:
    st.info("👈 Utilize a barra lateral para inserir seu problema e clique em **Resolver**.")