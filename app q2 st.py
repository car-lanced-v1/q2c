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
# CONFIGURAÇÃO DA PÁGINA E ESTILO ACADÊMICO
# =============================================================================
st.set_page_config(
    page_title="Solver Gráfico de PL - Mestrado PO",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo Matplotlib para publicação
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

# =============================================================================
# 1. MÓDULO DE PARSING E LÓGICA MATEMÁTICA
# =============================================================================

def ler_problema_texto(texto_entrada):
    """
    Interpreta o texto de entrada e converte em matrizes para processamento.
    Utiliza Regex para capturar coeficientes e variáveis.
    """
    lines = [l.strip() for l in texto_entrada.split('\n') if l.strip()]
    tipo_opt = 'max'
    c = []
    A = []
    b = []
    sinais = []
    
    lendo_restricoes = False
    
    # Regex para capturar: (sinal?)(número?)(x)(índice)
    term_pattern = re.compile(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)')
    variaveis_indices = set()

    for line in lines:
        line_lower = line.lower()
        
        # Identificação da Função Objetivo
        if 'maximizar' in line_lower or 'minimizar' in line_lower:
            tipo_opt = 'min' if 'minimizar' in line_lower else 'max'
            if '=' not in line:
                raise ValueError("A função objetivo deve conter '=' (ex: Z = ...)")
            
            eq_part = line.split('=')[1].strip()
            matches = term_pattern.findall(eq_part)
            
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

        # Controle de Fluxo
        if 'sujeito a' in line_lower:
            lendo_restricoes = True
            continue
        if 'tal que' in line_lower:
            lendo_restricoes = False
            continue

        # Leitura das Restrições
        if lendo_restricoes:
            if '<=' in line: sep, sinal = '<=', '<='
            elif '>=' in line: sep, sinal = '>=', '>='
            elif '=' in line: sep, sinal = '=', '='
            else: continue 
            
            lhs, rhs = line.split(sep)
            
            # Sanitização do RHS (remover comentários ou caracteres estranhos)
            try:
                b_val = float(rhs.strip())
            except ValueError:
                raise ValueError(f"Valor do lado direito inválido na linha: {line}")
                
            b.append(b_val)
            sinais.append(sinal)
            
            matches = term_pattern.findall(lhs)
            row = np.zeros(len(c))
            for coeff_str, idx_str in matches:
                idx = int(idx_str) - 1
                if idx < len(c):
                    coeff_str = coeff_str.replace(' ', '')
                    if coeff_str in ['', '+']: coeff = 1.0
                    elif coeff_str == '-': coeff = -1.0
                    else: coeff = float(coeff_str)
                    row[idx] = coeff
            A.append(row)

    return np.array(c), np.array(A), np.array(b), sinais, tipo_opt

def verificar_limitacao_region(A, sinais):
    """
    Verifica se a região factível é LIMITADA (polígono fechado) analisando
    o cone gerado pelos vetores normais das restrições.
    """
    normais = []
    for i, row in enumerate(A):
        if sinais[i] == '<=': normais.append(row)
        elif sinais[i] == '>=': normais.append(-row)
        elif sinais[i] == '=':
            normais.append(row)
            normais.append(-row)
            
    # Restrições de não-negatividade
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
    """
    Núcleo matemático: Encontra interseções, filtra viabilidade e otimiza.
    Retorna dicionário com resultados.
    """
    # 1. Encontrar Interseções
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
            
    # 2. Filtrar Factibilidade
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

    # 3. Otimização e Ordenação
    vertices = np.array(vertices_factiveis)
    vertices = vertices[np.lexsort((vertices[:,1], vertices[:,0]))] # Ordenação inicial
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
# 2. MÓDULO DE VISUALIZAÇÃO E RELATÓRIOS
# =============================================================================

def gerar_texto_latex(c, A, b, sinais, tipo):
    """Gera a representação LaTeX do problema."""
    sinal_z = "Min" if tipo == 'min' else "Max"
    termos = []
    for i, val in enumerate(c):
        if abs(val) > 1e-9:
            op = "+" if val >= 0 else "-"
            v = abs(val)
            v_str = f"{v:.2f}" if abs(v - 1.0) > 1e-9 else ""
            termos.append(f"{op} {v_str}x_{{{i+1}}}")
    z_eq = " ".join(termos).strip().lstrip("+")
    
    latex_str = f"\\text{{{sinal_z}}} \\ Z = {z_eq} \\\\"
    latex_str += "\\text{Sujeito a:} \\\\"
    latex_str += "\\begin{cases} "
    
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
        
        s_latex = "\\le" if sinais[i] == '<=' else "\\ge" if sinais[i] == '>=' else "="
        latex_str += f"{lhs} {s_latex} {val} \\\\"
        
    latex_str += "x_1, x_2 \\ge 0"
    latex_str += "\\end{cases}"
    return latex_str

def gerar_grafico(res, A, b, tipo, titulo):
    """Gera o objeto Figure do Matplotlib com o gráfico profissional."""
    vertices = res["vertices"]
    otimos = res["otimos"]
    eh_ilimitada = res["eh_ilimitada"]
    
    # Cálculo de Escala (Zoom Inteligente)
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
    
    # Desenho da Região
    pontos_para_poligono = vertices.copy()
    if eh_ilimitada:
        pontos_fundo = [[limite_zoom, 0], [limite_zoom, limite_zoom], [0, limite_zoom]]
        todos_pontos = np.vstack([vertices, pontos_fundo])
        centro = np.mean(todos_pontos, axis=0)
        angulos = np.arctan2(todos_pontos[:,1] - centro[1], todos_pontos[:,0] - centro[0])
        pontos_para_poligono = todos_pontos[np.argsort(angulos)]
        label_reg = 'Região Factível (Ilimitada)'
        cor_fundo = 'mediumseagreen'
        texto_extra = "ILIMITADA ↗"
    else:
        if len(vertices) > 2:
            centro = np.mean(vertices, axis=0)
            angulos = np.arctan2(vertices[:,1] - centro[1], vertices[:,0] - centro[0])
            pontos_para_poligono = vertices[np.argsort(angulos)]
        label_reg = 'Região Factível'
        cor_fundo = 'mediumseagreen'
        texto_extra = None

    poly = Polygon(pontos_para_poligono, closed=True, alpha=0.3, color=cor_fundo, label=label_reg)
    ax.add_patch(poly)
    
    if texto_extra:
        ax.text(limite_zoom*0.95, limite_zoom*0.95, texto_extra, 
                ha='right', va='top', color='darkgreen', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Plotagem de Pontos
    ax.scatter(vertices[:,0], vertices[:,1], color='black', s=30, zorder=5)
    ax.scatter(otimos[:,0], otimos[:,1], color='red', s=120, marker='*', zorder=6, label='Ótimo Global')
    
    # Restrições
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
    """Gera o relatório textual completo em Markdown."""
    md = "# Relatório de Solução - Método Gráfico\n\n"
    md += "## 1. Definição do Problema\n"
    md += "```text\n" + texto_original + "\n```\n\n"
    
    md += "## 2. Análise dos Vértices\n"
    md += "| Vértice ($x_1, x_2$) | Valor de Z | Status |\n"
    md += "| :--- | :--- | :--- |\n"
    
    z_opt = res['z_opt']
    for v, z in zip(res['vertices'], res['z_vals']):
        tag = "⭐ ÓTIMO" if np.abs(z - z_opt) < 1e-7 else ""
        md += f"| ({v[0]:.2f}, {v[1]:.2f}) | {z:.4f} | {tag} |\n"
        
    md += "\n## 3. Conclusão\n"
    if res['multiplos_otimos']:
        md += "**Obs:** Identificada multiplicidade de soluções ótimas (segmento de reta).\n\n"
        
    v_opt = res['otimos'][0]
    md += f"A solução ótima ocorre em **x1 = {v_opt[0]:.4f}**, **x2 = {v_opt[1]:.4f}**\n"
    md += f"Com valor objetivo **Z = {z_opt:.4f}**."
    
    return md

# =============================================================================
# 3. INTERFACE STREAMLIT
# =============================================================================

st.title("📊 Solver Gráfico de Programação Linear")
st.markdown("""
Esta ferramenta resolve problemas de PL com **2 variáveis** utilizando o Método Gráfico.
O algoritmo inclui detecção automática de **regiões ilimitadas** e **múltiplos ótimos**.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Entrada de Dados")
    st.info("Insira o problema no formato textual padrão.")
    
    exemplo_texto = """Minimizar : Z = 10x1 + 16x2
Sujeito a : 
x1 + 2x2 >= 40
2x1 + 5x2 >= 50
Tal que : 
x1, x2 >= 0"""
    
    texto_input = st.text_area("Formulação do Problema:", value=exemplo_texto, height=250)
    
    btn_resolver = st.button("🚀 Resolver Problema", type="primary")
    
    st.markdown("---")
    st.markdown("**Formato Aceito:**")
    st.code("""[Tipo] : Z = ...
Sujeito a :
[Restrições]
Tal que :
x1, x2 >= 0""", language="text")

# --- Lógica Principal ---
if btn_resolver:
    try:
        # 1. Parsing
        c, A, b, sinais, tipo = ler_problema_texto(texto_input)
        
        if len(c) != 2:
            st.error("❌ O método gráfico suporta estritamente 2 variáveis de decisão ($x_1$ e $x_2$).")
        else:
            # 2. Resolução
            res = resolver_sistema_grafico(c, A, b, sinais, tipo)
            
            if res["status"] == "inviavel":
                st.warning("⚠️ O problema não possui solução factível (Região Vazia).")
            else:
                # Layout de Resultados
                col1, col2 = st.columns([1, 1.5])
                
                with col1:
                    st.subheader("1. Formulação Matemática")
                    latex_code = gerar_texto_latex(c, A, b, sinais, tipo)
                    st.latex(latex_code)
                    
                    st.subheader("2. Resultados Numéricos")
                    
                    # Tabela de Vértices formatada
                    data_vertices = []
                    for v, z in zip(res['vertices'], res['z_vals']):
                        is_opt = np.abs(z - res['z_opt']) < 1e-7
                        marker = "⭐" if is_opt else ""
                        data_vertices.append({
                            "x1": f"{v[0]:.2f}",
                            "x2": f"{v[1]:.2f}",
                            "Z": f"{z:.2f}",
                            "Status": marker
                        })
                    st.dataframe(data_vertices, use_container_width=True)
                    
                    st.success(f"**Valor Ótimo (Z*): {res['z_opt']:.4f}**")
                    if res['multiplos_otimos']:
                        st.info("ℹ️ Múltiplas soluções ótimas encontradas.")

                with col2:
                    st.subheader("3. Visualização Gráfica")
                    fig = gerar_grafico(res, A, b, tipo, "Solução Gráfica")
                    st.pyplot(fig)
                
                # --- Área de Download ---
                st.markdown("---")
                st.subheader("📥 Exportar Resultados")
                
                # Gerar arquivos em memória
                relatorio_md = gerar_relatorio_markdown(texto_input, res, tipo)
                
                # Salvar imagem em buffer
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                
                # Criar ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    zf.writestr("relatorio_solucao.md", relatorio_md)
                    zf.writestr("grafico_solucao.png", img_buffer.getvalue())
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Baixar Pacote Completo (.zip)",
                    data=zip_buffer,
                    file_name="solucao_pl_grafica.zip",
                    mime="application/zip"
                )

    except Exception as e:
        st.error(f"❌ Erro ao processar o problema: {str(e)}")
        st.info("Verifique se o formato de entrada está correto e tente novamente.")

else:
    st.info("👈 Edite o problema na barra lateral e clique em 'Resolver' para começar.")