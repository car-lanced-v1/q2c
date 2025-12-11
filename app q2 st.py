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
# CONFIGURAÇÃO DO AMBIENTE E ESTILOS
# =============================================================================
st.set_page_config(
    page_title="Solver de Programação Linear - Método Gráfico",
    page_icon="📐",
    layout="wide"
)

# Configuração de parâmetros de plotagem para saída de alta resolução
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

# =============================================================================
# 1. MÓDULO DE INTERPRETAÇÃO DE DADOS (PARSER)
# =============================================================================

def ler_problema_texto(texto_entrada):
    """
    Realiza a análise sintática do texto de entrada para extrair os coeficientes
    da função objetivo e das restrições.
    
    Retorna:
        c (np.array): Vetor de coeficientes da função objetivo.
        A (np.array): Matriz de coeficientes das restrições.
        b (np.array): Vetor de termos independentes (Lado Direito).
        sinais (list): Lista de operadores relacionais (<=, >=, =).
        tipo_opt (str): Tipo de otimização ('max' ou 'min').
    """
    # Normalização de quebras de linha para garantir a leitura correta das seções
    texto_entrada = re.sub(r'(?i)(Sujeito a\s*[:]?)\s*', r'\1\n', texto_entrada)
    texto_entrada = re.sub(r'(?i)(Tal que\s*[:]?)\s*', r'\n\1\n', texto_entrada)
    
    lines = [l.strip() for l in texto_entrada.split('\n') if l.strip()]
    tipo_opt = None
    c = []
    A = []
    b = []
    sinais = []
    
    lendo_restricoes = False
    # Expressão regular para captura de termos algébricos (coeficiente e índice da variável)
    term_pattern = re.compile(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)')
    variaveis_indices = set()

    for line in lines:
        line_lower = line.lower()
        
        # Identificação da Função Objetivo e Tipo de Otimização
        if 'maximizar' in line_lower or 'minimizar' in line_lower:
            tipo_opt = 'min' if 'minimizar' in line_lower else 'max'
            if '=' in line:
                eq_part = line.split('=')[1].strip()
                matches = term_pattern.findall(eq_part)
                
                if not matches:
                    raise ValueError("Formato da função objetivo inválido. Verifique a sintaxe (ex: Z = 2x1 + 3x2).")
                
                temp_c = {}
                for coeff_str, idx_str in matches:
                    idx = int(idx_str) - 1
                    variaveis_indices.add(idx)
                    # Tratamento de coeficientes implícitos (1 ou -1)
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

        # Identificação de Seções do Modelo
        if 'sujeito a' in line_lower:
            lendo_restricoes = True
            continue
        if 'tal que' in line_lower:
            lendo_restricoes = False
            continue

        # Processamento das Restrições Técnicas
        if lendo_restricoes:
            sinal = None
            sep = None
            
            # Detecção do operador relacional
            if '<=' in line: sep, sinal = '<=', '<='
            elif '>=' in line: sep, sinal = '>=', '>='
            elif '≤' in line: sep, sinal = '≤', '<='
            elif '≥' in line: sep, sinal = '≥', '>='
            elif '=' in line: sep, sinal = '=', '='
            
            if sep:
                parts = line.split(sep)
                lhs, rhs = parts[0], parts[1]
                
                try:
                    # Extração do valor numérico do RHS, ignorando anotações
                    rhs_clean = re.split(r'\s', rhs.strip())[0]
                    b_val = float(rhs_clean)
                except ValueError:
                    continue 
                
                b.append(b_val)
                sinais.append(sinal)
                
                matches = term_pattern.findall(lhs)
                row = np.zeros(len(c)) if len(c) > 0 else np.zeros(2)
                
                for coeff_str, idx_str in matches:
                    idx = int(idx_str) - 1
                    # Ajuste dinâmico da dimensão do vetor se necessário
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

    if tipo_opt is None:
        raise ValueError("Declaração do objetivo (Maximizar/Minimizar) não encontrada.")
    if len(A) == 0:
        raise ValueError("Nenhuma restrição técnica identificada.")

    # Padronização dimensional das matrizes
    max_len = max(len(c), max([len(row) for row in A]) if A else 0)
    if len(c) < max_len:
        c_new = np.zeros(max_len)
        c_new[:len(c)] = c
        c = c_new
    
    A_final = [np.pad(row, (0, max_len - len(row))) if len(row) < max_len else row for row in A]
            
    return np.array(c), np.array(A_final), np.array(b), sinais, tipo_opt

# =============================================================================
# 2. MÓDULO DE CÁLCULO E GEOMETRIA
# =============================================================================

def verificar_limitacao_region(A, sinais):
    """
    Determina se a região factível é limitada (polígono fechado) através da
    análise vetorial do cone de direções normais.
    
    Se os vetores normais às restrições cobrirem todas as direções (gap angular < 180°),
    a região é limitada. Caso contrário, existe direção de recessão.
    """
    normais = []
    for i, row in enumerate(A):
        if sinais[i] == '<=': normais.append(row)
        elif sinais[i] == '>=': normais.append(-row)
        elif sinais[i] == '=':
            normais.append(row); normais.append(-row)
            
    # Inclusão das restrições de não-negatividade
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
        gap = (angulos[(i + 1) % len(angulos)] - angulos[i]) % 360
        if gap > max_gap: max_gap = gap
            
    return max_gap < (180.0 - 1e-4)

def resolver_sistema_grafico(c, A, b, sinais, tipo):
    """
    Executa o algoritmo do método gráfico:
    1. Determinação de vértices via interseção de restrições.
    2. Verificação de factibilidade de cada candidato.
    3. Avaliação da função objetivo nos vértices extremos.
    """
    # Inclusão dos eixos coordenados como restrições
    A_ext = np.vstack([A, np.eye(2)]) 
    b_ext = np.concatenate([b, [0, 0]])
    pontos = []
    indices = range(len(b_ext))
    
    # Cálculo das interseções (combinação linear de pares de restrições)
    for i, j in itertools.combinations(indices, 2):
        m = np.array([A_ext[i], A_ext[j]])
        rhs = np.array([b_ext[i], b_ext[j]])
        if np.abs(np.linalg.det(m)) > 1e-10:
            try: pontos.append(np.linalg.solve(m, rhs))
            except: pass
            
    # Filtragem de pontos factíveis
    vertices_factiveis = []
    for p in pontos:
        # Tolerância numérica para não-negatividade
        if p[0] < -1e-7 or p[1] < -1e-7: continue
        viavel = True
        for k, row in enumerate(A):
            val = np.dot(row, p)
            if sinais[k] == '<=' and val > b[k] + 1e-7: viavel = False
            elif sinais[k] == '>=' and val < b[k] - 1e-7: viavel = False
            elif sinais[k] == '=' and abs(val - b[k]) > 1e-7: viavel = False
            if not viavel: break
        
        if viavel:
            # Remoção de duplicatas
            if not any(np.linalg.norm(p - v) < 1e-7 for v in vertices_factiveis):
                vertices_factiveis.append(p)
    
    if not vertices_factiveis:
        return {"status": "inviavel"}

    # Organização e Otimização
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
# 3. MÓDULO DE VISUALIZAÇÃO E RELATÓRIOS
# =============================================================================

def gerar_texto_latex(c, A, b, sinais, tipo):
    """
    Gera a representação do modelo matemático em formato LaTeX.
    """
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
    """
    Gera a visualização gráfica da região factível e solução ótima.
    Ajusta automaticamente a escala dos eixos para garantir a visualização correta.
    """
    vertices = res["vertices"]
    otimos = res["otimos"]
    eh_ilimitada = res["eh_ilimitada"]
    
    # Definição dos limites dos eixos com base nos vértices encontrados
    max_x_v = np.max(vertices[:, 0]) if len(vertices) > 0 else 0
    max_y_v = np.max(vertices[:, 1]) if len(vertices) > 0 else 0
    
    # Identificação dos interceptos das restrições com os eixos
    int_x = [b[i]/row[0] for i, row in enumerate(A) if abs(row[0]) > 1e-6]
    int_y = [b[i]/row[1] for i, row in enumerate(A) if abs(row[1]) > 1e-6]
    
    # Filtragem de interceptos muito distantes para evitar distorção da escala
    limite_filtro_x = max_x_v * 4 + 20
    limite_filtro_y = max_y_v * 4 + 20
    
    int_x = [x for x in int_x if 0 < x < limite_filtro_x]
    int_y = [y for y in int_y if 0 < y < limite_filtro_y]
    
    max_int_x = max(int_x) if int_x else 0
    max_int_y = max(int_y) if int_y else 0
    
    # Cálculo final dos limites dos eixos com margem de segurança
    xlim_final = max(max_x_v, min(max_int_x, max_x_v * 3)) 
    ylim_final = max(max_y_v, min(max_int_y, max_y_v * 3))
    
    if xlim_final < 10: xlim_final = 10
    if ylim_final < 10: ylim_final = 10
    
    xlim_final *= 1.2
    ylim_final *= 1.2
    
    # Inicialização da Figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pontos_para_poligono = vertices.copy()
    
    if eh_ilimitada:
        # Tratamento visual para regiões ilimitadas: estensão do polígono aos limites
        pontos_fundo = [
            [xlim_final, 0], 
            [xlim_final, ylim_final], 
            [0, ylim_final]
        ]
        todos_pontos = np.vstack([vertices, pontos_fundo])
        # Ordenação polar para renderização correta do polígono
        centro = np.mean(todos_pontos, axis=0)
        angulos = np.arctan2(todos_pontos[:,1] - centro[1], todos_pontos[:,0] - centro[0])
        pontos_para_poligono = todos_pontos[np.argsort(angulos)]
        
        cor_fundo = 'mediumseagreen'
        lbl = 'Região Factível (Ilimitada)'
        extra_txt = "ILIMITADA"
    else:
        # Tratamento para regiões limitadas (convexas e fechadas)
        if len(vertices) > 2:
            centro = np.mean(vertices, axis=0)
            angulos = np.arctan2(vertices[:,1] - centro[1], vertices[:,0] - centro[0])
            pontos_para_poligono = vertices[np.argsort(angulos)]
            
        cor_fundo = 'mediumseagreen'
        lbl = 'Região Factível'
        extra_txt = None

    # Desenho da Região
    poly = Polygon(pontos_para_poligono, closed=True, alpha=0.3, color=cor_fundo, label=lbl)
    ax.add_patch(poly)
    
    if extra_txt:
        ax.text(xlim_final*0.95, ylim_final*0.95, extra_txt, 
                ha='right', va='top', color='darkgreen', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Plotagem dos Vértices e Solução Ótima
    ax.scatter(vertices[:,0], vertices[:,1], color='black', s=30, zorder=5)
    ax.scatter(otimos[:,0], otimos[:,1], color='red', s=120, marker='*', zorder=6, label='Ótimo Global')
    
    # Plotagem das Linhas de Restrição
    colors = plt.cm.tab10(np.linspace(0, 1, len(A)))
    vals_x = np.linspace(0, xlim_final, 200)
    
    for i, (row, val) in enumerate(zip(A, b)):
        color = colors[i]
        if abs(row[1]) > 1e-6:
            y_vals = (val - row[0]*vals_x) / row[1]
            # Filtragem para plotar apenas dentro da área visível
            mask = (y_vals >= -ylim_final*0.1) & (y_vals <= ylim_final*1.2)
            if np.any(mask):
                ax.plot(vals_x[mask], y_vals[mask], label=f'R{i+1}', color=color)
        else:
            x_v = val / row[0]
            if 0 <= x_v <= xlim_final:
                ax.vlines(x_v, 0, ylim_final, label=f'R{i+1}', color=color)

    ax.set_xlim(0, xlim_final)
    ax.set_ylim(0, ylim_final)
    ax.set_xlabel('Variável $x_1$')
    ax.set_ylabel('Variável $x_2$')
    ax.set_title(f"{titulo}\nOtimização: {tipo.upper()}", pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), borderaxespad=0)
    
    return fig

def gerar_relatorio_markdown(texto_original, res, tipo):
    """
    Gera um relatório formatado em Markdown com a descrição do problema e solução.
    """
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
# 4. INTERFACE DE USUÁRIO (STREAMLIT)
# =============================================================================

st.title("📊 Solver de Programação Linear (Método Gráfico)")
st.markdown("Ferramenta para resolução e análise gráfica de problemas de PL com duas variáveis.")

# Barra lateral para entrada de dados
with st.sidebar:
    st.header("Entrada de Dados")
    
    exemplo_padrao = """Minimizar : Z = 14x1 + 20x2
Sujeito a : x1 + 2x2 >= 4
7x1 + 6x2 >= 20
Tal que : x1, x2 >= 0"""
    
    texto_input = st.text_area("Formulação do Modelo:", value=exemplo_padrao, height=300)
    btn_resolver = st.button("🚀 Resolver", type="primary")
    
    st.markdown("""
    **Instruções de Formatação:**
    - Defina o objetivo: `Maximizar :` ou `Minimizar :`.
    - Liste as restrições após `Sujeito a :`.
    - Finalize com as restrições de sinal em `Tal que :`.
    """)

# Lógica de Execução
if btn_resolver:
    if not texto_input.strip():
        st.warning("⚠️ Insira o modelo matemático na caixa de texto.")
    else:
        try:
            # Processamento
            c, A, b, sinais, tipo = ler_problema_texto(texto_input)
            
            if len(c) != 2:
                st.error("❌ Erro Dimensional: O método gráfico requer exatamente 2 variáveis de decisão ($x_1, x_2$).")
            else:
                res = resolver_sistema_grafico(c, A, b, sinais, tipo)
                
                if res["status"] == "inviavel":
                    st.warning("⚠️ Problema Inviável: Não existe região factível que atenda a todas as restrições.")
                else:
                    # Exibição de Resultados
                    col1, col2 = st.columns([1, 1.5])
                    
                    with col1:
                        st.subheader("1. Formulação Matemática")
                        st.latex(gerar_texto_latex(c, A, b, sinais, tipo))
                        
                        st.subheader("2. Resultados Numéricos")
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
                        fig = gerar_grafico(res, A, b, tipo, "Solução Gráfica")
                        st.pyplot(fig)
                    
                    # Seção de Download
                    st.markdown("---")
                    st.subheader("📥 Exportação de Resultados")
                    
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
                        label="📦 Baixar Relatório Completo (.zip)",
                        data=zip_buf,
                        file_name="solucao_pl.zip",
                        mime="application/zip"
                    )

        except ValueError as ve:
            st.error(f"❌ Erro de Sintaxe: {str(ve)}")
            st.markdown("Verifique se o modelo segue o padrão: `Objetivo`, `Sujeito a`, `Tal que`.")
        except Exception as e:
            st.error(f"❌ Erro Inesperado: {str(e)}")
else:
    st.info("👈 Utilize a barra lateral para inserir o modelo e clique em **Resolver**.")