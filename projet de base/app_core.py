
# app_core.py — logique, chargement des données, fonctions utilitaires, graphes, layout (AUCUN callback)

import random
import pandas as pd
import networkx as nx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
from dash import dcc, html
import colorsys
from copy import copy
import numpy as np

# Instance Dash
app = dash.Dash(__name__)

# Lecture des données Excel
file_path = "all_Competence.xlsx"  # Chemin vers le fichier Excel
df = pd.read_excel(file_path)

# Couleurs par UE
UE_colors = {
    "UE1": "#E25012",
    "UE2": "#E28A12",
    "UE3": "#155992",
    "UE4": "#0C9A61",
}


def extract_ue(ue_string):
    """Extraire la valeur UE à partir d'une chaîne telle que "UE1 - xxx"."""
    if pd.isna(ue_string):
        return None
    ue_parts = [part.strip() for part in ue_string.split('-')]
    for part in ue_parts:
        if part.startswith("UE"):
            return part
    return None


# Mettre à jour la colonne UE dès le départ
df['UE'] = df['UE'].apply(extract_ue)


def get_ue_color(ue):
    """Obtenir la couleur associée à une UE."""
    return UE_colors.get(ue, "#7f7f7f")


def adjust_color_brightness(hex_color, factor):
    """Ajuster la luminosité d'une couleur hex suivant un facteur."""
    rgb = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    h, l, s = colorsys.rgb_to_hls(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def get_level_color(level, ue_or_base_color):
    """Ajuster la couleur de base selon le niveau (N/A/M/E)."""
    if level == 'N':
        factor = 1.5   # plus clair
    elif level == 'A':
        factor = 1.2   # légèrement plus clair
    elif level == 'M':
        factor = 0.8   # légèrement plus sombre
    elif level == 'E':
        factor = 0.6   # sombre
    else:
        factor = 1.0
    return adjust_color_brightness(ue_or_base_color, factor)

# Couleurs de base par catégorie (TC / IDU)
CATEGORY_COLORS = {
    "IDU-1": "#FF513F",
    "IDU-2": "#FFA03F",
    "IDU-3": "#DC3785",
    "IDU-4": "#FFC83F",
    "TC-1": "#2DB593",
    "TC-2": "#3A6CB7",
    "TC-3": "#65E038",
    "TC-4": "#5740BD",
}


def get_competency_color(competency, level=None):
    """Couleur d'une compétence (ex: "IDU-1.2") avec option d'ajustement par niveau."""
    parts = competency.split("-")
    if len(parts) < 2:
        return "#777777"
    category_key = f"{parts[0]}-{parts[1][0]}"  # ex: IDU-1, TC-3
    base_color = CATEGORY_COLORS.get(category_key, "#AAAAAA")
    return get_level_color(level, base_color) if level else base_color


def get_category_base_color(category_key):
    """Renvoyer la couleur de base d'une catégorie (TC-1, IDU-2)."""
    return CATEGORY_COLORS.get(category_key, "#777777")

# ---------------------- Parsing / Préparation ----------------------

def parse_competencies(row):
    """Rassembler jusqu'à 12 colonnes Competencies.i en une liste."""
    competencies = []
    for i in range(1, 13):
        comp_col = f"Competencies.{i}"
        if comp_col in row and pd.notna(row[comp_col]):
            competencies.append(row[comp_col])
    return competencies


def get_competency_type(comp):
    if str(comp).startswith('TC'):
        return 'TC'
    if str(comp).startswith('IDU'):
        return 'IDU'
    return None


def get_sorted_competencies(selected_category):
    """Options triées (TC / IDU / ALL) pour les listes."""
    competencies = {'TC': [], 'IDU': [], 'ALL': []}
    for _, row in df.iterrows():
        competencies_list = row['All_Competencies']
        for comp in competencies_list:
            comp_type = get_competency_type(comp)
            if comp_type:
                competencies[comp_type].append(comp)
            competencies['ALL'].append(comp)

    for category in competencies:
        competencies[category] = sorted(
            set(competencies[category]),
            key=lambda x: (
                x.split('-')[0],
                float(x.split('-')[1]) if len(x.split('-')) > 1 and x.split('-')[1].replace('.', '', 1).isdigit() else 0,
            ),
        )
    return [{'label': comp, 'value': comp} for comp in competencies.get(selected_category, [])]


# Colonnes dérivées
df["All_Competencies"] = df.apply(parse_competencies, axis=1)
df['Competency_Type'] = df['All_Competencies'].apply(lambda lst: [get_competency_type(comp) for comp in lst])

# ---------------------- Comptages & visuels compacts ----------------------

def compute_competency_counts():
    """Comptage d'apparitions par semestre (sparklines)."""
    semesters = sorted(df['Semestre'].dropna().unique())
    competency_counts = {}
    for comp in df["All_Competencies"].explode().dropna().unique():
        counts = []
        for sem in semesters:
            count = df[df['Semestre'] == sem]["All_Competencies"].apply(lambda x: comp in x).sum()
            counts.append(count)
        competency_counts[comp] = counts
    return competency_counts, semesters


competency_counts_spark, semesters = compute_competency_counts()


def create_sparkline(data, semesters):
    """Créer une sparkline Plotly compacte pour une compétence."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=semesters, y=data, mode='lines+markers', line=dict(color='black')))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 13]),
        height=30,
        width=100,
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


# Comptages par niveau & semestre pour barres empilées

def compute_competency_counts_with_levels():
    """Comptages par (compétence, niveau, semestre)."""
    semesters = sorted(df['Semestre'].dropna().unique())
    competency_counts = {}
    levels = ['N', 'A', 'M', 'E']

    for comp in df["All_Competencies"].explode().dropna().unique():
        counts = {level: [0] * len(semesters) for level in levels}
        for sem_index, sem in enumerate(semesters):
            semester_data = df[df['Semestre'] == sem]
            for level in levels:
                level_count = semester_data[semester_data["Level"] == level].apply(
                    lambda x: comp in x["All_Competencies"] if isinstance(x["All_Competencies"], list) else False,
                    axis=1,
                ).sum()
                counts[level][sem_index] = level_count
        competency_counts[comp] = counts
    return competency_counts, semesters


competency_counts_bar, semesters = compute_competency_counts_with_levels()

# ---------------------- Préparation heatmaps ----------------------

def compute_competency_counts_per_module():
    """Préparer les pivots pour heatmaps global / semestre / module."""
    df_count = (
        df.explode("All_Competencies").reset_index(drop=True)
          .groupby(["Semestre", "EC", "All_Competencies"], as_index=False)
          .size()
    )
    df_pivot_global = (
        df.explode("All_Competencies").reset_index(drop=True)
          .groupby(["Semestre", "All_Competencies"], as_index=False)
          .size()
          .pivot(index="Semestre", columns="All_Competencies", values="size")
          .fillna(0)
          .astype("Float64")
    )
    df_pivot_semester = (
        df_count.pivot(index=["Semestre", "EC"], columns="All_Competencies", values="size")
           .fillna(0)
           .astype("Float64")
    )
    df_pivot_module = (
        df.explode("All_Competencies").reset_index(drop=True)
          .pivot(index=["Semestre", "EC", "Goal"], columns="All_Competencies", values="Level")
          .fillna(0)
    )
    return df_pivot_global, df_pivot_semester, df_pivot_module


df_pivot_global, df_pivot_semester, df_pivot_module = compute_competency_counts_per_module()

# ---------------------- Fabrication des heatmaps ----------------------

def create_heatmap_for_global(df_pivot_global):
    fig = go.Figure()
    for j in range(df_pivot_global.shape[1]):
        competency = df_pivot_global.columns[j]
        color_competency = get_competency_color(competency, level=None)
        color_scale = ["white", color_competency]
        df_comp = copy(df_pivot_global)
        df_comp.loc[:, df_comp.columns != competency] = pd.NA
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_pivot_global.columns,
                y=df_pivot_global.index,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
            )
        )
    fig.update_layout(
        title=f"All Competencies",
        xaxis_title="Competencies",
        xaxis=dict(tickmode="array", tickvals=df_pivot_global.columns),
        yaxis_title="Semesters",
        yaxis=dict(
            type="category",
            autorange="reversed",
            categoryorder="array",
            categoryarray=list(df_pivot_global.index),
            showgrid=False,
        ),
    )
    return fig


def create_heatmap_for_semester(semestre, df_pivot_semester, ylegend=True):
    df_used = df_pivot_semester.xs(semestre, level="Semestre")
    fig = go.Figure()
    z_min = 0
    z_max = df_pivot_semester.max().max()
    for j in range(df_used.shape[1]):
        competency = df_used.columns[j]
        color_competency = get_competency_color(competency, level=None)
        color_scale = ["white", color_competency]
        df_comp = copy(df_used)
        df_comp.loc[:, df_comp.columns != competency] = pd.NA
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=df_used.index,  # index = EC
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1,
                ygap=1
            )
        )
    fig.update_layout(
        title=f"All Competencies for Semester {semestre}",
        xaxis_title="Competencies",
        yaxis_title="Modules",
    )
    fig.update_layout(plot_bgcolor="black")   
    if not ylegend:
        fig.update_yaxes(showticklabels=False)
    return fig


def create_heatmap_for_module(module, df_pivot_module):
    # Remplacer niveaux par numériques pour l'intensité, conserver la légende par couleurs de compétence
    df_reorganized = df_pivot_module.replace({'N': 1, 'A': 2, 'M': 3, 'E': 4})
    df_used = df_reorganized.xs(module, level="EC").rename(columns={np.nan: "No Competency"}).astype("Int64")

    fig = go.Figure()
    z_min = 0
    z_max = df_used.max().max()
    for j in range(df_used.shape[1]):
        competency = df_used.columns[j]
        color_competency = get_competency_color(competency, level=None)
        color_scale = ["white", color_competency]
        df_comp = copy(df_used)
        df_comp.loc[:, df_comp.columns != competency] = pd.NA
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=df_used.index.get_level_values("Goal"),
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1,
                ygap=1
            )
        )
    fig.update_layout(
        title=f"All Competencies for {module}",
        xaxis_title="Competencies",
        yaxis_title="Learning Outcomes",
    )
    fig.update_layout(plot_bgcolor="black")
    
    return fig

# ---------------------- Barres empilées ----------------------

def create_stacked_bar_chart_for_competency(competency_counts, comp_name, color_mode):
    fig = go.Figure()
    levels = ['N', 'A', 'M', 'E']
    semesters = [5, 6, 7, 8, 9, 10]
    grayscale_colors = {sem: '#777777' for sem in semesters}

    if comp_name in competency_counts:
        counts = competency_counts[comp_name]
        stacked_data = []
        for sem_index, sem in enumerate(semesters):
            sem_total = sum(counts[level][sem_index] for level in levels)
            if sem_total == 0:
                continue
            for level in levels:
                count = counts[level][sem_index]
                if count > 0:
                    base_color = grayscale_colors[sem] if color_mode == 'grayscale' else get_competency_color(comp_name, level)
                    level_color = get_level_color(level, base_color)
                    hover_text = f"Semester: {sem}<br>Level: {level}<br>Count: {count}"
                    stacked_data.append((count, level, sem, level_color, hover_text))
        stacked_data.sort(key=lambda x: x[2])
        for count, level, sem, color, hover_text in stacked_data:
            fig.add_trace(go.Bar(
                y=[comp_name], x=[count], marker=dict(color=color), orientation='h',
                text=hover_text, hoverinfo="text", textposition="none",
            ))

    fig.update_layout(
        barmode='stack', height=200, width=500,
        margin=dict(l=20, r=20, t=20, b=60),
        xaxis=dict(title='Count', showgrid=True, zeroline=True, range=[0, 45]),
        yaxis=dict(title='', tickmode='array', tickvals=[comp_name], ticktext=[comp_name], showgrid=False, zeroline=False),
        showlegend=False,
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_vertical_stacked_bar_chart_for_competency(competency_counts, comp_name, color_mode):
    fig = go.Figure()
    levels = ['N', 'A', 'M', 'E']
    semesters = [5, 6, 7, 8, 9, 10]
    grayscale_colors = {sem: '#777777' for sem in semesters}

    if comp_name in competency_counts:
        counts = competency_counts[comp_name]
        stacked_data = []
        for sem_index, sem in enumerate(semesters):
            sem_total = sum(counts[level][sem_index] for level in levels)
            if sem_total == 0:
                continue
            for level in levels:
                count = counts[level][sem_index]
                if count > 0:
                    base_color = grayscale_colors[sem] if color_mode == 'grayscale' else get_competency_color(comp_name, level)
                    level_color = get_level_color(level, base_color)
                    hover_text = f"Semester: {sem}<br>Level: {level}<br>Count: {count}"
                    stacked_data.append((count, level, sem, level_color, hover_text))
        stacked_data.sort(key=lambda x: x[2])
        for count, level, sem, color, hover_text in stacked_data:
            fig.add_trace(go.Bar(
                x=[sem], y=[count], marker=dict(color=color), text=hover_text,
                hoverinfo="text", textposition="none", width=0.4,
            ))

    fig.update_layout(
        barmode='stack', height=400, width=500,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(title='', tickvals=semesters, ticktext=[str(s) for s in semesters], range=[4.5, 10.5], showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='', showgrid=True, zeroline=True, range=[0, 15]),
        showlegend=False,
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# ---------------------- Construction réseau (hors callbacks) ----------------------

ec_dict = {}
for _, row in df.iterrows():
    ec = row['EC']
    semester = row['Semestre']
    goal = row['Goal']
    competencies = row["All_Competencies"]
    level = row['Level']

    # Stocker niveaux par EC
    if ec not in ec_dict:
        ec_dict[ec] = {'semester': semester, 'level': level, 'goals': {}}
    if goal not in ec_dict[ec]['goals']:
        ec_dict[ec]['goals'][goal] = {'level': level, 'competencies': []}
    ec_dict[ec]['goals'][goal]['competencies'].extend(competencies)

# Graphe de réseau
G = nx.Graph()
for _, row in df.iterrows():
    ec = row['EC']
    semester = row['Semestre']
    goal = row['Goal']
    competencies = row["All_Competencies"]
    level = row['Level']
    class_node = f"{ec} (S{semester})"
    ue = row['UE']
    ue_color = get_ue_color(ue)
    G.add_node(class_node, node_type="class", semester=semester, label=ec, color=ue_color)

    if ec in ec_dict and goal in ec_dict[ec]['goals']:
        goal_level = ec_dict[ec]['goals'][goal]['level']
        goal_color = get_level_color(goal_level, ue_color)
        goal_label_with_level = f"{goal} ({goal_level})"
        goal_node = f"{ec}::{goal}"
        G.add_node(goal_node, node_type="goal", semester=semester, label=goal_label_with_level, color=goal_color)
        G.add_edge(class_node, goal_node)

        for competency in competencies:
            competency_node = f"{class_node}::{goal}::{competency}"
            competency_color = get_competency_color(competency, goal_level)
            if competency_node not in G:
                G.add_node(competency_node, node_type="competency", label=competency, color=competency_color)
            G.add_edge(goal_node, competency_node)

# Positionnement des nœuds (simple grille par semestre)
pos = {}
y_offsets = {semester: 0 for semester in df['Semestre'].unique()}
spacing = 3
for ec, data in ec_dict.items():
    semester = data['semester']
    class_node = f"{ec} (S{semester})"
    pos[class_node] = (semester, y_offsets[semester])
    y_offsets[semester] += spacing
    for goal, goal_data in data['goals'].items():
        goal_node = f"{ec}::{goal}"
        pos[goal_node] = (semester + 0.2, y_offsets[semester])
        y_offsets[semester] += spacing
        for competency in goal_data['competencies']:
            competency_node = f"{class_node}::{goal}::{competency}"
            pos[competency_node] = (semester + 0.4, y_offsets[semester])
            y_offsets[semester] += spacing

# ---------------------- Aide non-callback pour le réseau ----------------------

def update_network(selected_category, selected_competencies):
    """Construire la figure réseau filtrée (appelée depuis un callback)."""
    nodes_to_keep = set()
    if not selected_competencies:
        selected_competencies = [comp['value'] for comp in get_sorted_competencies(selected_category)]

    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'competency' and G.nodes[node]['label'] in selected_competencies:
            nodes_to_keep.add(node)
            for neighbor in G.neighbors(node):
                nodes_to_keep.add(neighbor)
                for grandparent in G.neighbors(neighbor):
                    nodes_to_keep.add(grandparent)

    edge_x, edge_y = [], []
    for edge in G.edges():
        if edge[0] in nodes_to_keep and edge[1] in nodes_to_keep:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in nodes_to_keep:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['label'])
        node_color.append(G.nodes[node]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text', marker=dict(size=10, color=node_color)
    )

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=True,
            tickmode='array',
            tickvals=list(range(5, 11)),
            range=[4.5, 10.5],
        ),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        height=1000,
        width=1500,
    )

    return {'data': [edge_trace, node_trace], 'layout': layout}

# ==============================================================================
# ----- NOUVELLES VISUALISATIONS (Question 6) -----
# ==============================================================================

# 1. Préparation des données pour le classement des compétences
competency_total_counts = df.explode("All_Competencies")["All_Competencies"].value_counts().reset_index()
competency_total_counts.columns = ['Competency', 'Count']
# On garde le Top 15 pour la clarté du graphique
top_15_competencies = competency_total_counts.head(15).sort_values(by='Count', ascending=True)

# 2. Fonction de création du graphique "Classement"
def create_competency_leaderboard_graph(data):
    """Crée un graphique à barres horizontales pour classer les compétences."""
    fig = go.Figure()

    # Créer une couleur pour chaque barre basée sur la couleur de la compétence
    colors = [get_competency_color(comp) for comp in data['Competency']]

    fig.add_trace(go.Bar(
        y=data['Competency'],
        x=data['Count'],
        orientation='h',
        marker=dict(color=colors)
    ))

    fig.update_layout(
        title_text='<b>Top 15 des Compétences du Cursus</b>',
        title_x=0.5,
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Nombre d'occurrences dans les modules",
        yaxis_title="",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=False)
    )
    return fig

# ==============================================================================
# ---------------------- Layout (aucun callback ici) ----------------------

# Nouvelle disposition en 3 colonnes incluant le classement
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
    
    # --- EN-TÊTE ---
    html.Div([
        html.H1("Tableau de Bord des Compétences du Cursus", style={'margin': '0'}),
        html.P("Explorez les liens entre cours, objectifs et compétences.", style={'margin': '0'})
    ], style={'padding': '20px', 'backgroundColor': '#f2f2f2', 'borderBottom': '1px solid #ddd'}),

    # --- CORPS PRINCIPAL DE L'APPLICATION (Conteneur Flex) ---
    html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[

        # --- BLOC 1 : COLONNE DE GAUCHE (CONTRÔLES & FILTRES) ---
        html.Div([
            html.H3("Pilotez votre Analyse", style={'borderBottom': '2px solid #ccc', 'paddingBottom': '10px'}),
            
            html.Label("1. Choisir la catégorie de compétences :"),
            dcc.Dropdown(
                id='competency-category',
                options=[{'label': 'Toutes les compétences', 'value': 'ALL'}, {'label': 'Compétences Transversales (TC)', 'value': 'TC'}, {'label': 'Compétences Spécifiques (IDU)', 'value': 'IDU'}],
                value='ALL',
                clearable=False
            ),
            
            html.Br(),
            
            html.Label("2. Sélectionner une ou plusieurs compétences :"),
            html.P("Les graphiques se mettront à jour automatiquement.", style={'fontSize': '12px', 'color': 'gray'}),
            dcc.Checklist(
                id='competency-list',
                options=[],
                value=[],
                inline=False,
                style={'width': '100%', 'height': '70vh', 'overflowY': 'auto', 'border': '1px solid #ccc', 'padding': '10px', 'borderRadius': '5px'},
            ),
            
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f9f9f9'}),

        # --- BLOC 2 : COLONNE CENTRALE (VISUALISATION PRINCIPALE) ---
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='heatmap-or-forest',
                    options=[{'label': 'Vue Synthétique (Heatmap)', 'value': 'Heatmaps'}, {'label': 'Vue Détaillée (Forêt)', 'value': 'Forest'}],
                    value='Heatmaps',
                    inline=True,
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                ),
                html.Button("Précédent (zoom arrière)", id="go_back", n_clicks=0, style={"margin-left": "40px"}),
            ], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            
            dcc.Graph(id='network-graph', style={'height': '80vh'}),

        ], style={'width': '50%', 'padding': '20px'}),

        # --- BLOC 3 : COLONNE DE DROITE (RÉSUMÉS ANALYTIQUES) ---
        html.Div([
            dcc.Graph(id='competency-leaderboard') # Le graphique est maintenant vide, prêt à être piloté
            ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f9f9f9'}),

    ]),

    # --- ÉLÉMENTS MASQUÉS ET STORES ---
    dcc.Store(id='checklist-store', data=[]),
    dcc.Store(id='legend-visible', data=True),
    dcc.Store(id='heatmaps-data', data={'level': 0, 'semester': None, 'module': None}),
])

# Export des symboles nécessaires aux callbacks
__all__ = [
    'app', 'df', 'competency_counts_spark', 'semesters', 'competency_counts_bar',
    'df_pivot_global', 'df_pivot_semester', 'df_pivot_module',
    'get_sorted_competencies', 'create_sparkline', 'create_heatmap_for_semester',
    'create_heatmap_for_global', 'create_heatmap_for_module',
    'create_vertical_stacked_bar_chart_for_competency', 'create_stacked_bar_chart_for_competency',
    'update_network',
    'top_15_competencies', 'create_competency_leaderboard_graph' # On exporte les nouveaux éléments
]