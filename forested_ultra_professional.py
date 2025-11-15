"""
╔══════════════════════════════════════════════════════════════════════════╗
║               ForestED ULTRA PROFESSIONAL - Vue Étudiant Expert          ║
║                                                                          ║
║  🎓 Projet de Visualisation - Réponses aux Questions 1-6                ║
║  🚀 8 Visualisations Innovantes Synchronisées + DRILL-DOWN 3 NIVEAUX   ║
║  🎨 Design shadcn/ui Professionnel                                     ║
║  📊 Perspective 100% Étudiant avec Narration des Données               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from datetime import datetime
import networkx as nx
from copy import copy
from io import StringIO  # AJOUTER cet import en haut du fichier

# ═══════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    assets_folder='assets',
    title="ForestED Ultra Professional - Vue Étudiant"
)

# Chargement des données
df = pd.read_excel("all_Competence.xlsx")

# ═══════════════════════════════════════════════════════════════════
#                    TRAITEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════

def parse_competencies(row):
    """Parser les compétences de chaque ligne"""
    competencies = []
    for i in range(1, 13):
        comp_col = f"Competencies.{i}"
        if comp_col in row.index and pd.notna(row[comp_col]):
            competencies.append(row[comp_col])
    return competencies

df["Competencies_List"] = df.apply(parse_competencies, axis=1)

# CORRECTION: Éviter le chained assignment (ligne 54)
df_expanded = df.copy()
df_expanded = df_expanded.assign(All_Competencies=df_expanded['Competencies_List'])

def extract_ue(ue_string):
    """Extraire l'UE proprement"""
    if pd.isna(ue_string):
        return None
    parts = [part.strip() for part in ue_string.split('-')]
    for part in parts:
        if part.startswith("UE"):
            return part
    return None

df['UE_Clean'] = df['UE'].apply(extract_ue)

# ═══════════════════════════════════════════════════════════════════
#            DESIGN SYSTEM SHADCN/UI - COULEURS PROFESSIONNELLES
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    # Palette principale (HSL pour accessibilité)
    'primary': '#3B82F6',      # Bleu moderne
    'secondary': '#8B5CF6',    # Violet
    'success': '#10B981',      # Vert
    'warning': '#F59E0B',      # Orange
    'danger': '#EF4444',       # Rouge
    'info': '#06B6D4',         # Cyan
    'light': '#F8FAFC',        # Gris clair
    'dark': '#1E293B',         # Gris foncé
    
    # Couleurs par UE (harmonisées)
    'ue': {
        'UE1': '#E25012',  # Orange vif
        'UE2': '#E28A12',  # Doré
        'UE3': '#155992',  # Bleu profond
        'UE4': '#0C9A61',  # Vert émeraude
    },
    
    # Niveaux de maîtrise (gradient cohérent)
    'levels': {
        'N': '#BFDBFE',  # Novice - bleu pastel
        'A': '#60A5FA',  # Apprenti - bleu clair
        'M': '#2563EB',  # Maître - bleu moyen
        'E': '#1E40AF',  # Expert - bleu foncé
    },
    
    # Catégories de compétences
    'categories': {
        "IDU-1": "#FF513F",  # Rouge corail
        "IDU-2": "#FFA03F",  # Orange
        "IDU-3": "#DC3785",  # Rose
        "IDU-4": "#FFC83F",  # Jaune
        "TC-1": "#2DB593",   # Turquoise
        "TC-2": "#3A6CB7",   # Bleu
        "TC-3": "#65E038",   # Vert lime
        "TC-4": "#5740BD",   # Violet foncé
    }
}

def get_competency_color(competency, level=None):
    """Obtenir la couleur d'une compétence"""
    parts = competency.split("-")
    if len(parts) < 2:
        return "#777777"
    category_key = f"{parts[0]}-{parts[1][0]}"
    return COLORS['categories'].get(category_key, "#AAAAAA")

# ═══════════════════════════════════════════════════════════════════
#           FONCTIONS DE PRÉPARATION DES DONNÉES POUR HEATMAPS
# ═══════════════════════════════════════════════════════════════════

def compute_competency_counts_per_module(df_input):
    """Préparer les pivots pour heatmaps global / semestre / module."""
    # CORRECTION: Éviter le chained assignment (ligne 66)
    df_work = df_input.copy()
    if 'Goal' not in df_work.columns:
        df_work = df_work.assign(Goal=df_work.index.astype(str))
    
    # Exploser les compétences
    df_exploded = df_work.explode("All_Competencies").reset_index(drop=True)
    
    # Compter les occurrences
    df_count = (
        df_exploded
        .groupby(["Semestre", "EC", "All_Competencies"], as_index=False)
        .size()
    )
    
    # Pivot global (Semestre x Compétences)
    df_pivot_global = (
        df_exploded
        .groupby(["Semestre", "All_Competencies"], as_index=False)
        .size()
        .pivot(index="Semestre", columns="All_Competencies", values="size")
        .fillna(0)
        .astype("Float64")
    )
    
    # Pivot par semestre (EC x Compétences pour chaque semestre)
    df_pivot_semester = (
        df_count.pivot(index=["Semestre", "EC"], columns="All_Competencies", values="size")
        .fillna(0)
        .astype("Float64")
    )
    
    # Pivot par module (Goal x Compétences avec niveaux)
    df_pivot_module = (
        df_exploded
        .pivot(index=["Semestre", "EC", "Goal"], columns="All_Competencies", values="Level")
        .fillna(0)
    )
    
    return df_pivot_global, df_pivot_semester, df_pivot_module

# ═══════════════════════════════════════════════════════════════════
#              FONCTIONS DE CRÉATION DES HEATMAPS (3 NIVEAUX)
# ═══════════════════════════════════════════════════════════════════

def create_heatmap_for_global(df_pivot_global):
    """
    NIVEAU 1: Vue globale par semestre
    Affiche toutes les compétences par semestre
    """
    fig = go.Figure()
    
    for j in range(df_pivot_global.shape[1]):
        competency = df_pivot_global.columns[j]
        color_competency = get_competency_color(competency, level=None)
        color_scale = ["white", color_competency]
        df_comp = df_pivot_global.copy()
        # CORRECTION: Utiliser loc pour éviter chained assignment
        for col in df_comp.columns:
            if col != competency:
                df_comp.loc[:, col] = pd.NA
        
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_pivot_global.columns,
                y=df_pivot_global.index,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                hovertemplate='<b>Semestre %{y}</b><br>%{x}<br>Occurrences: %{z}<br><i>Cliquez pour voir les modules</i><extra></extra>'
            )
        )
    
    fig.update_layout(
        title={
            'text': "🎯 MON PARCOURS - Vue Globale<br><sub>Cliquez sur une cellule pour voir les modules du semestre</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter, sans-serif'}
        },
        xaxis_title="Compétences",
        xaxis=dict(
            tickmode="array", 
            tickvals=list(range(len(df_pivot_global.columns))),
            ticktext=df_pivot_global.columns,
            tickangle=-45
        ),
        yaxis_title="Semestres",
        yaxis=dict(
            type="category",
            autorange="reversed",
            categoryorder="array",
            categoryarray=list(df_pivot_global.index),
            showgrid=False,
        ),
        height=500,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        margin=dict(l=100, r=50, t=100, b=150),
        clickmode='event+select'
    )
    
    return fig


def create_heatmap_for_semester(semestre, df_pivot_semester, ylegend=True):
    """
    NIVEAU 2: Vue détaillée d'un semestre
    Affiche tous les modules (EC) avec leurs compétences
    """
    df_used = df_pivot_semester.xs(semestre, level="Semestre")
    
    fig = go.Figure()
    z_min = 0
    z_max = df_pivot_semester.max().max()
    
    for j in range(df_used.shape[1]):
        competency = df_used.columns[j]
        color_competency = get_competency_color(competency, level=None)
        color_scale = ["white", color_competency]
        df_comp = df_used.copy()
        # CORRECTION: Utiliser loc pour éviter chained assignment
        for col in df_comp.columns:
            if col != competency:
                df_comp.loc[:, col] = pd.NA
        
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
                ygap=1,
                hovertemplate='<b>%{y}</b><br>%{x}<br>Occurrences: %{z}<br><i>Cliquez pour voir les objectifs</i><extra></extra>'
            )
        )
    
    fig.update_layout(
        title={
            'text': f"📚 Semestre {semestre} - Vue Détaillée<br><sub>Cliquez sur une cellule pour voir les objectifs pédagogiques du module</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter, sans-serif'}
        },
        xaxis_title="Compétences",
        yaxis_title="Modules (EC)",
        xaxis=dict(tickangle=-45),
        height=max(400, len(df_used.index) * 30 + 150),
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='black',
        paper_bgcolor=COLORS['light'],
        margin=dict(l=250, r=50, t=100, b=150),
        clickmode='event+select'
    )
    
    if not ylegend:
        fig.update_yaxes(showticklabels=False)
    
    return fig


def create_heatmap_for_module(module, df_pivot_module):
    """
    NIVEAU 3: Vue détaillée d'un module
    Affiche les objectifs pédagogiques (Goals) avec niveaux de compétences
    """
    # Remplacer niveaux par numériques pour l'intensité
    df_reorganized = df_pivot_module.replace({'N': 1, 'A': 2, 'M': 3, 'E': 4})
    df_used = df_reorganized.xs(module, level="EC").rename(columns={np.nan: "No Competency"}).astype("Int64")

    fig = go.Figure()
    z_min = 0
    z_max = 4  # Max niveau = Expert
    
    for j in range(df_used.shape[1]):
        competency = df_used.columns[j]
        color_competency = get_competency_color(competency, level=None)
        color_scale = ["white", color_competency]
        df_comp = df_used.copy()
        # CORRECTION: Utiliser loc pour éviter chained assignment
        for col in df_comp.columns:
            if col != competency:
                df_comp.loc[:, col] = pd.NA
        
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=df_used.index.get_level_values("Goal") if hasattr(df_used.index, 'get_level_values') else df_used.index,
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1,
                ygap=1,
                hovertemplate='<b>Objectif %{y}</b><br>%{x}<br>Niveau: %{z}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title={
            'text': f"🎓 {module} - Objectifs Pédagogiques<br><sub>Niveaux: 1=Novice, 2=Apprenti, 3=Maître, 4=Expert</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter, sans-serif'}
        },
        xaxis_title="Compétences",
        yaxis_title="Objectifs Pédagogiques",
        xaxis=dict(tickangle=-45),
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='black',
        paper_bgcolor=COLORS['light'],
        margin=dict(l=250, r=50, t=100, b=150),
        annotations=[
            dict(
                text="<b>Légende:</b> Intensité de la couleur = Niveau de maîtrise attendu",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════
#                    COMPOSANTS UI RÉUTILISABLES
# ═══════════════════════════════════════════════════════════════════

def create_stat_card(title, value, subtitle, icon, color, trend=None, comparison=None):
    """Créer une carte de statistique moderne (shadcn/ui style)"""
    card_content = [
        html.Div([
            html.Div(title, className="stat-label"),
            html.Div(icon, className="stat-icon", 
                    style={'background': f'{color}20', 'color': color})
        ], className="stat-header"),
        html.Div(str(value), className="stat-value", style={'color': color}),
    ]
    
    if trend is not None:
        trend_class = "stat-trend-up" if trend > 0 else "stat-trend-down"
        card_content.append(
            html.Div([
                html.Span("↑" if trend > 0 else "↓"),
                html.Span(f"{abs(trend)}%")
            ], className=f"stat-trend {trend_class}")
        )
    
    if comparison:
        card_content.append(
            html.Div(comparison, className="stat-comparison")
        )
    
    card_content.append(html.Div(subtitle, className="stat-description"))
    
    return html.Div(card_content, className="card stat-card card-content")

def create_section_header(title, subtitle, icon=""):
    """Créer un header de section cohérent"""
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '2rem', 'marginRight': '1rem'}),
            html.Div([
                html.H2(title, className="section-title"),
                html.P(subtitle, className="section-subtitle")
            ])
        ], className="section-header-content")
    ], className="section-header")

# ═══════════════════════════════════════════════════════════════════
#                 VISUALISATIONS INNOVANTES (Autres que viz1)
# ═══════════════════════════════════════════════════════════════════

def viz2_workload_radar(filtered_df):
    """VIZ 2: RADAR DE CHARGE - Anticiper les Semestres Difficiles"""
    semesters = sorted(filtered_df['Semestre'].unique())
    
    workload_data = []
    for sem in semesters:
        sem_data = filtered_df[filtered_df['Semestre'] == sem]
        
        unique_comps = set()
        for comp_list in sem_data['Competencies_List']:
            unique_comps.update(comp_list)
        
        level_counts = sem_data['Level'].value_counts()
        total_objectives = len(sem_data)
        modules_count = sem_data['EC'].nunique()
        
        workload_data.append({
            'Semestre': f'S{sem}',
            'Competences': len(unique_comps),
            'Objectifs': total_objectives,
            'Modules': modules_count,
            'Expert_Level': level_counts.get('E', 0),
        })
    
    wl_df = pd.DataFrame(workload_data)
    
    fig = go.Figure()
    
    for metric in ['Competences', 'Objectifs', 'Modules', 'Expert_Level']:
        fig.add_trace(go.Scatterpolar(
            r=wl_df[metric],
            theta=wl_df['Semestre'],
            fill='toself',
            name=metric,
            hovertemplate=f'{metric}: %{{r}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="CHARGE DE TRAVAIL - Comparaison par Semestre",
        polar=dict(radialaxis=dict(visible=True)),
        height=500,
        font=dict(size=11, family='Inter, sans-serif'),
        paper_bgcolor=COLORS['light']
    )
    
    return fig

def viz3_competency_heatmap(filtered_df):
    """VIZ 3: CARTE DE COMPÉTENCES - Heatmap avec Drill-down"""
    all_comps = set()
    for comps in filtered_df['Competencies_List']:
        all_comps.update(comps)
    all_comps = sorted(all_comps)[:15]
    
    modules = filtered_df['EC'].unique()[:20]
    
    matrix_data = []
    for module in modules:
        row = []
        module_data = filtered_df[filtered_df['EC'] == module]
        module_comps = set()
        for comps in module_data['Competencies_List']:
            module_comps.update(comps)
        
        for comp in all_comps:
            row.append(1 if comp in module_comps else 0)
        matrix_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=all_comps,
        y=[m[:30] for m in modules],
        colorscale=[[0, '#f0f0f0'], [1, COLORS['success']]],
        showscale=False,
        hovertemplate='<b>%{y}</b><br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="CARTE DE COMPETENCES - Ou developper quoi?",
        xaxis_title="Competences",
        yaxis_title="Modules",
        height=600,
        font=dict(size=10, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        xaxis={'side': 'top'}
    )
    
    return fig

def viz4_learning_flow(filtered_df):
    """VIZ 4: FLUX D'APPRENTISSAGE - Sankey Diagram"""
    nodes = []
    node_dict = {}
    links = {'source': [], 'target': [], 'value': [], 'color': []}
    
    semesters = sorted(filtered_df['Semestre'].unique())
    for sem in semesters:
        node_dict[f'S{sem}'] = len(nodes)
        nodes.append(f'S{sem}')
    
    ues = filtered_df['UE_Clean'].dropna().unique()
    for ue in ues:
        node_dict[ue] = len(nodes)
        nodes.append(ue)
    
    for sem in semesters:
        sem_data = filtered_df[filtered_df['Semestre'] == sem]
        ue_counts = sem_data['UE_Clean'].value_counts()
        
        for ue, count in ue_counts.items():
            if pd.notna(ue):
                links['source'].append(node_dict[f'S{sem}'])
                links['target'].append(node_dict[ue])
                links['value'].append(count)
                links['color'].append('rgba(59, 130, 246, 0.3)')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=[COLORS['primary'] if l.startswith('S') else COLORS['ue'].get(l, '#999') 
                   for l in nodes]
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    )])
    
    fig.update_layout(
        title="FLUX D'APPRENTISSAGE - Parcours Semestre > UE",
        font=dict(size=11, family='Inter, sans-serif'),
        height=500,
        paper_bgcolor=COLORS['light']
    )
    
    return fig

def viz5_critical_competencies(filtered_df):
    """VIZ 5: COMPÉTENCES CRITIQUES - Top Compétences pour Réussir"""
    comp_freq = {}
    comp_levels = {}
    
    for idx, row in filtered_df.iterrows():
        for comp in row['Competencies_List']:
            comp_freq[comp] = comp_freq.get(comp, 0) + 1
            if comp not in comp_levels:
                comp_levels[comp] = []
            comp_levels[comp].append(row['Level'])
    
    top_comps = sorted(comp_freq.items(), key=lambda x: x[1], reverse=True)[:12]
    
    fig = go.Figure()
    
    for comp, freq in top_comps:
        levels = comp_levels[comp]
        level_order = {'N': 1, 'A': 2, 'M': 3, 'E': 4}
        avg_level = np.mean([level_order[l] for l in levels if l in level_order])
        
        color = get_competency_color(comp)
        
        fig.add_trace(go.Bar(
            x=[freq],
            y=[comp],
            orientation='h',
            name=comp,
            marker_color=color,
            text=f'Niveau moy: {avg_level:.1f}',
            textposition='auto',
            hovertemplate=f'<b>{comp}</b><br>Frequence: %{{x}}<br>Niveau moyen: {avg_level:.1f}<extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        title="COMPETENCES CRITIQUES - Les plus importantes",
        xaxis_title="Frequence d'apparition",
        yaxis_title="",
        height=500,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light']
    )
    
    return fig

def create_sunburst_hierarchy(filtered_df):
    """
    VIZ 6: SUNBURST HIÉRARCHIQUE - Navigation Semestre → UE → Modules → Compétences
    
    OBJECTIF ÉTUDIANT: "Comment est organisé mon curriculum?"
    INNOVATION: Sunburst interactif avec drill-down circulaire
    """
    hierarchy_data = []
    
    for _, row in filtered_df.iterrows():
        sem = f"S{row['Semestre']}"
        ue = row['UE_Clean'] if pd.notna(row['UE_Clean']) else 'Autre'
        module = row['EC'][:30]
        
        for comp in row['Competencies_List']:
            hierarchy_data.append({
                'Semestre': sem,
                'UE': ue,
                'Module': module,
                'Competence': comp,
                'Value': 1
            })
    
    hierarchy_df = pd.DataFrame(hierarchy_data)
    
    if hierarchy_df.empty:
        return go.Figure()
    
    # Créer un dictionnaire de couleurs pour toutes les compétences
    color_map = {comp: get_competency_color(comp) for comp in hierarchy_df['Competence'].unique()}
    
    fig = px.sunburst(
        hierarchy_df,
        path=['Semestre', 'UE', 'Module', 'Competence'],
        values='Value',
        color='Competence',
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        title="VUE HIERARCHIQUE DU CURRICULUM - Navigation circulaire interactive",
        height=700,
        font=dict(size=11, family='Inter, sans-serif'),
        paper_bgcolor=COLORS['light']
    )
    
    # MODIFICATION: Retirer les pourcentages, afficher seulement le label
    fig.update_traces(
        textinfo="label",  # Seulement le nom, pas de pourcentage
        hovertemplate='<b>%{label}</b><br>Occurrences: %{value}<extra></extra>'
    )
    
    return fig
def viz7_statistics_dashboard(filtered_df):
    """VIZ 7: STATISTIQUES DYNAMIQUES - KPIs avec Comparaison"""
    total_modules = filtered_df['EC'].nunique()
    total_competencies = len(set([comp for comps in filtered_df['Competencies_List'] for comp in comps]))
    total_objectives = len(filtered_df)
    
    ue_counts = filtered_df.groupby('UE_Clean').size()
    avg_per_ue = ue_counts.mean() if len(ue_counts) > 0 else 0
    
    tc_count = sum(1 for comps in filtered_df['Competencies_List'] 
                   for c in comps if c.startswith('TC'))
    idu_count = sum(1 for comps in filtered_df['Competencies_List'] 
                    for c in comps if c.startswith('IDU'))
    
    return [
        create_stat_card(
            "Modules Totaux",
            total_modules,
            "Cours a suivre dans le curriculum",
            "📚",
            COLORS['primary'],
            comparison=f"Moyenne {avg_per_ue:.0f} par UE"
        ),
        create_stat_card(
            "Competences Uniques",
            total_competencies,
            "Competences a maitriser",
            "🎯",
            COLORS['success'],
            trend=5,
            comparison="Top 20% du programme"
        ),
        create_stat_card(
            "Objectifs",
            total_objectives,
            "Objectifs d'apprentissage totaux",
            "✅",
            COLORS['info'],
            comparison=f"Moy. {total_objectives/total_modules:.1f} par module"
        ),
        create_stat_card(
            "Equilibre TC/IDU",
            f"{tc_count}/{idu_count}",
            "Ratio Transversal / Discipline",
            "⚖️",
            COLORS['warning'],
            comparison="Curriculum equilibre" if abs(tc_count - idu_count) < 50 else "Desequilibre detecte"
        )
    ]

def viz8_network_graph(filtered_df):
    """
    VIZ 8: RÉSEAU DE RELATIONS 3D - Graph Interactif Expert
    
    OBJECTIF ÉTUDIANT: "Quelles compétences sont liées et comment se structurent-elles?"
    INNOVATION: Visualisation 3D interactive avec détection de communautés
    """
    import networkx as nx
    from networkx.algorithms import community
    
    # ═══ CONSTRUCTION DU GRAPHE ═══
    G = nx.Graph()
    
    for _, row in filtered_df.iterrows():
        comps = row['Competencies_List']
        for i, comp1 in enumerate(comps):
            for comp2 in comps[i+1:]:
                if G.has_edge(comp1, comp2):
                    G[comp1][comp2]['weight'] += 1
                else:
                    G.add_edge(comp1, comp2, weight=1)
    
    # Limiter aux nœuds les plus connectés
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:30]
    G_sub = G.subgraph(top_nodes).copy()
    
    if len(G_sub.nodes()) == 0:
        return go.Figure().add_annotation(
            text="Aucune relation trouvée dans les données filtrées",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # ═══ LAYOUT 3D AVEC FORCE-DIRECTED ═══
    # Utiliser un layout 3D optimisé
    pos = nx.spring_layout(G_sub, dim=3, k=2, iterations=50, seed=42)
    
    # ═══ DÉTECTION DE COMMUNAUTÉS ═══
    try:
        communities = community.greedy_modularity_communities(G_sub)
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
    except:
        node_to_community = {node: 0 for node in G_sub.nodes()}
    
    # ═══ PRÉPARATION DES DONNÉES 3D ═══
    
    # EDGES (arêtes) en 3D
    edge_x = []
    edge_y = []
    edge_z = []
    edge_weights = []
    
    for edge in G_sub.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_weights.append(G_sub[edge[0]][edge[1]]['weight'])
    
    # Normaliser les poids pour l'opacité
    max_weight = max(edge_weights) if edge_weights else 1
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            color='rgba(125, 125, 125, 0.3)',
            width=2
        ),
        hoverinfo='none',
        showlegend=False
    )
    
    # NODES (nœuds) en 3D
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_categories = []
    
    # Palette de couleurs pour les communautés
    community_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
        '#F8B88B', '#A8E6CF'
    ]
    
    for node in G_sub.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        degree = G_sub.degree(node)
        neighbors = list(G_sub.neighbors(node))
        
        # Texte au survol
        comp_color = get_competency_color(node)
        community_id = node_to_community.get(node, 0)
        
        hover_text = (
            f"<b>{node}</b><br>"
            f"━━━━━━━━━━━━━━━━<br>"
            f"🔗 Connexions: {degree}<br>"
            f"👥 Communauté: {community_id + 1}<br>"
            f"📊 Voisins: {', '.join(neighbors[:3])}"
            + (f"... +{len(neighbors)-3}" if len(neighbors) > 3 else "")
        )
        node_text.append(hover_text)
        
        # Couleur par communauté
        node_colors.append(community_colors[community_id % len(community_colors)])
        
        # Taille proportionnelle au degré
        node_sizes.append(10 + degree * 2)
        node_categories.append(f"Communauté {community_id + 1}")
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='white', width=2),
            opacity=0.9,
            symbol='circle'
        ),
        text=[node.split('-')[0] for node in G_sub.nodes()],
        textposition="top center",
        textfont=dict(size=10, color='black'),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    )
    
    # ═══ CRÉATION DU GRAPHIQUE 3D ═══
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # ═══ LAYOUT AVANCÉ ═══
    fig.update_layout(
        title={
            'text': "🌐 RÉSEAU DE RELATIONS 3D - Communautés de Compétences<br><sub>Rotation 3D | Zoom | Hover pour détails</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter, sans-serif'}
        },
        showlegend=False,
        hovermode='closest',
        height=700,
        paper_bgcolor=COLORS['light'],
        font=dict(size=11, family='Inter, sans-serif'),
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
            ),
            bgcolor='rgba(240, 240, 255, 0.3)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        # Annotations pour la légende
        annotations=[
            dict(
                text=(
                    f"<b>📊 Statistiques du réseau</b><br>"
                    f"Nœuds: {len(G_sub.nodes())}<br>"
                    f"Connexions: {len(G_sub.edges())}<br>"
                    f"Communautés: {len(set(node_to_community.values()))}<br>"
                    f"Densité: {nx.density(G_sub):.2%}"
                ),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#ccc',
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    return fig

def viz8_network_graph_advanced(filtered_df):
    """
    VIZ 8 ADVANCED: RÉSEAU 3D avec Sphères de Communautés
    """
    import networkx as nx
    from networkx.algorithms import community
    import numpy as np
    
    # Construction du graphe
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        comps = row['Competencies_List']
        for i, comp1 in enumerate(comps):
            for comp2 in comps[i+1:]:
                if G.has_edge(comp1, comp2):
                    G[comp1][comp2]['weight'] += 1
                else:
                    G.add_edge(comp1, comp2, weight=1)
    
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:30]
    G_sub = G.subgraph(top_nodes).copy()
    
    if len(G_sub.nodes()) == 0:
        return go.Figure().add_annotation(
            text="Aucune relation trouvée",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Layout 3D
    pos = nx.spring_layout(G_sub, dim=3, k=2, iterations=100, seed=42)
    
    # Détection de communautés
    try:
        communities = community.greedy_modularity_communities(G_sub)
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
    except:
        node_to_community = {node: 0 for node in G_sub.nodes()}
    
    # Créer les traces
    traces = []
    
    # EDGES avec gradient de couleur selon le poids
    for edge in G_sub.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        weight = G_sub[edge[0]][edge[1]]['weight']
        
        # Couleur et épaisseur selon le poids
        opacity = 0.1 + (weight / 10) * 0.4
        width = 1 + weight * 0.5
        
        traces.append(go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode='lines',
            line=dict(color=f'rgba(100, 100, 100, {opacity})', width=width),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # NODES groupés par communauté
    community_colors = {
        0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#FFA07A',
        4: '#98D8C8', 5: '#F7DC6F', 6: '#BB8FCE', 7: '#85C1E2'
    }
    
    for comm_id in set(node_to_community.values()):
        nodes_in_comm = [n for n in G_sub.nodes() if node_to_community[n] == comm_id]
        
        node_x = [pos[n][0] for n in nodes_in_comm]
        node_y = [pos[n][1] for n in nodes_in_comm]
        node_z = [pos[n][2] for n in nodes_in_comm]
        
        node_sizes = [10 + G_sub.degree(n) * 3 for n in nodes_in_comm]
        node_text_labels = [n.split('-')[0] for n in nodes_in_comm]
        
        hover_texts = [
            f"<b>{n}</b><br>Connexions: {G_sub.degree(n)}<br>Communauté: {comm_id + 1}"
            for n in nodes_in_comm
        ]
        
        traces.append(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=community_colors.get(comm_id, '#999'),
                line=dict(color='white', width=2),
                opacity=0.95
            ),
            text=node_text_labels,
            textposition="top center",
            textfont=dict(size=9, color='black'),
            hovertext=hover_texts,
            hoverinfo='text',
            name=f'Communauté {comm_id + 1}',
            showlegend=True
        ))
    
    # Créer le graphique
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title={
            'text': "🌐 RÉSEAU 3D EXPERT - Architecture des Compétences<br><sub>Cliquez-Glissez pour rotation | Molette pour zoom | Double-clic pour reset</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter, sans-serif'}
        },
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#ccc',
            borderwidth=1
        ),
        hovermode='closest',
        height=800,
        paper_bgcolor=COLORS['light'],
        font=dict(size=11, family='Inter, sans-serif'),
        scene=dict(
            xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            bgcolor='rgba(245, 245, 250, 0.5)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        annotations=[
            dict(
                text=(
                    f"<b>📊 Métriques</b><br>"
                    f"• Nœuds: {len(G_sub.nodes())}<br>"
                    f"• Liens: {len(G_sub.edges())}<br>"
                    f"• Communautés: {len(set(node_to_community.values()))}<br>"
                    f"• Densité: {nx.density(G_sub):.1%}<br>"
                    f"• Modularité: {community.modularity(G_sub, communities):.2f}" if 'communities' in locals() else ""
                ),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#3B82F6',
                borderwidth=2,
                font=dict(size=11, color='#1E293B')
            )
        ]
    )
    
    return fig
    """
    VIZ 8 ADVANCED: RÉSEAU 3D avec Sphères de Communautés
    """
    import networkx as nx
    from networkx.algorithms import community
    import numpy as np
    
    # Construction du graphe
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        comps = row['Competencies_List']
        for i, comp1 in enumerate(comps):
            for comp2 in comps[i+1:]:
                if G.has_edge(comp1, comp2):
                    G[comp1][comp2]['weight'] += 1
                else:
                    G.add_edge(comp1, comp2, weight=1)
    
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:30]
    G_sub = G.subgraph(top_nodes).copy()
    
    if len(G_sub.nodes()) == 0:
        return go.Figure().add_annotation(
            text="Aucune relation trouvée",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Layout 3D
    pos = nx.spring_layout(G_sub, dim=3, k=2, iterations=100, seed=42)
    
    # Détection de communautés
    try:
        communities = community.greedy_modularity_communities(G_sub)
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
    except:
        node_to_community = {node: 0 for node in G_sub.nodes()}
    
    # Créer les traces
    traces = []
    
    # EDGES avec gradient de couleur selon le poids
    for edge in G_sub.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        weight = G_sub[edge[0]][edge[1]]['weight']
        
        # Couleur et épaisseur selon le poids
        opacity = 0.1 + (weight / 10) * 0.4
        width = 1 + weight * 0.5
        
        traces.append(go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode='lines',
            line=dict(color=f'rgba(100, 100, 100, {opacity})', width=width),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # NODES groupés par communauté
    community_colors = {
        0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#FFA07A',
        4: '#98D8C8', 5: '#F7DC6F', 6: '#BB8FCE', 7: '#85C1E2'
    }
    
    for comm_id in set(node_to_community.values()):
        nodes_in_comm = [n for n in G_sub.nodes() if node_to_community[n] == comm_id]
        
        node_x = [pos[n][0] for n in nodes_in_comm]
        node_y = [pos[n][1] for n in nodes_in_comm]
        node_z = [pos[n][2] for n in nodes_in_comm]
        
        node_sizes = [10 + G_sub.degree(n) * 3 for n in nodes_in_comm]
        node_text_labels = [n.split('-')[0] for n in nodes_in_comm]
        
        hover_texts = [
            f"<b>{n}</b><br>Connexions: {G_sub.degree(n)}<br>Communauté: {comm_id + 1}"
            for n in nodes_in_comm
        ]
        
        traces.append(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=community_colors.get(comm_id, '#999'),
                line=dict(color='white', width=2),
                opacity=0.95
            ),
            text=node_text_labels,
            textposition="top center",
            textfont=dict(size=9, color='black'),
            hovertext=hover_texts,
            hoverinfo='text',
            name=f'Communauté {comm_id + 1}',
            showlegend=True
        ))
    
    # Créer le graphique
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title={
            'text': "🌐 RÉSEAU 3D EXPERT - Architecture des Compétences<br><sub>Cliquez-Glissez pour rotation | Molette pour zoom | Double-clic pour reset</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter, sans-serif'}
        },
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#ccc',
            borderwidth=1
        ),
        hovermode='closest',
        height=800,
        paper_bgcolor=COLORS['light'],
        font=dict(size=11, family='Inter, sans-serif'),
        scene=dict(
            xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            bgcolor='rgba(245, 245, 250, 0.5)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        annotations=[
            dict(
                text=(
                    f"<b>📊 Métriques</b><br>"
                    f"• Nœuds: {len(G_sub.nodes())}<br>"
                    f"• Liens: {len(G_sub.edges())}<br>"
                    f"• Communautés: {len(set(node_to_community.values()))}<br>"
                    f"• Densité: {nx.density(G_sub):.1%}<br>"
                    f"• Modularité: {community.modularity(G_sub, communities):.2f}" if 'communities' in locals() else ""
                ),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#3B82F6',
                borderwidth=2,
                font=dict(size=11, color='#1E293B')
            )
        ]
    )
    
    return fig
# ═══════════════════════════════════════════════════════════════════
#                          PANNEAU DE FILTRES
# ═══════════════════════════════════════════════════════════════════

def create_filter_section():
    """Panneau de filtres interactifs synchronisés"""
    return html.Div([
        html.Div([
            html.H3("🎛️ Filtres Interactifs", className="card-title"),
            html.P("Personnalisez votre vue pour explorer le curriculum", className="card-description")
        ], className="card-header"),
        html.Div([
            html.Div([
                html.Div([
                    html.Label("📅 Semestre", className="filter-label"),
                    dcc.Dropdown(
                        id='semester-filter',
                        options=[{'label': f'Semestre {s}', 'value': s} 
                                for s in sorted(df['Semestre'].unique())],
                        value=sorted(df['Semestre'].unique()),
                        multi=True,
                        className="filter-input",
                        placeholder="Tous les semestres..."
                    )
                ], className="filter-group"),
                
                html.Div([
                    html.Label("🎓 Unité d'Enseignement", className="filter-label"),
                    dcc.Dropdown(
                        id='ue-filter',
                        options=[{'label': ue, 'value': ue} 
                                for ue in sorted(df['UE_Clean'].dropna().unique())],
                        value=sorted(df['UE_Clean'].dropna().unique()),
                        multi=True,
                        className="filter-input",
                        placeholder="Toutes les UE..."
                    )
                ], className="filter-group"),
                
                html.Div([
                    html.Label("⭐ Niveau de Maîtrise", className="filter-label"),
                    dcc.Checklist(
                        id='level-filter',
                        options=[
                            {'label': ' Novice (N)', 'value': 'N'},
                            {'label': ' Apprenti (A)', 'value': 'A'},
                            {'label': ' Maître (M)', 'value': 'M'},
                            {'label': ' Expert (E)', 'value': 'E'}
                        ],
                        value=['N', 'A', 'M', 'E'],
                        inline=True,
                        className="filter-badges"
                    )
                ], className="filter-group"),
            ], className="filter-grid"),
            
            html.Div([
                html.Div([
                    html.Span("📊", style={'marginRight': '8px'}),
                    html.Span(id='filter-status', children="Tous les elements affiches")
                ], className="filter-status"),
                html.Button(
                    "🔄 Reinitialiser",
                    id='reset-filters',
                    className="btn btn-outline btn-sm"
                )
            ], className="filter-footer")
        ], className="card-content")
    ], className="card filter-panel")

# ═══════════════════════════════════════════════════════════════════
#                          LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

app.layout = html.Div([
    # Stores pour synchronisation
    dcc.Store(id='filtered-data-store'),
    dcc.Store(id='drill-down-level', data='global'),  # global, semester, module
    dcc.Store(id='selected-semester', data=None),
    dcc.Store(id='selected-module', data=None),
    
    # ═══ HEADER PROFESSIONNEL ═══
    html.Div([
        html.Div([
            html.Div([
                html.Div("🎓", className="logo-icon"),
                html.Div([
                    html.H1("ForestED Ultra Professional", className="header-title"),
                    html.P("Tableau de bord intelligent - Perspective Etudiant avec Drill-Down", 
                          className="header-subtitle")
                ])
            ], className="header-logo"),
            html.Div([
                html.Span(f"Derniere MAJ: {datetime.now().strftime('%d/%m/%Y')}", 
                         style={'fontSize': '14px', 'opacity': 0.9})
            ], className="header-actions")
        ], className="header-content")
    ], className="app-header"),
    
    # ═══ SECTION 1: STATISTIQUES DYNAMIQUES ═══
    html.Div([
        create_section_header(
            "Vue d'Ensemble",
            "Metriques cles de votre curriculum",
            "📊"
        ),
        html.Div(id='stats-dashboard', className="stats-grid")
    ], className="container section mt-xl"),
    
    # ═══ SECTION 2: FILTRES INTERACTIFS ═══
    html.Div([
        create_filter_section()
    ], className="container section"),
    
    # ═══ SECTION 3: MON PARCOURS AVEC DRILL-DOWN ═══
    html.Div([
        create_section_header(
            "Mon Parcours Personnel - Navigation Interactive",
            "Explorez par: Vue Globale → Semestre → Module → Objectifs",
            "📈"
        ),
        
        # Breadcrumb de navigation
        html.Div([
            html.Div(id='breadcrumb-nav', className="breadcrumb")
        ], style={'marginBottom': '20px'}),
        
        # Boutons de navigation
        html.Div([
            html.Button(
                "🏠 Vue Globale",
                id='btn-global',
                n_clicks=0,
                className="btn btn-primary btn-sm",
                style={'marginRight': '10px'}
            ),
            html.Button(
                "⬅️ Retour",
                id='btn-back',
                n_clicks=0,
                className="btn btn-outline btn-sm"
            )
        ], style={'marginBottom': '20px'}),
        
        # Graphique principal avec drill-down
        html.Div([
            html.Div([
                dcc.Graph(
                    id='viz1-timeline-drilldown', 
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="card-content")
        ], className="card")
    ], className="container section"),


    
    # ═══ SECTION 4: ANALYSE DÉTAILLÉE (Grid 2×2) ═══
# ═══ SECTION 4: ANALYSE DÉTAILLÉE (Grid 2×2 symétrique) ═══
html.Div([
    create_section_header(
        "Analyse Detaillee",
        "Explorez votre curriculum en profondeur",
        "🔍"
    ),

    html.Div([
        
        # Radar – Colonne 1 / Ligne 1
        html.Div([
            html.Div([
                dcc.Graph(id='viz2-radar', config={'displayModeBar': True, 'displaylogo': False})
            ], className="card-content")
        ], className="card", style={'width': '48%', 'marginBottom': '24px'}),

        # Compétences critiques – Colonne 2 / Ligne 1
        html.Div([
            html.Div([
                dcc.Graph(id='viz5-critical', config={'displayModeBar': True, 'displaylogo': False})
            ], className="card-content")
        ], className="card", style={'width': '48%', 'marginBottom': '24px'}),

        # Heatmap – Colonne 1 / Ligne 2
        html.Div([
            html.Div([
                dcc.Graph(id='viz3-heatmap', config={'displayModeBar': True, 'displaylogo': False})
            ], className="card-content")
        ], className="card", style={'width': '48%'}),

        # Sankey – Colonne 2 / Ligne 2
        html.Div([
            html.Div([
                dcc.Graph(id='viz4-flow', config={'displayModeBar': True, 'displaylogo': False})
            ], className="card-content")
        ], className="card", style={'width': '48%'})

    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-between',
        'gap': '24px'
    })
], className="container section"),

    
    # ═══ SECTION 5: VISUALISATIONS AVANCÉES ═══
# ═══ SECTION 5: VISUALISATIONS AVANCÉES (grid 2 colonnes symétriques) ═══
html.Div([
    create_section_header(
        "Visualisations Avancees",
        "Relations et hierarchies",
        "🌐"
    ),

    # Sélecteur du mode de visualisation (placé proprement dans la section)
    html.Div([
        html.Label(
            "Mode de visualisation :", 
            style={'fontWeight': 'bold', 'marginRight': '10px'}
        ),
        dcc.RadioItems(
            id='network-mode',
            options=[
                {'label': ' 3D Standard', 'value': 'standard'},
                {'label': ' 3D Expert (Communautés)', 'value': 'advanced'}
            ],
            value='advanced',
            inline=True,
            style={'marginBottom': '20px'}
        )
    ], style={'marginBottom': '20px'}),

    # Grille symétrique
    html.Div([

        # Sunburst — Colonne gauche
        html.Div([
            html.Div([
                dcc.Graph(
                    id='viz6-sunburst',
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="card-content")
        ], className="card", style={'width': '48%'}),

        # Network — Colonne droite
        html.Div([
            html.Div([
                dcc.Graph(
                    id='viz8-network',
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="card-content")
        ], className="card", style={'width': '48%'})

    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-between',
        'gap': '24px'
    })

], className="container section"),


    # ═══ FOOTER ═══
    html.Div([
        html.Div([
            html.P("💡 Astuce: Cliquez sur les heatmaps pour naviguer dans les niveaux de détail", 
                  style={'textAlign': 'center', 'margin': '0', 'fontSize': '14px', 'color': COLORS['dark']}),
            html.P("ForestED Ultra Professional - 2025 - Vue Etudiant Expert avec Drill-Down Interactif", 
                  style={'textAlign': 'center', 'margin': '10px 0 0 0', 'fontSize': '12px', 'color': '#64748b'})
        ])
    ], style={'padding': '40px 20px', 'backgroundColor': COLORS['light'], 'marginTop': '60px'})
    
], style={'fontFamily': 'Inter, sans-serif', 'backgroundColor': 'white', 'minHeight': '100vh'})

# ═══════════════════════════════════════════════════════════════════
#                          CALLBACKS
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    [Output('filtered-data-store', 'data'),
     Output('filter-status', 'children')],
    [Input('semester-filter', 'value'),
     Input('ue-filter', 'value'),
     Input('level-filter', 'value'),
     Input('reset-filters', 'n_clicks')],
    prevent_initial_call=False
)
def filter_data(semesters, ues, levels, reset_clicks):
    """Filtrer les donnees et synchroniser toutes les visualisations"""
    ctx = dash.callback_context
    
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-filters.n_clicks':
        filtered_df = df
        status = f"Tous les elements affiches ({len(df)} objectifs)"
    else:
        filtered_df = df.copy()
        
        if semesters:
            filtered_df = filtered_df[filtered_df['Semestre'].isin(semesters)]
        if ues:
            filtered_df = filtered_df[filtered_df['UE_Clean'].isin(ues)]
        if levels:
            filtered_df = filtered_df[filtered_df['Level'].isin(levels)]
        
        status = f"{len(filtered_df)} objectifs affiches sur {len(df)} total"
    
    return filtered_df.to_json(date_format='iso', orient='split'), status

from io import StringIO  # AJOUTER cet import en haut du fichier

# ═══════════════════════════════════════════════════════════════════
#          CALLBACK PRINCIPAL POUR LE DRILL-DOWN VIZ1 (CORRIGÉ)
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    [Output('viz1-timeline-drilldown', 'figure'),
     Output('drill-down-level', 'data'),
     Output('selected-semester', 'data'),
     Output('selected-module', 'data'),
     Output('breadcrumb-nav', 'children')],
    [Input('filtered-data-store', 'data'),
     Input('viz1-timeline-drilldown', 'clickData'),
     Input('btn-global', 'n_clicks'),
     Input('btn-back', 'n_clicks')],
    [State('drill-down-level', 'data'),
     State('selected-semester', 'data'),
     State('selected-module', 'data')],
    prevent_initial_call=False
)
def update_drilldown_viz1(filtered_data_json, clickData, btn_global_clicks, btn_back_clicks,
                          current_level, current_semester, current_module):
    """
    Gérer la navigation drill-down à 3 niveaux pour viz1
    """
    ctx = dash.callback_context
    
    # Préparer les données filtrées avec StringIO
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    # Créer la colonne All_Competencies
    filtered_df = filtered_df.assign(All_Competencies=filtered_df['Competencies_List'])
    
    # Déterminer quel input a déclenché le callback
    if not ctx.triggered:
        trigger_id = 'none'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialiser les variables de navigation
    new_level = current_level if current_level else 'global'
    new_semester = current_semester
    new_module = current_module
    
    # ═══ GESTION DES BOUTONS DE NAVIGATION ═══
    if trigger_id == 'btn-global':
        new_level = 'global'
        new_semester = None
        new_module = None
        print("DEBUG - Bouton global cliqué")
    
    elif trigger_id == 'btn-back':
        if current_level == 'module':
            new_level = 'semester'
            new_module = None
            print(f"DEBUG - Retour: module -> semester {current_semester}")
        elif current_level == 'semester':
            new_level = 'global'
            new_semester = None
            print("DEBUG - Retour: semester -> global")
    
    # ═══ GESTION DES CLICS SUR LE GRAPHIQUE ═══
    elif trigger_id == 'viz1-timeline-drilldown' and clickData and 'points' in clickData:
        point = clickData['points'][0]
        
        if current_level == 'global' or current_level is None:
            # Click sur un semestre → niveau 2
            clicked_semester = point.get('y')
            print(f"DEBUG - Click niveau global: semestre={clicked_semester}, type={type(clicked_semester)}")
            
            if clicked_semester is not None:
                # CORRECTION: Convertir en int si c'est une string
                try:
                    if isinstance(clicked_semester, str):
                        clicked_semester = int(clicked_semester)
                except (ValueError, TypeError):
                    print(f"DEBUG - Impossible de convertir le semestre: {clicked_semester}")
                    clicked_semester = None
                
                if clicked_semester is not None:
                    new_level = 'semester'
                    new_semester = clicked_semester
                    new_module = None
                    print(f"DEBUG - Passage au niveau semester: {new_semester} (type={type(new_semester)})")
            
        elif current_level == 'semester':
            # Click sur un module → niveau 3
            clicked_module = point.get('y')
            print(f"DEBUG - Click niveau semester: module={clicked_module}, type={type(clicked_module)}")
            
            if clicked_module is not None:
                new_level = 'module'
                new_module = clicked_module
                print(f"DEBUG - Passage au niveau module: {new_module}")
    
    # ═══ PRÉPARER LES DONNÉES POUR LES HEATMAPS ═══
    try:
        df_pivot_global, df_pivot_semester, df_pivot_module = compute_competency_counts_per_module(filtered_df)
        print(f"DEBUG - Niveaux disponibles: global={df_pivot_global.shape}, semester={df_pivot_semester.index.get_level_values('Semestre').unique().tolist()}")
    except Exception as e:
        print(f"ERROR - Erreur dans compute_competency_counts_per_module: {e}")
        fig = go.Figure().add_annotation(
            text=f"Erreur lors du calcul des données: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig, 'global', None, None, html.Div([html.Span("Erreur")])
    
    # ═══ GÉNÉRER LE GRAPHIQUE SELON LE NIVEAU ═══
    print(f"DEBUG - Génération graphique pour niveau={new_level}, semester={new_semester}, module={new_module}")
    
    if new_level == 'global':
        fig = create_heatmap_for_global(df_pivot_global)
        breadcrumb = html.Div([
            html.Span("🏠 Vue Globale", style={'fontWeight': 'bold', 'color': COLORS['primary']})
        ])
        print("DEBUG - Graphique global généré")
    
    elif new_level == 'semester' and new_semester is not None:
        # Vérifier que le semestre existe dans les données
        available_semesters = df_pivot_semester.index.get_level_values('Semestre').unique().tolist()
        print(f"DEBUG - Semestres disponibles: {available_semesters} (types: {[type(s) for s in available_semesters[:3]]})")
        print(f"DEBUG - Recherche du semestre: {new_semester} (type: {type(new_semester)})")
        
        if new_semester in available_semesters:
            fig = create_heatmap_for_semester(new_semester, df_pivot_semester, ylegend=True)
            breadcrumb = html.Div([
                html.Span("🏠 Vue Globale", style={'marginRight': '10px', 'color': '#666'}),
                html.Span(" / ", style={'marginRight': '10px'}),
                html.Span(f"📚 Semestre {new_semester}", style={'fontWeight': 'bold', 'color': COLORS['primary']})
            ])
            print(f"DEBUG - Graphique semester {new_semester} généré")
        else:
            # Fallback si le semestre n'existe pas
            print(f"DEBUG - Semestre {new_semester} non trouvé, retour à global")
            fig = create_heatmap_for_global(df_pivot_global)
            new_level = 'global'
            new_semester = None
            breadcrumb = html.Div([
                html.Span("🏠 Vue Globale", style={'fontWeight': 'bold', 'color': COLORS['primary']})
            ])
    
    elif new_level == 'module' and new_module is not None:
        # Vérifier que le module existe dans les données
        available_modules = df_pivot_module.index.get_level_values('EC').unique().tolist()
        print(f"DEBUG - Modules disponibles: {available_modules[:5]}... (total: {len(available_modules)})")
        
        if new_module in available_modules:
            fig = create_heatmap_for_module(new_module, df_pivot_module)
            breadcrumb = html.Div([
                html.Span("🏠 Vue Globale", style={'marginRight': '10px', 'color': '#666'}),
                html.Span(" / ", style={'marginRight': '10px'}),
                html.Span(f"📚 Semestre {new_semester}", style={'marginRight': '10px', 'color': '#666'}),
                html.Span(" / ", style={'marginRight': '10px'}),
                html.Span(f"🎓 {new_module[:40]}", style={'fontWeight': 'bold', 'color': COLORS['primary']})
            ])
            print(f"DEBUG - Graphique module {new_module} généré")
        else:
            # Fallback si le module n'existe pas
            print(f"DEBUG - Module {new_module} non trouvé, retour à semester")
            if new_semester and new_semester in df_pivot_semester.index.get_level_values('Semestre'):
                fig = create_heatmap_for_semester(new_semester, df_pivot_semester, ylegend=True)
                new_level = 'semester'
                new_module = None
                breadcrumb = html.Div([
                    html.Span("🏠 Vue Globale", style={'marginRight': '10px', 'color': '#666'}),
                    html.Span(" / ", style={'marginRight': '10px'}),
                    html.Span(f"📚 Semestre {new_semester}", style={'fontWeight': 'bold', 'color': COLORS['primary']})
                ])
            else:
                fig = create_heatmap_for_global(df_pivot_global)
                new_level = 'global'
                new_semester = None
                breadcrumb = html.Div([
                    html.Span("🏠 Vue Globale", style={'fontWeight': 'bold', 'color': COLORS['primary']})
                ])
    
    else:
        # Fallback: vue globale
        print("DEBUG - Fallback vers vue globale")
        fig = create_heatmap_for_global(df_pivot_global)
        new_level = 'global'
        new_semester = None
        new_module = None
        breadcrumb = html.Div([
            html.Span("🏠 Vue Globale", style={'fontWeight': 'bold', 'color': COLORS['primary']})
        ])
    
    print(f"DEBUG - Retour: level={new_level}, semester={new_semester}, module={new_module}")
    return fig, new_level, new_semester, new_module, breadcrumb
# ═══════════════════════════════════════════════════════════════════
#          CALLBACKS POUR LES AUTRES VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    Output('stats-dashboard', 'children'),
    Input('filtered-data-store', 'data')
)
def update_stats(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz7_statistics_dashboard(filtered_df)

@app.callback(
    Output('viz2-radar', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz2(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz2_workload_radar(filtered_df)

@app.callback(
    Output('viz3-heatmap', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz3(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz3_competency_heatmap(filtered_df)

@app.callback(
    Output('viz4-flow', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz4(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz4_learning_flow(filtered_df)

@app.callback(
    Output('viz5-critical', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz5(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz5_critical_competencies(filtered_df)

@app.callback(
    Output('viz6-sunburst', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz6(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return create_sunburst_hierarchy(filtered_df)

@app.callback(
    Output('viz8-network', 'figure'),
    [Input('filtered-data-store', 'data'),
     Input('network-mode', 'value')]
)
def update_viz8(filtered_data_json, mode):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    if mode == 'advanced':
        return viz8_network_graph_advanced(filtered_df)
    else:
        return viz8_network_graph(filtered_df)
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    if mode == 'advanced':
        return viz8_network_graph(filtered_df)
    else:
        return viz8_network_graph(filtered_df)
# ═══════════════════════════════════════════════════════════════════
#                          LANCEMENT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # CORRECTION: Désactiver le mode debug pour éviter l'erreur avec Python 3.14
    app.run_server(debug=False, host='0.0.0.0', port=8050)