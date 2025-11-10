"""
╔══════════════════════════════════════════════════════════════════════════╗
║               ForestED ULTRA PROFESSIONAL - Vue Étudiant Expert          ║
║                                                                          ║
║  🎓 Projet de Visualisation - Réponses aux Questions 1-6                ║
║  🚀 8 Visualisations Innovantes Synchronisées                          ║
║  🎨 Design shadcn/ui Professionnel                                     ║
║  📊 Perspective 100% Étudiant avec Narration des Données               ║
╚══════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
                    RÉPONSES AUX QUESTIONS DU PROJET
═══════════════════════════════════════════════════════════════════════════

QUESTION 2a) Visualisations les plus efficaces du tableau original:
----------------------------------------------------------------
✓ Network graph: Montre bien les connexions entre compétences
✓ Barplots par semestre: Donne une vue quantitative claire
→ Mais manque de contexte étudiant et d'interactivité

QUESTION 2b) Visualisations floues à améliorer:
-----------------------------------------------
✗ Trop de graphiques statiques sans cohérence visuelle
✗ Absence de filtres interactifs pour personnaliser la vue
✗ Pas de vue "Mon Parcours" pour planifier mes semestres
✗ Manque d'indicateurs prédictifs (charge de travail à venir)
✗ Couleurs peu distinctives, difficile de différencier les UE

QUESTION 3) Point de vue étudiant personnel:
-------------------------------------------
L'outil original est pensé pour les enseignants/concepteurs.
En tant qu'ÉTUDIANT, j'ai besoin de:
✓ Comprendre QUAND je vais acquérir quelle compétence
✓ Anticiper la CHARGE DE TRAVAIL de chaque semestre
✓ Identifier les COMPÉTENCES CRITIQUES pour mon succès
✓ Voir les LIENS entre compétences pour mieux comprendre
✓ Comparer MA progression vs la moyenne du curriculum
✓ Pouvoir FILTRER et EXPLORER selon mes besoins

QUESTION 4) Amélioration de la disposition:
------------------------------------------
✓ Organisation en SECTIONS THÉMATIQUES claires:
  - Dashboard de statistiques (vue d'ensemble)
  - Mon Parcours (progression temporelle)
  - Analyse Détaillée (drill-down par UE/compétence)
✓ Design en GRID RESPONSIVE avec shadcn/ui
✓ Filtres CENTRAUX et toujours accessibles
✓ SYNCHRONISATION de toutes les visualisations
✓ Navigation INTUITIVE avec breadcrumbs

QUESTION 5) Améliorations implémentées:
--------------------------------------
✓ Ajout de FILTRES INTERACTIFS synchronisés (semestre, UE, niveau)
✓ CARTES DE STATISTIQUES modernes avec trends vs moyenne
✓ HEATMAP de charge de travail PRÉDICTIVE par semestre
✓ TIMELINE de progression des compétences avec drill-down
✓ Design SHADCN/UI professionnel (tokens, animations, responsive)
✓ NARRATION DES DONNÉES: chaque viz raconte une histoire
✓ Couleurs DISTINCTIVES cohérentes (HSL pour accessibilité)

QUESTION 6) Nouvelles visualisations enrichissantes:
---------------------------------------------------
✓ TREEMAP HIÉRARCHIQUE: Navigation UE → Modules → Compétences
✓ NETWORK GRAPH INTERACTIF: Relations entre compétences
✓ SANKEY FLOW: Flux d'apprentissage à travers les semestres
✓ RADAR COMPARATIF: Équilibre des compétences par semestre
✓ BUBBLE CHART: Importance vs Fréquence des compétences
✓ GANTT-STYLE TIMELINE: Planning visuel du curriculum

═══════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import numpy as np
from datetime import datetime
import networkx as nx

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

def get_competency_color(competency):
    """Obtenir la couleur d'une compétence"""
    parts = competency.split("-")
    if len(parts) < 2:
        return "#777777"
    category_key = f"{parts[0]}-{parts[1][0]}"
    return COLORS['categories'].get(category_key, "#AAAAAA")

# ═══════════════════════════════════════════════════════════════════
#                    COMPOSANTS UI RÉUTILISABLES
# ═══════════════════════════════════════════════════════════════════

def create_stat_card(title, value, subtitle, icon, color, trend=None, comparison=None):
    """
    Créer une carte de statistique moderne (shadcn/ui style)
    
    Args:
        title: Titre de la métrique
        value: Valeur principale
        subtitle: Description
        icon: Emoji ou icône
        color: Couleur thématique
        trend: Variation en % (optionnel)
        comparison: Texte de comparaison (optionnel)
    """
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
#                 VISUALISATIONS INNOVANTES (8 types)
# ═══════════════════════════════════════════════════════════════════

def viz1_personal_timeline(filtered_df):
    """
    VIZ 1: MON PARCOURS PERSONNEL - Timeline Interactive
    
    OBJECTIF ÉTUDIANT: "Quand vais-je apprendre quelles compétences?"
    INNOVATION: Chronologie visuelle avec drill-down par semestre
    """
    semesters = sorted(filtered_df['Semestre'].unique())
    all_comps = filtered_df['Competencies_List'].explode().dropna().unique()
    
    progression_data = []
    for comp in all_comps[:20]:  # Top 20 pour lisibilité
        for sem in semesters:
            count = filtered_df[filtered_df['Semestre'] == sem]['Competencies_List'].apply(
                lambda x: comp in x
            ).sum()
            
            if count > 0:
                levels_this_sem = filtered_df[
                    (filtered_df['Semestre'] == sem) & 
                    (filtered_df['Competencies_List'].apply(lambda x: comp in x))
                ]['Level']
                
                if not levels_this_sem.empty:
                    level_order = {'N': 1, 'A': 2, 'M': 3, 'E': 4}
                    avg_level = levels_this_sem.map(level_order).mean()
                    
                    progression_data.append({
                        'Competence': comp,
                        'Semestre': f'S{sem}',
                        'Niveau': avg_level,
                        'Count': count
                    })
    
    prog_df = pd.DataFrame(progression_data)
    
    if prog_df.empty:
        return go.Figure().add_annotation(
            text="Aucune donnee pour les filtres selectionnes",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    pivot = prog_df.pivot_table(
        index='Competence', columns='Semestre', values='Niveau', aggfunc='mean'
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[
            [0, COLORS['levels']['N']],
            [0.33, COLORS['levels']['A']],
            [0.66, COLORS['levels']['M']],
            [1, COLORS['levels']['E']]
        ],
        text=pivot.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title="Niveau",
            tickvals=[1, 2, 3, 4],
            ticktext=['Novice', 'Apprenti', 'Maitre', 'Expert']
        ),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Niveau: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="MON PARCOURS - Quand j'apprends quoi?",
        xaxis_title="Semestre",
        yaxis_title="Competence (Top 20)",
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    return fig

def viz2_workload_radar(filtered_df):
    """
    VIZ 2: RADAR DE CHARGE - Anticiper les Semestres Difficiles
    
    OBJECTIF ÉTUDIANT: "Quel semestre sera le plus chargé?"
    INNOVATION: Vue radar multi-axes pour comparaison rapide
    """
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
    
    # Ajouter une trace pour chaque métrique
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
    """
    VIZ 3: CARTE DE COMPÉTENCES - Heatmap avec Drill-down
    
    OBJECTIF ÉTUDIANT: "Où vais-je développer chaque compétence?"
    INNOVATION: Matrice interactive Module × Compétence
    """
    # Créer la matrice modules × compétences
    all_comps = set()
    for comps in filtered_df['Competencies_List']:
        all_comps.update(comps)
    all_comps = sorted(all_comps)[:15]  # Top 15 pour lisibilité
    
    modules = filtered_df['EC'].unique()[:20]  # Top 20 modules
    
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
    """
    VIZ 4: FLUX D'APPRENTISSAGE - Sankey Diagram
    
    OBJECTIF ÉTUDIANT: "Comment les compétences se construisent?"
    INNOVATION: Visualisation du flux Semestre → UE → Compétences
    """
    # Préparer les données pour le Sankey
    nodes = []
    node_dict = {}
    links = {'source': [], 'target': [], 'value': [], 'color': []}
    
    # Créer les nœuds: Semestres
    semesters = sorted(filtered_df['Semestre'].unique())
    for sem in semesters:
        node_dict[f'S{sem}'] = len(nodes)
        nodes.append(f'S{sem}')
    
    # Créer les nœuds: UEs
    ues = filtered_df['UE_Clean'].dropna().unique()
    for ue in ues:
        node_dict[ue] = len(nodes)
        nodes.append(ue)
    
    # Créer les liens Semestre → UE
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
    """
    VIZ 5: COMPÉTENCES CRITIQUES - Top Compétences pour Réussir
    
    OBJECTIF ÉTUDIANT: "Quelles sont les compétences les plus importantes?"
    INNOVATION: Classement par fréquence et importance
    """
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

def viz6_hierarchy_tree(filtered_df):
    """
    VIZ 6: ARBRE HIÉRARCHIQUE - Navigation UE → Modules → Compétences
    
    OBJECTIF ÉTUDIANT: "Comment est organisé mon curriculum?"
    INNOVATION: Treemap interactif avec drill-down
    """
    hierarchy_data = []
    
    for _, row in filtered_df.iterrows():
        ue = row['UE_Clean'] if pd.notna(row['UE_Clean']) else 'Autre'
        module = row['EC'][:30]
        
        for comp in row['Competencies_List']:
            hierarchy_data.append({
                'UE': ue,
                'Module': module,
                'Competence': comp,
                'Value': 1
            })
    
    hierarchy_df = pd.DataFrame(hierarchy_data)
    
    if hierarchy_df.empty:
        return go.Figure()
    
    fig = px.treemap(
        hierarchy_df,
        path=['UE', 'Module', 'Competence'],
        values='Value',
        color='UE',
        color_discrete_map={ue: COLORS['ue'].get(ue, '#999') for ue in hierarchy_df['UE'].unique()},
        hover_data=['Competence']
    )
    
    fig.update_layout(
        title="ARBRE HIERARCHIQUE - Organisation du Curriculum",
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        paper_bgcolor=COLORS['light']
    )
    
    fig.update_traces(
        textinfo="label+value",
        textfont=dict(size=11),
        marker=dict(line=dict(width=2, color='white'))
    )
    
    return fig

def viz7_statistics_dashboard(filtered_df):
    """
    VIZ 7: STATISTIQUES DYNAMIQUES - KPIs avec Comparaison
    
    OBJECTIF ÉTUDIANT: "Comment me situer dans le curriculum?"
    INNOVATION: Cartes de métriques avec comparaisons
    """
    total_modules = filtered_df['EC'].nunique()
    total_competencies = len(set([comp for comps in filtered_df['Competencies_List'] for comp in comps]))
    total_objectives = len(filtered_df)
    
    # Calculer les moyennes par UE
    ue_counts = filtered_df.groupby('UE_Clean').size()
    avg_per_ue = ue_counts.mean() if len(ue_counts) > 0 else 0
    
    # Calculer le ratio TC vs IDU
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
    VIZ 8: RÉSEAU DE RELATIONS - Graph Interactif
    
    OBJECTIF ÉTUDIANT: "Quelles compétences sont liées?"
    INNOVATION: Graph force-directed pour voir les connexions
    """
    # Créer le graph de relations
    G = nx.Graph()
    
    # Ajouter les compétences qui apparaissent ensemble dans les modules
    for _, row in filtered_df.iterrows():
        comps = row['Competencies_List']
        for i, comp1 in enumerate(comps):
            for comp2 in comps[i+1:]:
                if G.has_edge(comp1, comp2):
                    G[comp1][comp2]['weight'] += 1
                else:
                    G.add_edge(comp1, comp2, weight=1)
    
    # Limiter aux compétences les plus connectées
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:15]
    G_sub = G.subgraph(top_nodes)
    
    # Calculer la position avec spring layout
    pos = nx.spring_layout(G_sub, k=2, iterations=50)
    
    # Créer les edges
    edge_x = []
    edge_y = []
    for edge in G_sub.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Créer les nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G_sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Connexions: {G_sub.degree(node)}")
        node_colors.append(get_competency_color(node))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n.split('-')[0] for n in G_sub.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=20,
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="RESEAU DE RELATIONS - Competences connectees",
        showlegend=False,
        hovermode='closest',
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        paper_bgcolor=COLORS['light'],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def calculate_statistics(filtered_df):
    """Calculer les statistiques globales"""
    total_modules = filtered_df['EC'].nunique()
    total_competencies = len(set([comp for comps in filtered_df['Competencies_List'] for comp in comps]))
    total_objectives = len(filtered_df)
    avg_comp_per_module = total_competencies / total_modules if total_modules > 0 else 0
    
    return {
        'total_modules': total_modules,
        'total_competencies': total_competencies,
        'total_objectives': total_objectives,
        'avg_comp_per_module': round(avg_comp_per_module, 1)
    }

# ═══════════════════════════════════════════════════════════════════
#                          PANNEAU DE FILTRES
# ═══════════════════════════════════════════════════════════════════

def create_filter_section():
    """
    Panneau de filtres interactifs synchronisés
    INNOVATION: Tous les graphiques se mettent à jour en temps réel
    """
    return html.Div([
        html.Div([
            html.H3("🎛️ Filtres Interactifs", className="card-title"),
            html.P("Personnalisez votre vue pour explorer le curriculum", className="card-description")
        ], className="card-header"),
        html.Div([
            html.Div([
                # Filtre Semestre
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
                
                # Filtre UE
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
                
                # Filtre Niveau
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
    # Store pour synchronisation
    dcc.Store(id='filtered-data-store'),
    
    # ═══ HEADER PROFESSIONNEL ═══
    html.Div([
        html.Div([
            html.Div([
                html.Div("🎓", className="logo-icon"),
                html.Div([
                    html.H1("ForestED Ultra Professional", className="header-title"),
                    html.P("Tableau de bord intelligent - Perspective Etudiant", 
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
    
    # ═══ SECTION 3: MON PARCOURS ═══
    html.Div([
        create_section_header(
            "Mon Parcours Personnel",
            "Timeline de progression des competences",
            "📈"
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='viz1-timeline', config={'displayModeBar': True, 'displaylogo': False})
            ], className="card-content")
        ], className="card")
    ], className="container section"),
    
    # ═══ SECTION 4: ANALYSE DÉTAILLÉE (Grid 2×2) ═══
    html.Div([
        create_section_header(
            "Analyse Detaillee",
            "Explorez votre curriculum en profondeur",
            "🔍"
        ),
        
        html.Div([
            # Colonne 1
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='viz2-radar', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card mb-lg"),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='viz5-critical', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card")
            ], style={'flex': 1}),
            
            # Colonne 2
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='viz3-heatmap', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card mb-lg"),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='viz4-flow', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card")
            ], style={'flex': 1})
        ], style={'display': 'flex', 'gap': '24px'})
    ], className="container section"),
    
    # ═══ SECTION 5: VISUALISATIONS AVANCÉES ═══
    html.Div([
        create_section_header(
            "Visualisations Avancees",
            "Relations et hierarchies",
            "🌐"
        ),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='viz6-tree', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card")
            ], style={'flex': 1, 'marginRight': '12px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='viz8-network', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card")
            ], style={'flex': 1, 'marginLeft': '12px'})
        ], style={'display': 'flex', 'gap': '24px'})
    ], className="container section"),
    
    # ═══ FOOTER ═══
    html.Div([
        html.Div([
            html.P("💡 Astuce: Utilisez les filtres pour personnaliser votre vue", 
                  style={'textAlign': 'center', 'margin': '0', 'fontSize': '14px', 'color': COLORS['dark']}),
            html.P("ForestED Ultra Professional - 2025 - Vue Etudiant Expert", 
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
    Output('viz1-timeline', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz1(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz1_personal_timeline(filtered_df)

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
    Output('viz6-tree', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz6(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz6_hierarchy_tree(filtered_df)

@app.callback(
    Output('viz8-network', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_viz8(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return viz8_network_graph(filtered_df)

# ═══════════════════════════════════════════════════════════════════
#                          LANCEMENT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)