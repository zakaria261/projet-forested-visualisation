"""
ForestED ULTRA PROFESSIONAL - Vue Étudiant OPTIMISÉE
Dashboard interactif avec drill-down et réseau 3D
Version Expert avec layout optimisé et visualisations améliorées
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from datetime import datetime
import networkx as nx
from io import StringIO

# ═══════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder='assets',
    title="ForestED - Tableau de bord étudiant"
)

df = pd.read_excel("all_Competence.xlsx")

# ═══════════════════════════════════════════════════════════════════
#                    TRAITEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════

def parse_competencies(row):
    comps = []
    for i in range(1, 13):
        col = f"Competencies.{i}"
        if col in row.index and pd.notna(row[col]):
            comps.append(row[col])
    return comps

df["Competencies_List"] = df.apply(parse_competencies, axis=1)

def extract_ue(ue_string):
    if pd.isna(ue_string):
        return None
    parts = [p.strip() for p in str(ue_string).split("-")]
    for p in parts:
        if p.startswith("UE"):
            return p
    return None

df["UE_Clean"] = df["UE"].apply(extract_ue)
df_expanded = df.copy()
df_expanded = df_expanded.assign(All_Competencies=df_expanded["Competencies_List"])

# ═══════════════════════════════════════════════════════════════════
#                      DESIGN SYSTEM COULEURS
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#3B82F6",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#06B6D4",
    "light": "#F8FAFC",
    "dark": "#1E293B",
    "ue": {
        "UE1": "#E25012",
        "UE2": "#E28A12",
        "UE3": "#155992",
        "UE4": "#0C9A61",
    },
    "levels": {
        "N": "#BFDBFE",
        "A": "#60A5FA",
        "M": "#2563EB",
        "E": "#1E40AF",
    },
    "categories": {
        "IDU-1": "#FF513F",
        "IDU-2": "#FFA03F",
        "IDU-3": "#DC3785",
        "IDU-4": "#FFC83F",
        "TC-1": "#2DB593",
        "TC-2": "#3A6CB7",
        "TC-3": "#65E038",
        "TC-4": "#5740BD",
    },
}

def get_competency_color(competency, level=None):
    parts = str(competency).split("-")
    if len(parts) < 2:
        return "#777777"
    key = f"{parts[0]}-{parts[1][0]}"
    return COLORS["categories"].get(key, "#AAAAAA")

# ═══════════════════════════════════════════════════════════════════
#           PIVOTS POUR HEATMAPS (3 niveaux)
# ═══════════════════════════════════════════════════════════════════

def compute_competency_counts_per_module(df_input):
    df_work = df_input.copy()
    if "Goal" not in df_work.columns:
        df_work = df_work.assign(Goal=df_work.index.astype(str))

    df_work = df_work.assign(All_Competencies=df_work["Competencies_List"])
    df_exploded = df_work.explode("All_Competencies").reset_index(drop=True)

    df_count = (
        df_exploded
        .groupby(["Semestre", "EC", "All_Competencies"], as_index=False)
        .size()
    )

    df_pivot_global = (
        df_exploded
        .groupby(["Semestre", "All_Competencies"], as_index=False)
        .size()
        .pivot(index="Semestre", columns="All_Competencies", values="size")
        .fillna(0)
        .astype("Float64")
    )

    df_pivot_semester = (
        df_count
        .pivot(index=["Semestre", "EC"], columns="All_Competencies", values="size")
        .fillna(0)
        .astype("Float64")
    )

    df_pivot_module = (
        df_exploded
        .pivot(index=["Semestre", "EC", "Goal"], columns="All_Competencies", values="Level")
        .fillna(0)
    )

    return df_pivot_global, df_pivot_semester, df_pivot_module

# ═══════════════════════════════════════════════════════════════════
#                       HEATMAPS DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════

def create_heatmap_for_global(df_pivot_global):
    fig = go.Figure()
    for j in range(df_pivot_global.shape[1]):
        comp = df_pivot_global.columns[j]
        color_comp = get_competency_color(comp)
        color_scale = ["white", color_comp]
        df_comp = df_pivot_global.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_pivot_global.columns,
                y=df_pivot_global.index,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                hovertemplate="<b>Semestre %{y}</b><br>%{x}<br>Occurrences: %{z}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Mes semestres et compétences<br><sub>Cliquez sur un semestre pour voir les modules</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis_title="Semestres",
        yaxis=dict(
            type="category",
            autorange="reversed",
            showgrid=False,
            tickfont=dict(size=12),
        ),
        height=600,
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=100, r=60, t=120, b=150),
        clickmode="event+select",
    )

    return fig

def create_heatmap_for_semester(semestre, df_pivot_semester, ylegend=True):
    df_used = df_pivot_semester.xs(semestre, level="Semestre")

    fig = go.Figure()
    z_min = 0
    z_max = df_pivot_semester.max().max()

    for j in range(df_used.shape[1]):
        comp = df_used.columns[j]
        color_comp = get_competency_color(comp)
        color_scale = ["white", color_comp]
        df_comp = df_used.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=df_used.index,
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1,
                ygap=1,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Occurrences: %{z}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Modules du semestre {semestre}<br><sub>Cliquez sur un module pour voir les objectifs</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        yaxis_title="Modules (EC)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(500, len(df_used.index) * 35 + 180),
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor=COLORS["light"],
        margin=dict(l=280, r=60, t=120, b=180),
        clickmode="event+select",
    )
    if not ylegend:
        fig.update_yaxes(showticklabels=False)
    return fig

def create_heatmap_for_module(module, df_pivot_module):
    df_reorganized = df_pivot_module.replace({"N": 1, "A": 2, "M": 3, "E": 4})
    df_used = (
        df_reorganized
        .xs(module, level="EC")
        .rename(columns={np.nan: "No Competency"})
        .astype("Int64")
    )

    fig = go.Figure()
    z_min = 0
    z_max = 4

    for j in range(df_used.shape[1]):
        comp = df_used.columns[j]
        color_comp = get_competency_color(comp)
        color_scale = ["white", color_comp]
        df_comp = df_used.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=df_used.index.get_level_values("Goal") if hasattr(df_used.index, "get_level_values") else df_used.index,
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1,
                ygap=1,
                hovertemplate="<b>Objectif %{y}</b><br>%{x}<br>Niveau: %{z}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{module} - Objectifs pédagogiques<br><sub>Niveaux: 1=Novice, 2=Apprenti, 3=Maîtrise, 4=Expert</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        yaxis_title="Objectifs pédagogiques",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=700,
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor=COLORS["light"],
        margin=dict(l=280, r=60, t=120, b=180),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════
#                    COMPOSANTS UI
# ═══════════════════════════════════════════════════════════════════

def create_stat_card(title, value, subtitle, color, trend=None, comparison=None):
    content = [
        html.Div([
            html.Div(title, className="stat-label"),
        ], className="stat-header"),
        html.Div(str(value), className="stat-value", style={"color": color}),
    ]
    if trend is not None:
        trend_class = "stat-trend-up" if trend > 0 else "stat-trend-down"
        content.append(
            html.Div(
                [
                    html.Span("↑" if trend > 0 else "↓"),
                    html.Span(f"{abs(trend)}%"),
                ],
                className=f"stat-trend {trend_class}",
            )
        )
    if comparison:
        content.append(html.Div(comparison, className="stat-description"))
    else:
        content.append(html.Div(subtitle, className="stat-description"))

    return html.Div(content, className="card stat-card card-content")

def create_section_header(title, subtitle):
    return html.Div(
        html.Div(
            [
                html.H2(title, className="section-title"),
                html.P(subtitle, className="section-subtitle"),
            ],
            className="section-header-content",
        ),
        className="section-header",
    )

def create_filter_section():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Filtres", className="sidebar-title"),
                    html.P(
                        "Personnalisez votre vue",
                        className="sidebar-subtitle",
                    ),
                ],
                className="sidebar-header",
            ),
            html.Div(
                [
                    # Semestre
                    html.Div(
                        [
                            html.Label(
                                "Semestres",
                                className="filter-label",
                            ),
                            dcc.Dropdown(
                                id="semester-filter",
                                options=[
                                    {"label": f"S{s}", "value": s}
                                    for s in sorted(df["Semestre"].unique())
                                ],
                                value=sorted(df["Semestre"].unique()),
                                multi=True,
                                className="filter-input",
                                placeholder="Tous les semestres",
                            ),
                        ],
                        className="filter-group",
                    ),
                    # UE
                    html.Div(
                        [
                            html.Label(
                                "Unités d'enseignement",
                                className="filter-label",
                            ),
                            dcc.Dropdown(
                                id="ue-filter",
                                options=[
                                    {"label": ue, "value": ue}
                                    for ue in sorted(df["UE_Clean"].dropna().unique())
                                ],
                                value=sorted(df["UE_Clean"].dropna().unique()),
                                multi=True,
                                className="filter-input",
                                placeholder="Toutes les UE",
                            ),
                        ],
                        className="filter-group",
                    ),
                    # Niveaux avec toggles modernes
                    html.Div(
                        [
                            html.Label(
                                "Niveaux de maîtrise",
                                className="filter-label",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="level-filter",
                                                options=[
                                                    {"label": html.Div([
                                                        html.Span("N", className="level-badge level-n"),
                                                        html.Span("Novice", className="level-text")
                                                    ], className="level-option"), "value": "N"},
                                                    {"label": html.Div([
                                                        html.Span("A", className="level-badge level-a"),
                                                        html.Span("Apprenti", className="level-text")
                                                    ], className="level-option"), "value": "A"},
                                                    {"label": html.Div([
                                                        html.Span("M", className="level-badge level-m"),
                                                        html.Span("Maîtrise", className="level-text")
                                                    ], className="level-option"), "value": "M"},
                                                    {"label": html.Div([
                                                        html.Span("E", className="level-badge level-e"),
                                                        html.Span("Expert", className="level-text")
                                                    ], className="level-option"), "value": "E"},
                                                ],
                                                value=["N", "A", "M", "E"],
                                                className="level-checklist",
                                            ),
                                        ],
                                    ),
                                ],
                                className="level-filter-container",
                            ),
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Button(
                                [
                                    html.Span("↻", style={"marginRight": "6px", "fontSize": "14px"}),
                                    "Réinitialiser"
                                ],
                                id="reset-filters",
                                className="btn btn-reset",
                            ),
                        ],
                        className="filter-footer",
                    ),
                    html.Div(
                        id="filter-status",
                        className="filter-status",
                    ),
                ],
                className="sidebar-content",
            ),
        ],
        className="sidebar-panel",
    )

# ═══════════════════════════════════════════════════════════════════
#                 AUTRES VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════

def viz2_workload_radar(filtered_df):
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée à afficher. Veuillez ajuster vos filtres.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=650,
        )
        return fig
    
    semesters = sorted(filtered_df["Semestre"].unique())
    workload_data = []
    for sem in semesters:
        sem_data = filtered_df[filtered_df["Semestre"] == sem]
        unique_comps = set()
        for comp_list in sem_data["Competencies_List"]:
            unique_comps.update(comp_list)
        level_counts = sem_data["Level"].value_counts()
        total_objectives = len(sem_data)
        modules_count = sem_data["EC"].nunique()
        workload_data.append(
            {
                "Semestre": f"S{sem}",
                "Compétences": len(unique_comps),
                "Objectifs": total_objectives,
                "Modules": modules_count,
                "Expert_Level": level_counts.get("E", 0),
            }
        )
    wl_df = pd.DataFrame(workload_data)
    fig = go.Figure()
    for metric in ["Compétences", "Objectifs", "Modules", "Expert_Level"]:
        fig.add_trace(
            go.Scatterpolar(
                r=wl_df[metric],
                theta=wl_df["Semestre"],
                fill="toself",
                name=metric,
                hovertemplate=f"{metric}: %{{r}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Charge de travail par semestre",
        polar=dict(radialaxis=dict(visible=True)),
        height=650,
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig

def viz3_competency_heatmap(filtered_df):
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée à afficher. Veuillez ajuster vos filtres.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=750,
        )
        return fig
    
    all_comps = set()
    for comps in filtered_df["Competencies_List"]:
        all_comps.update(comps)
    all_comps = sorted(all_comps)[:15]
    modules = filtered_df["EC"].unique()[:25]
    matrix_data = []
    for module in modules:
        row = []
        module_data = filtered_df[filtered_df["EC"] == module]
        module_comps = set()
        for comps in module_data["Competencies_List"]:
            module_comps.update(comps)
        for comp in all_comps:
            row.append(1 if comp in module_comps else 0)
        matrix_data.append(row)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_data,
            x=all_comps,
            y=[m[:35] for m in modules],
            colorscale=[[0, "#f0f0f0"], [1, COLORS["success"]]],
            showscale=False,
            hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Où développer quelles compétences ?",
        xaxis_title="Compétences",
        yaxis_title="Modules",
        height=750,
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"side": "top", "tickangle": -45},
        margin=dict(l=250, r=60, t=120, b=100),
    )
    return fig

def viz4_learning_flow(filtered_df):
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée à afficher. Veuillez ajuster vos filtres.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=650,
        )
        return fig
    
    nodes = []
    node_dict = {}
    links = {"source": [], "target": [], "value": [], "color": []}

    semesters = sorted(filtered_df["Semestre"].unique())
    for sem in semesters:
        node_dict[f"S{sem}"] = len(nodes)
        nodes.append(f"S{sem}")
    ues = filtered_df["UE_Clean"].dropna().unique()
    for ue in ues:
        node_dict[ue] = len(nodes)
        nodes.append(ue)

    for sem in semesters:
        sem_data = filtered_df[filtered_df["Semestre"] == sem]
        ue_counts = sem_data["UE_Clean"].value_counts()
        for ue, count in ue_counts.items():
            if pd.notna(ue):
                links["source"].append(node_dict[f"S{sem}"])
                links["target"].append(node_dict[ue])
                links["value"].append(count)
                links["color"].append("rgba(59, 130, 246, 0.3)")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=[
                        COLORS["primary"] if l.startswith("S") else COLORS["ue"].get(l, "#999")
                        for l in nodes
                    ],
                ),
                link=dict(
                    source=links["source"],
                    target=links["target"],
                    value=links["value"],
                    color=links["color"],
                ),
            )
        ]
    )
    fig.update_layout(
        title="Du semestre aux UE : comment les cours s'enchaînent",
        font=dict(size=12, family="Inter, sans-serif"),
        height=650,
        paper_bgcolor="white",
    )
    return fig

def viz5_critical_competencies(filtered_df):
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée à afficher. Veuillez ajuster vos filtres.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=650,
        )
        return fig
    
    comp_freq = {}
    comp_levels = {}
    for _, row in filtered_df.iterrows():
        for comp in row["Competencies_List"]:
            comp_freq[comp] = comp_freq.get(comp, 0) + 1
            if comp not in comp_levels:
                comp_levels[comp] = []
            comp_levels[comp].append(row["Level"])

    top_comps = sorted(comp_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    fig = go.Figure()
    for comp, freq in top_comps:
        levels = comp_levels[comp]
        level_order = {"N": 1, "A": 2, "M": 3, "E": 4}
        avg_level = np.mean([level_order[l] for l in levels if l in level_order])
        color = get_competency_color(comp)
        fig.add_trace(
            go.Bar(
                x=[freq],
                y=[comp],
                orientation="h",
                marker_color=color,
                text=f"Niveau moyen: {avg_level:.1f}",
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Fréquence: %{x}<br>Niveau moyen: "
                + f"{avg_level:.1f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Compétences les plus utiles pour réussir",
        xaxis_title="Fréquence d'apparition",
        yaxis_title="",
        height=650,
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=180, r=60, t=80, b=60),
    )
    return fig

def create_sunburst_hierarchy(filtered_df):
    hierarchy_data = []
    for _, row in filtered_df.iterrows():
        sem = f"S{row['Semestre']}"
        ue = row["UE_Clean"] if pd.notna(row["UE_Clean"]) else "Autre"
        module = row["EC"][:30]
        for comp in row["Competencies_List"]:
            hierarchy_data.append(
                {
                    "Semestre": sem,
                    "UE": ue,
                    "Module": module,
                    "Compétence": comp,
                    "Value": 1,
                }
            )
    hierarchy_df = pd.DataFrame(hierarchy_data)
    if hierarchy_df.empty:
        return go.Figure()

    color_map = {
        comp: get_competency_color(comp)
        for comp in hierarchy_df["Compétence"].unique()
    }

    fig = px.sunburst(
        hierarchy_df,
        path=["Semestre", "UE", "Module", "Compétence"],
        values="Value",
        color="Compétence",
        color_discrete_map=color_map,
    )
    fig.update_layout(
        title="Vue hiérarchique du curriculum",
        height=850,
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white",
    )
    fig.update_traces(
        textinfo="label",
        hovertemplate="<b>%{label}</b><br>Occurrences: %{value}<extra></extra>",
    )
    return fig

def viz7_statistics_dashboard(filtered_df):
    total_modules = filtered_df["EC"].nunique()
    total_competencies = len(
        set([comp for comps in filtered_df["Competencies_List"] for comp in comps])
    )
    total_objectives = len(filtered_df)

    ue_counts = filtered_df.groupby("UE_Clean").size()
    avg_per_ue = ue_counts.mean() if len(ue_counts) > 0 else 0

    tc_count = sum(
        1
        for comps in filtered_df["Competencies_List"]
        for c in comps
        if str(c).startswith("TC")
    )
    idu_count = sum(
        1
        for comps in filtered_df["Competencies_List"]
        for c in comps
        if str(c).startswith("IDU")
    )

    cards = [
        create_stat_card(
            "Nombre total de cours",
            total_modules,
            "Cours à suivre dans le curriculum.",
            COLORS["primary"],
            comparison=f"En moyenne {avg_per_ue:.0f} cours par UE.",
        ),
        create_stat_card(
            "Compétences différentes",
            total_competencies,
            "Compétences à maîtriser dans le programme.",
            COLORS["success"],
            trend=5,
            comparison="Programme riche en compétences.",
        ),
        create_stat_card(
            "Objectifs d'apprentissage",
            total_objectives,
            "Objectifs évalués dans les différents cours.",
            COLORS["info"],
            comparison=f"{total_objectives/ (total_modules or 1):.1f} objectifs par cours.",
        ),
        create_stat_card(
            "Équilibre TC / IDU",
            f"{tc_count}/{idu_count}",
            "Compétences transversales / disciplinaires.",
            COLORS["warning"],
            comparison="Curriculum équilibré" if abs(tc_count - idu_count) < 50 else "Déséquilibre détecté.",
        ),
    ]
    return cards

# ═══════════════════════════════════════════════════════════════════
#                 GRAPHE RÉSEAU 3D EXPERT
# ═══════════════════════════════════════════════════════════════════

def viz8_network_graph_pro(filtered_df):
    """
    Réseau 3D de compétences - version expert contrastée
    - Fond blanc pour le conteneur
    - Fond noir pour le canvas 3D uniquement
    - Liaisons orange très visibles
    - Nœuds colorés par communautés, taille = degré
    """
    try:
        import networkx as nx
        from networkx.algorithms import community
    except ImportError:
        fig = go.Figure()
        fig.add_annotation(
            text="NetworkX non installé. Installez avec: pip install networkx",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#ef4444", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=850,
        )
        return fig

    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        comps = row["Competencies_List"]
        if not isinstance(comps, list) or len(comps) == 0:
            continue
        for i, c1 in enumerate(comps):
            for c2 in comps[i + 1:]:
                if c1 == c2:
                    continue
                if G.has_edge(c1, c2):
                    G[c1][c2]["weight"] += 1
                else:
                    G.add_edge(c1, c2, weight=1)

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune relation de compétences trouvée avec les filtres actuels.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=850,
        )
        return fig

    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:35]
    G_sub = G.subgraph(top_nodes).copy()

    if len(G_sub.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Le réseau filtré est vide.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=850,
        )
        return fig

    pos = nx.spring_layout(G_sub, dim=3, k=1.6, iterations=60, seed=42)

    try:
        communities_list = community.greedy_modularity_communities(G_sub)
        node_to_community = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_to_community[node] = i
    except Exception:
        node_to_community = {node: 0 for node in G_sub.nodes}
        communities_list = [set(G_sub.nodes())]

    # Edges : orange vif sur fond sombre
    edge_x, edge_y, edge_z = [], [], []
    weights = []
    for u, v, data in G_sub.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
        weights.append(data.get("weight", 1))

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(
            color="rgba(249, 115, 22, 0.85)",
            width=2.5,
        ),
        hoverinfo="none",
        showlegend=False,
    )

    community_palette = [
        "#38bdf8", "#4ade80", "#a855f7",
        "#fbbf24", "#f97316", "#f472b6", "#22c55e",
    ]

    node_traces = []
    degrees = dict(G_sub.degree())
    threshold_label = np.percentile(list(degrees.values()), 55) if len(degrees) > 0 else 1

    for comm_id in sorted(set(node_to_community.values())):
        nodes_in_comm = [n for n in G_sub.nodes() if node_to_community[n] == comm_id]
        if not nodes_in_comm:
            continue

        node_x, node_y, node_z = [], [], []
        node_sizes, node_labels, hover_texts = [], [], []

        for n in nodes_in_comm:
            x, y, z = pos[n]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            deg = degrees[n]
            node_sizes.append(7 + 2.8 * deg)

            short_label = n.split("-")[0] if "-" in n else n
            node_labels.append(short_label if deg >= threshold_label else "")

            hover_texts.append(f"{n}<br>Connexions : {deg}")

        node_traces.append(
            go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=community_palette[comm_id % len(community_palette)],
                    opacity=0.9,
                    line=dict(color="#020617", width=1.2),
                ),
                text=node_labels,
                textposition="top center",
                textfont=dict(size=9, color="#f9fafb"),
                hovertext=hover_texts,
                hoverinfo="text",
                name=f"Communauté {comm_id + 1}",
                showlegend=True,
            )
        )

    density = nx.density(G_sub)
    try:
        modularity_val = community.modularity(G_sub, communities_list)
    except Exception:
        modularity_val = None

    stats_text = (
        f"<b>Réseau de compétences</b><br>"
        f"Nœuds : {len(G_sub.nodes())}<br>"
        f"Liens : {len(G_sub.edges())}<br>"
        f"Densité : {density:.2f}"
    )
    if modularity_val is not None:
        stats_text += f"<br>Modularité : {modularity_val:.2f}"

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(
            text="Réseau 3D de compétences<br><sub>Compétences fortement liées entre les cours</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, sans-serif", color="#1e293b"),
        ),
        showlegend=True,
        legend=dict(
            x=1.02, y=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(color="#1e293b", size=10),
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="#020617",
            camera=dict(
                eye=dict(x=1.7, y=1.8, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=850,
        margin=dict(l=0, r=0, t=80, b=0),
        font=dict(size=11, family="Inter, sans-serif", color="#1e293b"),
        annotations=[
            dict(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                font=dict(size=10, color="#1e293b"),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                borderpad=6,
            )
        ],
    )
    return fig

# ═══════════════════════════════════════════════════════════════════
#                         LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

app.layout = html.Div(
    [
        dcc.Store(id="filtered-data-store"),
        dcc.Store(id="drill-down-level", data="global"),
        dcc.Store(id="selected-semester", data=None),
        dcc.Store(id="selected-module", data=None),

        # HEADER
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("FE", className="header-logo-icon"),
                                html.Div(
                                    [
                                        html.H1(
                                            "ForestED - Tableau de bord étudiant",
                                            className="header-title",
                                        ),
                                        html.P(
                                            "Visualisez où, quand et comment développer vos compétences",
                                            className="header-subtitle",
                                        ),
                                    ]
                                ),
                            ],
                            className="header-logo",
                        ),
                        html.Div(
                            f"Mise à jour : {datetime.now().strftime('%d/%m/%Y')}",
                            className="header-actions",
                        ),
                    ],
                    className="header-content",
                )
            ],
            className="app-header",
        ),

        # MAIN LAYOUT avec sidebar fixe
        html.Div(
            [
                # Sidebar fixe à gauche
                html.Div(
                    create_filter_section(),
                    className="app-sidebar",
                ),

                # Contenu principal
                html.Div(
                    [
                        # KPIs
                        html.Div(
                            [
                                create_section_header(
                                    "Vue d'ensemble",
                                    "Indicateurs clés de votre curriculum",
                                ),
                                html.Div(id="stats-dashboard", className="stats-grid"),
                            ],
                            className="section",
                        ),

                        # Mon parcours
                        html.Div(
                            [
                                create_section_header(
                                    "Mon parcours",
                                    "Exploration interactive par niveaux",
                                ),
                                html.Div(
                                    id="breadcrumb-nav",
                                    className="breadcrumb",
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "🏠 Vue globale",
                                            id="btn-global",
                                            n_clicks=0,
                                            className="btn btn-primary",
                                        ),
                                        html.Button(
                                            "← Retour",
                                            id="btn-back",
                                            n_clicks=0,
                                            className="btn btn-outline",
                                        ),
                                    ],
                                    className="drill-controls",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="viz1-timeline-drilldown",
                                            config={"displayModeBar": True, "displaylogo": False},
                                        )
                                    ],
                                    className="card viz-card",
                                ),
                            ],
                            className="section",
                        ),

                        # Analyses principales
                        html.Div(
                            [
                                create_section_header(
                                    "Analyses principales",
                                    "Charge de travail et compétences clés",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz2-radar",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz5-critical",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                    ],
                                    className="viz-grid-2",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz3-heatmap",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz4-flow",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                    ],
                                    className="viz-grid-2",
                                ),
                            ],
                            className="section",
                        ),

                        # Relations et hiérarchies
                        html.Div(
                            [
                                create_section_header(
                                    "Relations et hiérarchies",
                                    "Organisation globale et liens entre compétences",
                                ),
                                # Grid 2 colonnes professionnel
                                html.Div(
                                    [
                                        # SUNBURST à gauche
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div("Vue hiérarchique", className="viz-header-title"),
                                                                html.Div("Curriculum complet", className="viz-header-subtitle"),
                                                            ],
                                                            className="viz-header",
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Graph(
                                                                    id="viz6-sunburst",
                                                                    config={
                                                                        "displayModeBar": True,
                                                                        "displaylogo": False,
                                                                    },
                                                                )
                                                            ],
                                                            className="viz-content",
                                                        ),
                                                    ],
                                                    className="card viz-card-pro",
                                                )
                                            ],
                                            className="viz-col",
                                        ),
                                        
                                        # RÉSEAU 3D à droite
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div("Réseau 3D de compétences", className="viz-header-title"),
                                                                html.Div("Interconnexions", className="viz-header-subtitle"),
                                                            ],
                                                            className="viz-header",
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Graph(
                                                                    id="viz8-network",
                                                                    config={
                                                                        "displayModeBar": True,
                                                                        "displaylogo": False,
                                                                        "modeBarButtonsToAdd": ['pan3d', 'zoom3d', 'orbitRotation', 'tableRotation'],
                                                                    },
                                                                )
                                                            ],
                                                            className="viz-content",
                                                        ),
                                                    ],
                                                    className="card viz-card-pro",
                                                )
                                            ],
                                            className="viz-col",
                                        ),
                                    ],
                                    className="viz-grid-professional",
                                ),
                            ],
                            className="section",
                        ),
                    ],
                    className="app-main-content",
                ),
            ],
            className="app-container",
        ),

        # Footer
        html.Div(
            [
                html.P(
                    "Utilisez les filtres pour adapter le dashboard à votre parcours",
                    className="footer-text",
                ),
                html.P(
                    "ForestED - Vue Étudiant Expert, 2025",
                    className="footer-credit",
                ),
            ],
            className="app-footer",
        ),
    ],
    className="app-wrapper",
)

# ═══════════════════════════════════════════════════════════════════
#                          CALLBACKS
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    [Output("filtered-data-store", "data"), Output("filter-status", "children")],
    [
        Input("semester-filter", "value"),
        Input("ue-filter", "value"),
        Input("level-filter", "value"),
        Input("reset-filters", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def filter_data(semesters, ues, levels, reset_clicks):
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-filters.n_clicks":
        filtered_df = df.copy()
        status = f"{len(df)} objectifs affichés"
    else:
        filtered_df = df.copy()
        if semesters:
            filtered_df = filtered_df[filtered_df["Semestre"].isin(semesters)]
        if ues:
            filtered_df = filtered_df[filtered_df["UE_Clean"].isin(ues)]
        if levels:
            filtered_df = filtered_df[filtered_df["Level"].isin(levels)]
        status = f"{len(filtered_df)} / {len(df)} objectifs"
    return filtered_df.to_json(date_format="iso", orient="split"), status

@app.callback(
    [
        Output("viz1-timeline-drilldown", "figure"),
        Output("drill-down-level", "data"),
        Output("selected-semester", "data"),
        Output("selected-module", "data"),
        Output("breadcrumb-nav", "children"),
    ],
    [
        Input("filtered-data-store", "data"),
        Input("viz1-timeline-drilldown", "clickData"),
        Input("btn-global", "n_clicks"),
        Input("btn-back", "n_clicks"),
    ],
    [
        State("drill-down-level", "data"),
        State("selected-semester", "data"),
        State("selected-module", "data"),
    ],
    prevent_initial_call=False,
)
def update_drilldown_viz1(
    filtered_data_json,
    clickData,
    btn_global_clicks,
    btn_back_clicks,
    current_level,
    current_semester,
    current_module,
):
    ctx = dash.callback_context
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)

    filtered_df = filtered_df.assign(All_Competencies=filtered_df["Competencies_List"])

    trigger_id = (
        ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "none"
    )

    new_level = current_level if current_level else "global"
    new_semester = current_semester
    new_module = current_module

    if trigger_id == "btn-global":
        new_level = "global"
        new_semester = None
        new_module = None
    elif trigger_id == "btn-back":
        if current_level == "module":
            new_level = "semester"
            new_module = None
        elif current_level == "semester":
            new_level = "global"
            new_semester = None
    elif trigger_id == "viz1-timeline-drilldown" and clickData and "points" in clickData:
        point = clickData["points"][0]
        if current_level in [None, "global"]:
            clicked_semester = point.get("y")
            if clicked_semester is not None:
                try:
                    if isinstance(clicked_semester, str):
                        clicked_semester = int(clicked_semester)
                except (ValueError, TypeError):
                    clicked_semester = None
                if clicked_semester is not None:
                    new_level = "semester"
                    new_semester = clicked_semester
                    new_module = None
        elif current_level == "semester":
            clicked_module = point.get("y")
            if clicked_module is not None:
                new_level = "module"
                new_module = clicked_module

    try:
        df_pivot_global, df_pivot_semester, df_pivot_module = compute_competency_counts_per_module(
            filtered_df
        )
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erreur: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig, "global", None, None, html.Div("Erreur")

    if new_level == "global":
        fig = create_heatmap_for_global(df_pivot_global)
        breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    elif new_level == "semester" and new_semester is not None:
        available_semesters = (
            df_pivot_semester.index.get_level_values("Semestre").unique().tolist()
        )
        if new_semester in available_semesters:
            fig = create_heatmap_for_semester(new_semester, df_pivot_semester, True)
            breadcrumb = html.Div([
                html.Span("Vue globale", className="breadcrumb-item"),
                html.Span(" / ", className="breadcrumb-separator"),
                html.Span(f"Semestre {new_semester}", className="breadcrumb-item active"),
            ])
        else:
            fig = create_heatmap_for_global(df_pivot_global)
            new_level = "global"
            new_semester = None
            breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    elif new_level == "module" and new_module is not None:
        available_modules = (
            df_pivot_module.index.get_level_values("EC").unique().tolist()
        )
        if new_module in available_modules:
            fig = create_heatmap_for_module(new_module, df_pivot_module)
            breadcrumb = html.Div([
                html.Span("Vue globale", className="breadcrumb-item"),
                html.Span(" / ", className="breadcrumb-separator"),
                html.Span(f"Semestre {new_semester}", className="breadcrumb-item"),
                html.Span(" / ", className="breadcrumb-separator"),
                html.Span(new_module[:40], className="breadcrumb-item active"),
            ])
        else:
            fig = create_heatmap_for_global(df_pivot_global)
            new_level = "global"
            new_semester = None
            new_module = None
            breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    else:
        fig = create_heatmap_for_global(df_pivot_global)
        new_level = "global"
        new_semester = None
        new_module = None
        breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")

    return fig, new_level, new_semester, new_module, breadcrumb

@app.callback(
    Output("stats-dashboard", "children"),
    Input("filtered-data-store", "data"),
)
def update_stats(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return viz7_statistics_dashboard(filtered_df)

@app.callback(
    Output("viz2-radar", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz2(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return viz2_workload_radar(filtered_df)

@app.callback(
    Output("viz3-heatmap", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz3(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return viz3_competency_heatmap(filtered_df)

@app.callback(
    Output("viz4-flow", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz4(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return viz4_learning_flow(filtered_df)

@app.callback(
    Output("viz5-critical", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz5(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return viz5_critical_competencies(filtered_df)

@app.callback(
    Output("viz6-sunburst", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz6(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return create_sunburst_hierarchy(filtered_df)

@app.callback(
    Output("viz8-network", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz8(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
    filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    return viz8_network_graph_pro(filtered_df)

# ═══════════════════════════════════════════════════════════════════
#                          LANCEMENT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)