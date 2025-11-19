import plotly.graph_objects as go
from ischemist.plotly import Styler

from retrocast.models.stats import ModelComparison, ModelStatistics, RankResult, StratifiedMetric


def _create_metric_trace(metric: StratifiedMetric, name: str, color: str) -> go.Bar:
    """
    Helper to create a single bar trace with error bars.
    """
    # Extract sorted data
    # Sort by depth (assuming keys are ints or sortable)
    sorted_keys = sorted(metric.by_group.keys())

    x_vals = [f"Length {k}" for k in sorted_keys]
    y_vals = [metric.by_group[k].value * 100 for k in sorted_keys]  # Convert to %

    # Error bars
    # Plotly expects the length of the error bar, not the absolute value
    ci_upper = [metric.by_group[k].ci_upper * 100 for k in sorted_keys]
    ci_lower = [metric.by_group[k].ci_lower * 100 for k in sorted_keys]

    error_plus = [upp - y for upp, y in zip(ci_upper, y_vals, strict=True)]
    error_minus = [y - low for y, low in zip(y_vals, ci_lower, strict=True)]

    custom_data = []
    for low, upp, n, rel in zip(
        ci_lower,
        ci_upper,
        [metric.by_group[k].n_samples for k in sorted_keys],
        [metric.by_group[k].reliability.code for k in sorted_keys],
        strict=True,
    ):
        custom_data.append([n, low, upp, rel])

    return go.Bar(
        name=name,
        x=x_vals,
        y=y_vals,
        marker_color=color,
        error_y=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus, visible=True),
        customdata=custom_data,
        hovertemplate=(
            f"<b>{name}</b>: %{{y:.1f}}%<br>"
            + "N=%{customdata[0]}<br>"
            + "CI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>"
            + "Status: %{customdata[3]}"
            + "<extra></extra>"
        ),
    )


def plot_single_model_diagnostics(stats: ModelStatistics) -> go.Figure:
    """
    Creates a grouped bar chart showing performance metrics stratified by depth.
    Includes Solvability, Top-1, Top-5, Top-10.
    """
    fig = go.Figure()

    # 1. Solvability (The Baseline)
    fig.add_trace(
        _create_metric_trace(
            stats.solvability,
            name="Solvability",
            color="#b892ff",  # Strong Blue
        )
    )

    # 2. Top-K Accuracies
    # We choose a few key Ks to avoid clutter
    k_colors = {
        1: "#ffc2e2",  # Orange
        5: "#ff90b3",  # Green
        10: "#ef7a85",  # Purple
    }

    for k in [1, 5, 10]:
        if k in stats.top_k_accuracy:
            fig.add_trace(
                _create_metric_trace(stats.top_k_accuracy[k], name=f"Top-{k}", color=k_colors.get(k, "#95A5A6"))
            )

    # 3. Layout Polish
    full_title = (
        f"<b>Performance Diagnostics: {stats.model_name}</b><br>"
        f"<span style='font-size: 12px;'>Benchmark: {stats.benchmark} | Stock: {stats.stock}</span>"
    )
    fig.update_layout(
        title=full_title,
        yaxis=dict(range=[0, 100]),
        xaxis=dict(title="Route Length"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1000,
        height=500,
    )
    Styler().apply_style(fig)

    return fig


def plot_multi_model_comparison(
    models_stats: list[ModelStatistics], metric_type: str = "Top-1", k: int = 1
) -> go.Figure:
    fig = go.Figure()

    # 1. Gather all depths
    all_depths = set()
    for m in models_stats:
        if metric_type == "Solvability":
            all_depths.update(m.solvability.by_group.keys())
        else:
            if k in m.top_k_accuracy:
                all_depths.update(m.top_k_accuracy[k].by_group.keys())

    sorted_depths = sorted(list(all_depths))  # e.g. [2, 3, 4, 5, 6]

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # 2. Calculate Offsets
    # We want to center the cluster around the integer Depth.
    # e.g. Depth 2 with 3 models: 1.9, 2.0, 2.1
    num_models = len(models_stats)
    width_per_cluster = 0.6  # How much space the cluster takes (out of 1.0)
    offset_step = width_per_cluster / max(1, (num_models - 1))
    start_offset = -width_per_cluster / 2

    # If only 1 model, offset is 0.
    if num_models == 1:
        offsets = [0]
    else:
        offsets = [start_offset + (i * offset_step) for i in range(num_models)]

    for i, model in enumerate(models_stats):
        if metric_type == "Solvability":
            metric_obj = model.solvability
            label = "Solvability"
        else:
            metric_obj = model.top_k_accuracy.get(k)
            label = f"Top-{k}"
            if not metric_obj:
                continue

        x_vals = []  # These will be floats now!
        y_vals = []
        y_upper = []
        y_lower = []
        hover_texts = []

        for depth in sorted_depths:
            if depth in metric_obj.by_group:
                res = metric_obj.by_group[depth]

                # Apply Jitter
                # X = Integer Depth + Model Offset
                x_pos = int(depth) + offsets[i]

                x_vals.append(x_pos)
                y_vals.append(res.value * 100)
                y_upper.append((res.ci_upper - res.value) * 100)
                y_lower.append((res.value - res.ci_lower) * 100)

                hover_texts.append(
                    f"<b>{model.model_name}</b><br>"
                    f"Depth {depth}<br>"
                    f"{label}: {res.value:.1%}<br>"
                    f"CI: [{res.ci_lower:.1%}, {res.ci_upper:.1%}]<br>"
                    f"N={res.n_samples}"
                )

        fig.add_trace(
            go.Scatter(
                name=model.model_name,
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(color=colors[i % len(colors)], size=10, symbol="circle"),
                error_y=dict(
                    type="data",
                    array=y_upper,
                    arrayminus=y_lower,
                    visible=True,
                    width=4,  # Slightly wider whiskers
                    thickness=1.5,
                ),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    # 3. Polish Layout
    fig.update_layout(
        title=f"Model Comparison: {metric_type} (k={k} if applicable)",
        yaxis=dict(title="Percentage (%)", range=[0, 100], gridcolor="#ecf0f1"),
        xaxis=dict(
            title="Route Difficulty (Depth)",
            gridcolor="#ecf0f1",
            # Force X-axis to show integer ticks for Depths
            tickmode="array",
            tickvals=sorted_depths,
            ticktext=[f"Depth {d}" for d in sorted_depths],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    Styler().apply_style(fig)

    return fig


def plot_overall_comparison(models_stats: list[ModelStatistics]) -> go.Figure:
    """
    Creates a summary plot comparing Overall performance across key metrics.
    X-Axis: Metric (Solvability, Top-1, Top-5, Top-10)
    Y-Axis: Percentage
    Grouped by Model.
    """
    fig = go.Figure()

    # Define the metrics we want to show in order
    metrics_config = [
        {"key": "solvability", "label": "Solvability"},
        {"key": "top-1", "label": "Top-1"},
        {"key": "top-5", "label": "Top-5"},
        {"key": "top-10", "label": "Top-10"},
    ]

    # Colors
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Jitter Setup
    # We treat the metrics as integer positions 0, 1, 2, 3
    num_models = len(models_stats)
    width_per_cluster = 0.6
    offset_step = width_per_cluster / max(1, (num_models - 1))
    start_offset = -width_per_cluster / 2

    if num_models == 1:
        offsets = [0]
    else:
        offsets = [start_offset + (i * offset_step) for i in range(num_models)]

    for i, model in enumerate(models_stats):
        x_vals = []
        y_vals = []
        y_upper = []
        y_lower = []
        hover_texts = []

        for m_idx, config in enumerate(metrics_config):
            key = config["key"]
            label = config["label"]

            # Fetch the data object
            res = None
            if key == "solvability":
                res = model.solvability.overall
            elif key.startswith("top-"):
                k = int(key.split("-")[1])
                if k in model.top_k_accuracy:
                    res = model.top_k_accuracy[k].overall

            if res:
                # X = Integer Metric Position + Model Offset
                x_pos = m_idx + offsets[i]

                x_vals.append(x_pos)
                y_vals.append(res.value * 100)
                y_upper.append((res.ci_upper - res.value) * 100)
                y_lower.append((res.value - res.ci_lower) * 100)

                # Reliability Icon
                flag = ""
                if res.reliability.code != "OK":
                    flag = f" (⚠️ {res.reliability.code})"

                hover_texts.append(
                    f"<b>{model.model_name}</b><br>"
                    f"{label}<br>"
                    f"Value: {res.value:.1%}<br>"
                    f"CI: [{res.ci_lower:.1%}, {res.ci_upper:.1%}]<br>"
                    f"N={res.n_samples}{flag}"
                )

        fig.add_trace(
            go.Scatter(
                name=model.model_name,
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(color=colors[i % len(colors)], size=12, symbol="circle"),
                error_y=dict(type="data", array=y_upper, arrayminus=y_lower, visible=True, width=4, thickness=1.5),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    # Layout
    fig.update_layout(
        title="Overall Performance Summary",
        yaxis=dict(title="Percentage (%)", range=[0, 100], gridcolor="#ecf0f1"),
        xaxis=dict(
            title="Metric",
            gridcolor="#ecf0f1",
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=[m["label"] for m in metrics_config],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    Styler().apply_style(fig)

    return fig


def plot_probabilistic_ranking(ranking: list[RankResult], metric_name: str) -> go.Figure:
    """
    Creates a Heatmap of Model vs Rank Probability.
    """
    # Prepare data for Heatmap
    # Y-axis: Model Names (sorted by expected rank, which `ranking` already is)
    y_labels = [r.model_name for r in ranking]

    # X-axis: Ranks (1, 2, 3...)
    n_models = len(ranking)
    x_labels = [f"Rank {i}" for i in range(1, n_models + 1)]

    # Z-matrix: shape (n_models, n_ranks)
    z_values = []
    annotation_text = []

    for r in ranking:
        row_probs = []
        row_text = []
        for rank in range(1, n_models + 1):
            prob = r.rank_probs.get(rank, 0.0)
            row_probs.append(prob)
            # Only show text if > 1% to keep it clean
            txt = f"{prob:.0%}" if prob > 0.01 else ""
            row_text.append(txt)
        z_values.append(row_probs)
        annotation_text.append(row_text)

    # Because Plotly Heatmap draws bottom-to-top by default for Y-axis?
    # No, standard matrix convention puts index 0 at top usually, but let's check.
    # Actually, let's just reverse the lists so the #1 model is at the TOP visually.
    y_labels = y_labels[::-1]
    z_values = z_values[::-1]
    annotation_text = annotation_text[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=annotation_text,
            texttemplate="%{text}",  # Show the percentage text
            colorscale="Blues",
            zmin=0,
            zmax=1,
            xgap=1,
            ygap=1,  # Grid lines
        )
    )

    fig.update_layout(
        title=f"Probabilistic Ranking: {metric_name}",
        xaxis_title="Rank",
        yaxis_title="Model",
        # Remove colorbar if the text is sufficient
        # coloraxis_showscale=False
    )
    Styler().apply_style(fig)

    return fig


def plot_pairwise_matrix(comparisons: list[ModelComparison], metric_name: str) -> go.Figure:
    """
    Plots a Win/Loss matrix.
    Row Model vs Column Model.
    """
    # 1. Extract unique models
    models = sorted(list(set([c.model_a for c in comparisons])))
    model_map = {m: i for i, m in enumerate(models)}
    n = len(models)

    # 2. Initialize Matrix
    z_values = [[None] * n for _ in range(n)]
    text_values = [[""] * n for _ in range(n)]

    # Track max value for symmetric color scaling
    max_diff = 0.0

    for c in comparisons:
        row = model_map[c.model_a]
        col = model_map[c.model_b]

        # Value is the mean difference
        diff = c.diff_mean
        z_values[row][col] = diff
        max_diff = max(max_diff, abs(diff))

        # Text annotation
        # We add a star if the CI excludes zero (Significant)
        sig_mark = "★" if c.is_significant else ""
        text_values[row][col] = f"{diff:+.1%}{sig_mark}"

    # 3. Plot
    # We reverse the Y-axis list so the matrix reads Top-to-Bottom, Left-to-Right
    # (i.e. Model A at top row)
    models_y = models[::-1]
    z_values = z_values[::-1]
    text_values = text_values[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=models,
            y=models_y,
            text=text_values,
            texttemplate="%{text}",
            # RdBu: Red (negative/loss) to Blue (positive/win).
            # Or RdYlGn (Red-Yellow-Green). Let's use RdBu for scientific look.
            colorscale="RdBu",
            zmid=0,
            zmin=-max_diff,
            zmax=max_diff,
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title=f"Pairwise Comparison Matrix: {metric_name}",
        xaxis_title="Opponent",
        yaxis_title="Model (Row - Col)",
        height=600 + (n * 20),  # Auto-scale height
        width=700 + (n * 20),
    )
    Styler().apply_style(fig)

    return fig
