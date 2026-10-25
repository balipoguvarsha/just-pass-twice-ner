"""
Streamlit Web Application: Restrained Academic Research Interface
for Just Pass Twice (JPT) Named Entity Recognition.
"""

import os
import sys
import json
import time
import pandas as pd
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import JPTPredictor

# -----------------------------------------------------------------------------
# Page Configuration & Academic Research Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Just Pass Twice NER Prototype",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Academic Aesthetic CSS:
# Warm white/light gray background (#fbfbfb), charcoal text (#22252a),
# subtle borders (#e2e4e8), muted forest green accents (#2d6a4f).
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #fafafa;
        color: #212529;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Academic Header */
    .research-header {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 14px;
        margin-bottom: 22px;
    }
    .research-title {
        font-size: 1.55rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .research-subtitle {
        font-size: 0.92rem;
        color: #4b5563;
        line-height: 1.4;
    }

    /* Section Cards with Subtle Borders */
    .academic-panel {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 18px;
    }
    .panel-heading {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 10px;
        border-bottom: 1px solid #f3f4f6;
        padding-bottom: 6px;
    }

    /* Entity Badges */
    .entity-highlight {
        display: inline-block;
        padding: 2px 7px;
        margin: 2px 3px;
        border-radius: 3px;
        font-size: 0.88rem;
        font-weight: 500;
        border: 1px solid #cbd5e1;
    }
    .entity-PER {
        background-color: #ecfdf5;
        border-color: #a7f3d0;
        color: #065f46;
    }
    .entity-ORG {
        background-color: #f0fdfa;
        border-color: #99f6e4;
        color: #115e59;
    }
    .entity-LOC {
        background-color: #f8fafc;
        border-color: #cbd5e1;
        color: #334155;
    }
    .entity-MISC {
        background-color: #fefce8;
        border-color: #fde047;
        color: #854d0e;
    }
    .entity-tag {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        opacity: 0.85;
        margin-left: 4px;
    }

    /* Status Indicator */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-trained {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-demo {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }

    /* Latency Metric Box */
    .metric-badge {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.86rem;
        color: #1f2937;
        background-color: #f3f4f6;
        padding: 4px 8px;
        border-radius: 3px;
        border: 1px solid #e5e7eb;
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background-color: #2d6a4f;
        color: #ffffff;
        border: 1px solid #1b4332;
        border-radius: 4px;
        padding: 7px 18px;
        font-weight: 500;
        font-size: 0.92rem;
        transition: background-color 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #1b4332;
        color: #ffffff;
        border-color: #081c15;
    }

    /* Academic Table */
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.88rem;
    }
    th {
        background-color: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        color: #475569;
        font-weight: 600;
        padding: 8px 10px;
        text-align: left;
    }
    td {
        border-bottom: 1px solid #f1f5f9;
        padding: 8px 10px;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Predictor Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor(model_choice: str):
    checkpoint_file = os.path.join(PROJECT_ROOT, "checkpoints", f"{model_choice}_model.pt")
    weights_path = os.path.join(PROJECT_ROOT, "distilgpt2_model.safetensors")
    return JPTPredictor(
        checkpoint_path=checkpoint_file,
        model_type=model_choice,
        weights_path=weights_path if os.path.exists(weights_path) else None,
        device="cpu"
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="research-header">
    <div class="research-title">Just Pass Twice: Efficient Named Entity Recognition with Small Language Models</div>
    <div class="research-subtitle">
        Academic research prototype based on <em>"Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER"</em> (ACL 2026 / arXiv:2604.05158).
        Enables causal SLMs to perform bidirectional token classification via input concatenation and definition-guided bilinear matching.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Layout: Left column (Interactive Inference), Right column (Benchmark Evaluation)
# -----------------------------------------------------------------------------
col_main, col_bench = st.columns([1.1, 0.9], gap="large")

with col_main:
    st.markdown('<div class="panel-heading">Interactive Model Inference</div>', unsafe_allow_html=True)

    # Model selector
    model_options = {
        "jpt": "Paper-Inspired JPT (Two-Pass + Canonical Definition)",
        "multi_def_jpt": "Multi-Def JPT (Proposed Improvement: Paraphrase Aggregation)",
        "baseline": "Conventional Baseline (Single-Pass Causal Token Classifier)"
    }
    selected_model_key = st.selectbox(
        "Select Model Architecture:",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k]
    )

    # Sample inputs
    sample_texts = {
        "CoNLL News": "EU rejects German call to boycott British lamb after Brussels meetings.",
        "Science & Nobel": "Marie Curie discovered radium and polonium at the University of Paris.",
        "Computer Science": "Alan Turing developed the Enigma cryptanalysis machine at Bletchley Park in the United Kingdom.",
        "Politics & Diplomacy": "Barack Obama and Angela Merkel addressed the United Nations General Assembly in New York.",
        "CrossNER AI": "Geoffrey Hinton and Yann LeCun presented a deep neural network algorithm at the NeurIPS conference.",
        "CrossNER Politics": "The Labour Party politician called for a national election referendum in the United Kingdom.",
        "CrossNER Music": "Freddie Mercury composed Bohemian Rhapsody for Queen in London."
    }
    selected_sample = st.selectbox("Load Benchmark Sample (optional):", options=list(sample_texts.keys()))

    input_text = st.text_area(
        "Input Sentence / Paragraph:",
        value=sample_texts[selected_sample],
        height=100
    )

    predict_btn = st.button("Run Entity Extraction", use_container_width=False)

    if predict_btn or input_text:
        predictor = get_predictor(selected_model_key)
        
        with st.spinner("Executing token classification..."):
            result = predictor.predict(input_text)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # Status and Latency Metadata
        stat_class = "status-trained" if result["is_trained"] else "status-demo"
        stat_text = "Trained Checkpoint" if result["is_trained"] else "Demonstration Mode (Pretrained Weights)"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span class="status-badge {stat_class}">{stat_text}</span>
            <span class="metric-badge">Inference Latency: <strong>{result['latency_ms']} ms</strong></span>
        </div>
        """, unsafe_allow_html=True)

        # Recognized Spans
        if result["entities"]:
            st.markdown("<strong>Extracted Entity Mentions:</strong>", unsafe_allow_html=True)
            
            # Inline highlighted visualizer
            words = result["tokens"]
            tags = result["bio_tags"]
            highlighted_html = []
            
            for word, tag in zip(words, tags):
                if tag.startswith("B-") or tag.startswith("I-"):
                    ent_type = tag.split("-", 1)[1]
                    css_cls = f"entity-{ent_type}" if ent_type in ("PER", "ORG", "LOC", "MISC") else "entity-LOC"
                    highlighted_html.append(f'<span class="entity-highlight {css_cls}">{word} <span class="entity-tag">{ent_type}</span></span>')
                else:
                    highlighted_html.append(f'<span style="padding: 2px 2px;">{word}</span>')
                    
            st.markdown(f"<div class='academic-panel' style='line-height: 2.1;'>{' '.join(highlighted_html)}</div>", unsafe_allow_html=True)

            # Compact table breakdown
            df_entities = pd.DataFrame([
                {
                    "Mention Text": e["entity_text"],
                    "Entity Type": e["entity_type"],
                    "Token Span": f"[{e['token_start']}:{e['token_end']}]",
                    "Confidence": f"{e['confidence'] * 100:.1f}%"
                }
                for e in result["entities"]
            ])
            st.dataframe(df_entities, hide_index=True, use_container_width=True)
        else:
            st.info("No named entities detected for this input.")


with col_bench:
    st.markdown('<div class="panel-heading">Empirical Results & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.caption("Measured entity-level Precision, Recall, Micro-F1, Latency, and Throughput under identical test splits.")

    results_file = os.path.join(PROJECT_ROOT, "outputs", "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            bench_data = json.load(f)

        rows = []
        for m_key, m_info in bench_data.items():
            name = {
                "baseline": "Baseline Token Classifier",
                "jpt": "Paper-Inspired JPT",
                "multi_def_jpt": "Multi-Def JPT (Proposed)"
            }.get(m_key, m_key)
            
            rows.append({
                "Architecture": name,
                "Precision": f"{m_info.get('test_precision', 0.0):.1f}%",
                "Recall": f"{m_info.get('test_recall', 0.0):.1f}%",
                "Micro F1": f"{m_info.get('test_f1', 0.0):.1f}%",
                "Latency": f"{m_info.get('speed', {}).get('mean_latency_ms', 0.0):.1f} ms",
                "Throughput": f"{m_info.get('speed', {}).get('throughput_tokens_per_sec', 0.0):.0f} tok/s"
            })
        
        df_bench = pd.DataFrame(rows)
        st.dataframe(df_bench, hide_index=True, use_container_width=True)
    else:
        st.info("Benchmark results file (outputs/results.json) will appear once training completes.")

    # Cross-Domain Zero-Shot Benchmark
    crossner_file = os.path.join(PROJECT_ROOT, "outputs", "crossner_results.json")
    if os.path.exists(crossner_file):
        st.markdown('<div class="panel-heading" style="margin-top: 20px;">Zero-Shot Cross-Domain Transfer (CrossNER)</div>', unsafe_allow_html=True)
        st.caption("Evaluates target domain adaptation by dynamically injecting entity type definitions at test time without fine-tuning.")
        with open(crossner_file, "r") as f:
            cross_data = json.load(f)
        cross_rows = []
        for dom, dom_info in cross_data.items():
            m = dom_info.get("models", {})
            cross_rows.append({
                "Target Domain": dom.upper(),
                "Baseline F1": f"{m.get('baseline', {}).get('f1', 0.0):.2f}%",
                "JPT (Paper) F1": f"{m.get('jpt', {}).get('f1', 0.0):.2f}%",
                "Multi-Def JPT F1": f"{m.get('multi_def_jpt', {}).get('f1', 0.0):.2f}%"
            })
        df_cross = pd.DataFrame(cross_rows)
        st.dataframe(df_cross, hide_index=True, use_container_width=True)

    # Methodology & Core Hypotheses
    st.markdown('<div class="panel-heading" style="margin-top: 20px;">Methodology & Architectural Principles</div>', unsafe_allow_html=True)
    with st.expander("1. Why Causal LLMs Fail Standard Token Classification", expanded=False):
        st.write(r"""
        Standard causal language models (GPT, Qwen, LLaMA) use lower-triangular causal attention where token $x_i$ can only attend to preceding tokens $x_1, \dots, x_i$. 
        In sequence labeling (NER), future context is often necessary to disambiguate tokens (e.g., distinguishing 'Paris' [Person] vs. 'Paris' [Location] based on words appearing later in the sentence).
        """)

    with st.expander("2. The Just Pass Twice (JPT) Mechanism", expanded=False):
        st.write(r"""
        JPT duplicates the input sequence: $\mathbf{x}' = [x_1, \dots, x_n, \text{SEP}, x_1, \dots, x_n]$.
        Under causal attention, tokens in the second pass attend to all tokens from the first pass (full sequence), providing complete bidirectional context in a single forward prefill pass without modifying model attention layers.
        """)

    with st.expander("3. Definition-Guided Bilinear Classification", expanded=False):
        st.write(r"""
        Entity types are represented by natural language definitions rather than static label IDs. 
        Token hidden representations $\mathbf{t}_i$ and definition embeddings $\mathbf{p}_j$ are projected into a shared semantic space $\mathbb{R}^{d_p}$, and scored via a learned bilinear interaction:
        $S_{i, j} = \mathbf{t}_i^T \mathbf{W} \mathbf{p}_j + b_j$.
        """)

    with st.expander("4. Proposed Improvement: Multi-Definition Aggregation", expanded=False):
        st.write(r"""
        Hypothesis: *'Using multiple paraphrased definitions for each entity type improves entity representation robustness.'*
        We aggregate multiple paraphrased definitions per entity type via mean pooling:
        $\mathbf{p}_j = \frac{1}{M} \sum_{m=1}^M \text{MLP}_{ent}(\mathbf{e}_{j, m})$, reducing sensitivity to specific lexical phrasings.
        """)

    with st.expander("5. Known Technical Limitations", expanded=False):
        st.write(r"""
        - **Flat NER Only:** Labels one tag per token; overlapping or nested entities are not natively represented.
        - **Doubled Sequence Length:** Input duplication increases quadratic self-attention complexity from $O(N^2)$ to $O((2N)^2)$, which is mitigated by segment chunking.
        - **Local CPU Constraints:** Fine-tuned on an 82M DistilGPT-2 backbone to ensure reproducible execution on CPU without GPU memory errors.
        """)
