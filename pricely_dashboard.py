import logging
import queue
import threading
import time
import gradio as gr
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity
from log_utils import reformat
import plotly.graph_objects as go

# Logging
class QueueHandler(logging.Handler):
    """Logging handler that sends log records to a queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        self.log_queue.put(self.format(record))

def html_for(log_data: list[str]) -> str:
    """Return styled HTML for logs."""
    formatted = []
    for line in log_data[-18:]:
        if line.startswith("["):
            parts = line.split("]", 1)
            formatted.append(
                f'<span style="color:#888;">{parts[0]}]</span>'
                f'<span style="color:#000;">{parts[1] if len(parts) > 1 else ""}</span>'
            )
        else:
            formatted.append(f'<span style="color:#000;">{line}</span>')
    return f"""
    <div style="height:400px;overflow-y:auto;padding:10px;
                background-color:#f9f9f9;font-family:monospace;font-size:13px;border-radius:8px;
                border:1px solid #e0e0e0;">
        {'<br>'.join(formatted)}
    </div>
    """

def setup_logging(log_queue: queue.Queue):
    """Configure root logger to push messages to queue."""
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S %z"))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# App
class App:
    """Pricely Dashboard."""

    def __init__(self):
        self.agent_framework: DealAgentFramework | None = None

    def framework(self) -> DealAgentFramework:
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
            self.agent_framework.init_agents_as_needed()
        return self.agent_framework

    # Helpers
    def table_data(self, opps: list[Opportunity]) -> list[list[str]]:
        return [
            [
                o.deal.product_description,
                f"${o.deal.price:.2f}",
                f"${o.estimate:.2f}",
                f"${o.discount:.2f}",
                o.deal.url
            ]
            for o in opps
        ]

    def bar_chart(self) -> go.Figure:
        opps = self.framework().memory
        if not opps:
            return go.Figure().update_layout(
                title="No data yet", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=300
            )
        names = [o.deal.product_description[:20] + "..." for o in opps]
        fig = go.Figure()
        fig.add_bar(x=names, y=[o.deal.price for o in opps], name="Price", marker_color="#1f77b4")
        fig.add_bar(x=names, y=[o.estimate for o in opps], name="Estimate", marker_color="#ff7f0e")
        fig.add_bar(x=names, y=[o.discount for o in opps], name="Discount", marker_color="#2ca02c")
        fig.update_layout(
            barmode="group",
            xaxis_title="Deal",
            yaxis_title="$ Amount",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=350,
            margin=dict(t=30, b=50),
        )
        return fig

    def vector_plot(self) -> go.Figure:
        docs, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=1000)
        fig = go.Figure(go.Scatter3d(
            x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
            mode="markers", marker=dict(size=3, color=colors, opacity=0.7), text=docs
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="manual", aspectratio=dict(x=2.2, y=2.2, z=1),
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))
            ),
            height=400,
            margin=dict(r=5, b=5, l=5, t=25),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        return fig

    # Runners
    def run_with_logging(self, logs: list[str]):
        log_q, result_q = queue.Queue(), queue.Queue()
        setup_logging(log_q)

        def worker(): result_q.put(self.table_data(self.framework().run()))
        threading.Thread(target=worker, daemon=True).start()

        initial_result = self.table_data(self.framework().memory)
        final_result = None
        while True:
            try:
                msg = log_q.get_nowait()
                logs.append(reformat(msg))
                yield logs, html_for(logs), final_result or initial_result
            except queue.Empty:
                try:
                    final_result = result_q.get_nowait()
                    yield logs, html_for(logs), final_result or initial_result
                except queue.Empty:
                    if final_result: break
                    time.sleep(0.1)

    def on_select(self, sel: gr.SelectData):
        opp = self.framework().memory[sel.index[0]]
        self.framework().planner.messenger.alert(opp)

    # Create Gradio UI
    def run(self):
        css = """
        body {background-color:#f4f5f7;font-family:'Segoe UI',Arial,sans-serif;color:#222;}
        .gr-row {gap:20px;}
        .gr-accordion {background:#ffffff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);padding:15px;}
        .gr-dataframe {background:#ffffff;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.08);
                        font-family:'Segoe UI',Arial,sans-serif;color:#111;font-size:14px;}
        .gr-plotly {border-radius:10px;background:#ffffff;box-shadow:0 4px 12px rgba(0,0,0,0.08);}
        .gr-html {background:#f9f9f9;color:#111;font-family:monospace;font-size:13px;
                  border-radius:8px;padding:10px;border:1px solid #e0e0e0;overflow-y:auto;}
        h1,h2,h3,.gr-markdown{color:#222;}
        """

        with gr.Blocks(title="Pricely Dashboard", fill_width=True, css=css) as ui:
            logs_state = gr.State([])

            # Header
            gr.Markdown('<div style="text-align:center;font-size:32px;font-weight:bold;">Pricely Dashboard</div>')
            gr.Markdown('<div style="text-align:center;font-size:14px;color:#555;">Multi-agent dashboard showing deal discovery, fine-tuned LLM price predictions, ensemble metrics, and real-time 3D embedding visualizations</div>')

            # Table
            df = gr.Dataframe(headers=["Deal", "Price", "Estimate", "Discount", "URL"],
                              wrap=True, row_count=10, col_count=5, max_height=400)

            # Bar chart
            with gr.Accordion("Price Comparison", open=True):
                bar = gr.Plot(value=self.bar_chart(), show_label=False)

            # Logs + 3D vector map
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Agent Logs (Real-Time Metrics)", open=True):
                        log_panel = gr.HTML()
                with gr.Column(scale=1):
                    with gr.Accordion("3D Vector Map", open=True):
                        vec_plot = gr.Plot(value=self.vector_plot(), show_label=False)

            # Load & refresh
            ui.load(self.run_with_logging, inputs=[logs_state], outputs=[logs_state, log_panel, df])
            timer = gr.Timer(value=300, active=True)
            timer.tick(self.run_with_logging, inputs=[logs_state], outputs=[logs_state, log_panel, df])
            df.select(self.on_select)

            # Footer
            gr.Markdown(
                """
                <p style='text-align:center; font-size:0.85em; color:gray; margin-top:2em;'>
                © 2025 Pricely Dashboard — Built by Jordan Matsumoto with Multi-Agent Framework & Gradio
                </p>
                """
            )

        ui.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    App().run()