
import pandas as pd, matplotlib.pyplot as plt, os, json
from pathlib import Path
from .utils import load_json
BASE = Path(__file__).resolve().parents[1]

def generate_report(output_dir='output', fmt='pdf'):
    results = load_json(BASE / 'data' / 'processed_results.json')
    df = pd.DataFrame(results)
    # basic stats
    topic_counts = df['topic'].value_counts().sort_values(ascending=False)
    # emotion per topic pivot
    pivot = pd.crosstab(df['topic'], df['emotion'])
    output_dir = BASE / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # plot topic counts
    plt.figure(figsize=(8,5))
    topic_counts.plot(kind='barh')
    plt.title("Message count per topic")
    plt.tight_layout()
    topic_plot = output_dir / 'topic_counts.png'
    plt.savefig(topic_plot)
    plt.close()
    # plot stacked bar for emotions by topic (top 10 topics)
    top_topics = topic_counts.index[:10].tolist()
    pivot_top = pivot.loc[top_topics]
    pivot_top.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title("Emotion distribution for top topics")
    plt.tight_layout()
    emo_plot = output_dir / 'emotion_by_topic.png'
    plt.savefig(emo_plot)
    plt.close()
    # create a simple PDF report by combining images and text using matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = output_dir / 'report.pdf'
    with PdfPages(pdf_path) as pdf:
        # first page: summary text
        fig, ax = plt.subplots(figsize=(8.27,11.69)) # A4
        ax.axis('off')
        ax.text(0.01,0.95,"SchoolWell Demo Report", fontsize=18, weight='bold')
        ax.text(0.01,0.92,f"Total messages: {len(df)}", fontsize=12)
        ax.text(0.01,0.88,"Top topics:", fontsize=12)
        y = 0.84
        for t,c in topic_counts.items():
            ax.text(0.02,y,f"- {t}: {c}", fontsize=10)
            y -= 0.02
            if y < 0.2: break
        pdf.savefig(fig)
        plt.close()
        # add topic plot
        fig = plt.figure(figsize=(8.27,11.69))
        img = plt.imread(topic_plot)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()
        # add emotion plot
        fig = plt.figure(figsize=(8.27,11.69))
        img = plt.imread(emo_plot)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()
        # Add example messages table
        fig, ax = plt.subplots(figsize=(8.27,11.69))
        ax.axis('off')
        ax.text(0.01,0.95,"Example classified messages:", fontsize=14)
        y = 0.92
        for i,row in df.head(12).iterrows():
            txt = f"{row['id']} | {row['topic']} | {row['emotion']} | {row['original_text']}"
            ax.text(0.01,y,txt, fontsize=9)
            y -= 0.03
            if y < 0.05: break
        pdf.savefig(fig)
        plt.close()
    return str(pdf_path)

if __name__ == '__main__':
    generate_report()
