
from src.pipeline import run_pipeline
from src.report_generator import generate_report
import yaml, os
BASE = os.path.dirname(__file__)

if __name__ == '__main__':
    cfg = yaml.safe_load(open(os.path.join(BASE,'config','config.yaml')))
    run_pipeline(cfg)
    rpt = generate_report(output_dir=cfg.get('output_dir','output'), fmt=cfg.get('report_format','pdf'))
    print("Report generated at:", rpt)
