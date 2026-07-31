import argparse
import json
from pathlib import Path
from tabulate import tabulate

def show(title, rows, headers):
    print()
    print(f'== {title}')
    print(tabulate(rows, headers=headers, floatfmt='.3f'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    args = ap.parse_args()
    R = Path(args.results)
    can = R / 'fullcal_canonical_b.json'
    if can.exists():
        d = json.loads(can.read_text())
        rows = []
        for a, blk in d['alphas'].items():
            for name, v in blk['methods']['primary'].items():
                if isinstance(v, dict) and 'fwe' in v:
                    rows.append([a, name, v.get('fwe'), v.get('availability'),
                                 v.get('mean_finite_reserve')])
        show('Six-book limits, main period (Table 1)', rows,
             ['target alpha', 'method', 'FWE', 'finite avail.', 'mean reserve'])
    arb = R / 'arb_margin_fullcal.json'
    if arb.exists():
        d = json.loads(arb.read_text())
        show('Worst admitted arbitrage margins (Table 2, top)',
             [[k, str(v)[:60]] for k, v in d.items() if not isinstance(v, (dict, list))],
             ['field', 'value'])
    mn = R / 'vs_multiname_joint.json'
    if mn.exists():
        d = json.loads(mn.read_text())['summary']
        rows = [[m, d['mean_coverage'][m], d['mean_admits'][m]]
                for m in ('ball', 'box', 'set')]
        show('Weekly 16-name study (Table 2, bottom)', rows,
             ['method', 'mean coverage', 'violator rate'])
    if not any((can.exists(), arb.exists(), mn.exists())):
        print('no result files found under', R, '- run the pipeline first')

if __name__ == '__main__':
    main()
