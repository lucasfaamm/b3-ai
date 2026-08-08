import argparse
import subprocess
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundamentals", action="store_true", help="Atualiza fundamentos antes do ranking")
    args = ap.parse_args()
    if args.fundamentals:
        subprocess.run([sys.executable, "refresh_fundamentals.py"], check=True)
    subprocess.run([sys.executable, "refresh_ranking.py"], check=True)
