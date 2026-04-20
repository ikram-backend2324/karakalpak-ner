"""
Custom management command: python manage.py test_ner

Usage examples:
  python manage.py test_ner
  python manage.py test_ner --text "Kompaniya bes jüz müng so'm daromad aldi."
  python manage.py test_ner --file path/to/document.txt
"""

import sys
from pathlib import Path
from django.core.management.base import BaseCommand

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.rule_ner import extract_from_sentences


SAMPLE_TEXT = """
Rayonda 2024-jılı dört müng gektar jerge egindilik ekildi.
Bu geçen jıldan jigirma bes protsent köp.
Kompaniya bes jüz müng so'm daromad aldi.
Üshinshi kvartalda saat on eki da jıyılıs bolıp ötti.
Bir butın bes million dollar investitsiya kirgizildi.
Birinshi ornında turgan talaba mükafatlandırıldı.
Bir neshe adam bu jumısqa qatnasdı.
2024-jılı 15-may da konferensiya ötti.
Ellik protsent ishtirokchilar onlayn qatnasti.
Tört müng üsh jüz ellik bes so'm tölenedi.
""".strip()

TYPE_COLORS = {
    'MON': '\033[91m',   # red
    'PCT': '\033[93m',   # yellow
    'DAT': '\033[94m',   # blue
    'TIM': '\033[96m',   # cyan
    'CNT': '\033[92m',   # green
    'FRC': '\033[95m',   # magenta
    'ORD': '\033[90m',   # dark gray
    'APX': '\033[37m',   # light gray
}
RESET = '\033[0m'
BOLD  = '\033[1m'


class Command(BaseCommand):
    help = 'Test the rule-based NER engine from the command line'

    def add_arguments(self, parser):
        parser.add_argument(
            '--text', type=str, default='',
            help='Text to analyze (uses built-in sample if omitted)'
        )
        parser.add_argument(
            '--file', type=str, default='',
            help='Path to a .txt file to analyze'
        )
        parser.add_argument(
            '--types', type=str, default='',
            help='Comma-separated list of entity types to show, e.g. MON,PCT,DAT'
        )

    def handle(self, *args, **options):
        # ── Get text ──────────────────────────────────────────
        if options['file']:
            path = Path(options['file'])
            if not path.exists():
                self.stderr.write(f"File not found: {path}")
                return
            text = path.read_text(encoding='utf-8')
        elif options['text']:
            text = options['text']
        else:
            text = SAMPLE_TEXT
            self.stdout.write(self.style.WARNING(
                "No --text or --file given. Using built-in sample text.\n"
            ))

        filter_types = (
            [t.strip().upper() for t in options['types'].split(',')]
            if options['types'] else []
        )

        # ── Run NER ───────────────────────────────────────────
        self.stdout.write(f"\n{BOLD}Input text:{RESET}")
        self.stdout.write("─" * 60)
        self.stdout.write(text[:500] + ("…" if len(text) > 500 else ""))
        self.stdout.write("─" * 60 + "\n")

        results, sentences, token_count = extract_from_sentences(text)

        if filter_types:
            results = [r for r in results if r['type'] in filter_types]

        # ── Print results ─────────────────────────────────────
        self.stdout.write(
            f"{BOLD}Found {len(results)} entities in {token_count} tokens"
            f" across {len(sentences)} sentences{RESET}\n"
        )

        if not results:
            self.stdout.write(self.style.WARNING("No entities found."))
            return

        # Table header
        col_w = [4, 6, 32, 28, 12, 6]
        sep   = "─" * (sum(col_w) + len(col_w) * 3 + 1)
        header = (
            f"{'#':>{col_w[0]}} │ {'TYPE':<{col_w[1]}} │ "
            f"{'RAW EXPRESSION':<{col_w[2]}} │ "
            f"{'NORMALIZED':<{col_w[3]}} │ "
            f"{'UNIT':<{col_w[4]}} │ "
            f"{'SENT':>{col_w[5]}}"
        )
        self.stdout.write(sep)
        self.stdout.write(f"{BOLD}{header}{RESET}")
        self.stdout.write(sep)

        for i, r in enumerate(results, 1):
            color = TYPE_COLORS.get(r['type'], '')
            raw   = r['raw'][:col_w[2]]
            fmt   = r['formatted'][:col_w[3]]
            unit  = r.get('unit', '')[:col_w[4]]
            line  = (
                f"{i:>{col_w[0]}} │ "
                f"{color}{r['type']:<{col_w[1]}}{RESET} │ "
                f"{raw:<{col_w[2]}} │ "
                f"{BOLD}{fmt:<{col_w[3]}}{RESET} │ "
                f"{unit:<{col_w[4]}} │ "
                f"{r.get('sent_idx', 0)+1:>{col_w[5]}}"
            )
            self.stdout.write(line)

        self.stdout.write(sep)

        # ── Summary by type ───────────────────────────────────
        self.stdout.write(f"\n{BOLD}Summary by type:{RESET}")
        type_counts = {}
        for r in results:
            type_counts[r['type']] = type_counts.get(r['type'], 0) + 1
        for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            color = TYPE_COLORS.get(t, '')
            bar   = '█' * cnt
            self.stdout.write(f"  {color}{t:<6}{RESET}  {bar}  {cnt}")

        self.stdout.write('')
