import os
import uuid
import json
from pathlib import Path
from datetime import datetime

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, FileResponse, Http404
from django.contrib import messages
from django.conf import settings
from django.utils import timezone
from django.db.models import Count, Sum

from .models import Analysis, Result
from .forms import UploadForm

# add ml/ to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.rule_ner      import extract_from_sentences
from ml.output_writer import write_json, write_excel


# ─── helpers ─────────────────────────────────────────────────────────────────

def _read_file(file_obj) -> str:
    """Read uploaded file as plain text."""
    name = file_obj.name.lower()
    if name.endswith('.docx'):
        try:
            import docx
            doc  = docx.Document(file_obj)
            return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise ValueError(f"Could not read .docx file: {e}")
    else:
        raw = file_obj.read()
        for enc in ('utf-8', 'utf-8-sig', 'cp1251', 'latin-1'):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode('utf-8', errors='replace')


def _output_dir(analysis_id: int) -> Path:
    d = Path(settings.MEDIA_ROOT) / 'outputs' / str(analysis_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── index ───────────────────────────────────────────────────────────────────

def index(request):
    form           = UploadForm()
    total_analyses = Analysis.objects.count()
    total_results  = Result.objects.count()
    recent         = Analysis.objects.filter(status='done')[:5]

    # aggregate type counts for the stats bar
    type_totals = dict(
        Result.objects.values('entity_type')
                      .annotate(cnt=Count('id'))
                      .values_list('entity_type', 'cnt')
    )

    ctx = {
        'form':           form,
        'total_analyses': total_analyses,
        'total_results':  total_results,
        'recent':         recent,
        'type_totals':    type_totals,
    }
    return render(request, 'ner/index.html', ctx)


# ─── analyze ─────────────────────────────────────────────────────────────────

def analyze(request):
    if request.method != 'POST':
        return redirect('index')

    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        for field, errors in form.errors.items():
            for err in errors:
                messages.error(request, err)
        return redirect('index')

    uploaded_file  = request.FILES['file']
    entity_types   = form.cleaned_data.get('entity_types') or [
        'MON','PCT','DAT','TIM','CNT','FRC','ORD','APX'
    ]

    # Save file to media/uploads/
    upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    save_path   = upload_dir / unique_name

    with open(save_path, 'wb') as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)

    # Create Analysis record
    analysis = Analysis.objects.create(
        filename      = unique_name,
        original_name = uploaded_file.name,
        file_path     = str(save_path),
        status        = 'processing',
    )

    # Run NER
    try:
        with open(save_path, 'rb') as f:
            # re-open as binary so _read_file handles encoding
            text = _read_file(f)

        all_results, sentences, token_count = extract_from_sentences(text)

        # Filter by selected entity types
        filtered = [r for r in all_results if r['type'] in entity_types]

        # Save to DB
        db_results = []
        for r in filtered:
            db_results.append(Result(
                analysis    = analysis,
                entity_type = r['type'],
                raw_value   = r['raw'],
                norm_value  = r['formatted'],
                unit        = r.get('unit', ''),
                sent_idx    = r.get('sent_idx', 0),
                start_char  = r.get('start', 0),
                end_char    = r.get('end', 0),
            ))
        Result.objects.bulk_create(db_results)

        # Generate output files
        out_dir = _output_dir(analysis.pk)
        meta = {
            'filename':    uploaded_file.name,
            'analyzed_at': timezone.now().strftime('%Y-%m-%d %H:%M'),
            'token_count': token_count,
            'result_count': len(filtered),
        }
        write_json(
            filtered,
            meta,
            str(out_dir / 'results.json'),
        )
        write_excel(
            filtered,
            meta,
            str(out_dir / 'results.xlsx'),
        )

        analysis.status       = 'done'
        analysis.token_count  = token_count
        analysis.result_count = len(filtered)
        analysis.save()

        messages.success(
            request,
            f"Analysis complete! Found {len(filtered)} numerical expressions."
        )

    except Exception as e:
        analysis.status        = 'error'
        analysis.error_message = str(e)
        analysis.save()
        messages.error(request, f"Analysis failed: {e}")
        return redirect('index')

    return redirect('result', analysis_id=analysis.pk)


# ─── result ──────────────────────────────────────────────────────────────────

def result(request, analysis_id):
    analysis = get_object_or_404(Analysis, pk=analysis_id)
    results  = analysis.results.all()

    # filter by type if requested
    active_type = request.GET.get('type', 'ALL')
    if active_type != 'ALL':
        results = results.filter(entity_type=active_type)

    type_counts = analysis.type_counts

    ctx = {
        'analysis':    analysis,
        'results':     results,
        'type_counts': type_counts,
        'active_type': active_type,
        'all_types':   ['MON','PCT','DAT','TIM','CNT','FRC','ORD','APX'],
    }
    return render(request, 'ner/result.html', ctx)


# ─── history ─────────────────────────────────────────────────────────────────

def history(request):
    analyses = Analysis.objects.all()[:50]
    ctx = {
        'analyses':       analyses,
        'total_done':     Analysis.objects.filter(status='done').count(),
        'total_results':  Result.objects.count(),
    }
    return render(request, 'ner/history.html', ctx)


# ─── diagrams ────────────────────────────────────────────────────────────────

def diagrams(request):
    """Analytics dashboard with charts derived from real DB data."""

    # ── aggregate counts ──────────────────────────────────────────────────────
    type_counts_qs = dict(
        Result.objects.values('entity_type')
                      .annotate(cnt=Count('id'))
                      .values_list('entity_type', 'cnt')
    )
    total_results  = sum(type_counts_qs.values())
    total_analyses = Analysis.objects.count()
    total_done     = Analysis.objects.filter(status='done').count()
    total_tokens   = Analysis.objects.aggregate(t=Sum('token_count'))['t'] or 0
    entity_types_used = len(type_counts_qs)

    # ── insight stats ─────────────────────────────────────────────────────────
    sorted_types   = sorted(type_counts_qs.items(), key=lambda x: x[1], reverse=True)
    most_used_type  = sorted_types[0][0]  if sorted_types else '—'
    most_used_count = sorted_types[0][1]  if sorted_types else 0
    rarest_type     = sorted_types[-1][0] if len(sorted_types) > 1 else '—'
    rarest_count    = sorted_types[-1][1] if len(sorted_types) > 1 else 0
    avg_per_doc     = round(total_results / total_done, 1) if total_done else 0
    extraction_rate = round(total_results / total_tokens * 100, 2) if total_tokens else 0

    # ── top files ─────────────────────────────────────────────────────────────
    top_qs    = Analysis.objects.filter(status='done').order_by('-result_count')[:5]
    max_count = top_qs[0].result_count if top_qs else 1
    top_files = []
    for a in top_qs:
        top_files.append({
            'original_name': a.original_name,
            'result_count':  a.result_count,
            'pct':           int((a.result_count / max(max_count, 1)) * 100),
        })

    # ── recent activity (last 7 completed analyses) ───────────────────────────
    recent_qs = Analysis.objects.filter(status='done').order_by('-upload_time')[:7]
    recent_activity = []
    for a in reversed(list(recent_qs)):
        recent_activity.append({
            'label':    a.upload_time.strftime('%d %b'),
            'entities': a.result_count,
            'tokens':   a.token_count,
        })

    # ── entity info reference ─────────────────────────────────────────────────
    ENTITY_META = {
        'MON': {'name': 'Money',       'example_raw': "bes jüz müng so'm", 'example_out': '500,000 UZS'},
        'PCT': {'name': 'Percent',     'example_raw': 'jigirma bes procent', 'example_out': '25.0%'},
        'DAT': {'name': 'Date',        'example_raw': "2024-jılı 15-may",    'example_out': '15.05.2024'},
        'TIM': {'name': 'Time',        'example_raw': 'saat on eki',         'example_out': '12:00'},
        'CNT': {'name': 'Count',       'example_raw': "dört müng gektar",    'example_out': '4,000 gektar'},
        'FRC': {'name': 'Fraction',    'example_raw': 'bir butın bes',       'example_out': '1.5'},
        'ORD': {'name': 'Ordinal',     'example_raw': "üshinshi",            'example_out': "3-o'rinchi"},
        'APX': {'name': 'Approximate', 'example_raw': 'bir neshe',           'example_out': '~3–5'},
    }
    entity_info = {}
    for tag, meta in ENTITY_META.items():
        cnt = type_counts_qs.get(tag, 0)
        entity_info[tag] = {
            **meta,
            'count': cnt,
            'pct':   f"{(cnt / total_results * 100):.1f}" if total_results else '0.0',
        }

    ctx = {
        'total_analyses':    total_analyses,
        'total_results':     total_results,
        'total_tokens':      total_tokens,
        'entity_types_used': entity_types_used,
        'most_used_type':    most_used_type,
        'most_used_count':   most_used_count,
        'rarest_type':       rarest_type,
        'rarest_count':      rarest_count,
        'avg_per_doc':       avg_per_doc,
        'extraction_rate':   extraction_rate,
        'top_files':         top_files,
        'entity_info':       entity_info,
        # JSON-safe blobs for the canvas charts
        'type_counts_json':    json.dumps(type_counts_qs),
        'recent_activity_json': json.dumps(recent_activity),
    }
    return render(request, 'ner/diagrams.html', ctx)


# ─── delete analysis ─────────────────────────────────────────────────────────

def delete_analysis(request, analysis_id):
    if request.method == 'POST':
        analysis = get_object_or_404(Analysis, pk=analysis_id)
        # Remove output files
        out_dir = Path(settings.MEDIA_ROOT) / 'outputs' / str(analysis_id)
        if out_dir.exists():
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        # Remove uploaded file
        if analysis.file_path and Path(analysis.file_path).exists():
            Path(analysis.file_path).unlink(missing_ok=True)
        analysis.delete()
        messages.success(request, 'Analysis deleted.')
    return redirect('history')


# ─── download views ──────────────────────────────────────────────────────────

def download_json(request, analysis_id):
    analysis = get_object_or_404(Analysis, pk=analysis_id, status='done')
    path = Path(settings.MEDIA_ROOT) / 'outputs' / str(analysis_id) / 'results.json'
    if not path.exists():
        raise Http404("JSON file not found.")
    response = FileResponse(
        open(path, 'rb'),
        content_type='application/json',
        as_attachment=True,
        filename=f"ner_results_{analysis_id}.json",
    )
    return response


def download_excel(request, analysis_id):
    analysis = get_object_or_404(Analysis, pk=analysis_id, status='done')
    path = Path(settings.MEDIA_ROOT) / 'outputs' / str(analysis_id) / 'results.xlsx'
    if not path.exists():
        raise Http404("Excel file not found.")
    response = FileResponse(
        open(path, 'rb'),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        filename=f"ner_results_{analysis_id}.xlsx",
    )
    return response


# ─── API endpoints ────────────────────────────────────────────────────────────

def api_stats(request):
    type_counts = dict(
        Result.objects.values('entity_type')
                      .annotate(cnt=Count('id'))
                      .values_list('entity_type', 'cnt')
    )
    return JsonResponse({
        'total_analyses': Analysis.objects.count(),
        'total_done':     Analysis.objects.filter(status='done').count(),
        'total_results':  Result.objects.count(),
        'by_type':        type_counts,
    })


def api_analyze_text(request):
    """Quick inline text analysis (no file upload, returns JSON)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        body = json.loads(request.body)
        text = body.get('text', '').strip()
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        if len(text) > 50_000:
            return JsonResponse({'error': 'Text too long (max 50 000 chars)'}, status=400)
        results, _, token_count = extract_from_sentences(text)
        return JsonResponse({
            'token_count': token_count,
            'count':       len(results),
            'items':       results,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
