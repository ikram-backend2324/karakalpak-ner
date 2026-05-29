from django.db import models
from django.utils import timezone


class Analysis(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('done',       'Done'),
        ('error',      'Error'),
    ]

    filename      = models.CharField(max_length=255)
    original_name = models.CharField(max_length=255, default='')
    upload_time   = models.DateTimeField(default=timezone.now)
    status        = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    token_count   = models.IntegerField(default=0)
    result_count  = models.IntegerField(default=0)
    file_path     = models.CharField(max_length=512, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ['-upload_time']
        verbose_name        = 'Analysis'
        verbose_name_plural = 'Analyses'

    def __str__(self):
        return f"Analysis #{self.pk} — {self.original_name} ({self.status})"

    @property
    def status_badge(self):
        return {
            'done':       'success',
            'processing': 'warning',
            'error':      'danger',
        }.get(self.status, 'secondary')

    @property
    def type_counts(self):
        from django.db.models import Count
        return dict(
            self.results.values('entity_type')
                        .annotate(cnt=Count('id'))
                        .values_list('entity_type', 'cnt')
        )


class Result(models.Model):
    ENTITY_TYPES = [
        ('MON', 'Money'),
        ('PCT', 'Percent'),
        ('DAT', 'Date'),
        ('TIM', 'Time'),
        ('CNT', 'Count'),
        ('FRC', 'Fraction'),
        ('ORD', 'Ordinal'),
        ('APX', 'Approximate'),
    ]

    analysis    = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name='results')
    entity_type = models.CharField(max_length=10, choices=ENTITY_TYPES)
    raw_value   = models.TextField()
    norm_value  = models.TextField()
    unit        = models.CharField(max_length=50, blank=True)
    sent_idx    = models.IntegerField(default=0)
    start_char  = models.IntegerField(default=0)
    end_char    = models.IntegerField(default=0)

    class Meta:
        ordering = ['sent_idx', 'start_char']

    def __str__(self):
        return f"{self.entity_type}: {self.raw_value} → {self.norm_value}"

    @property
    def badge_color(self):
        colors = {
            'MON': 'danger',
            'PCT': 'warning',
            'DAT': 'primary',
            'TIM': 'info',
            'CNT': 'success',
            'FRC': 'secondary',
            'ORD': 'dark',
            'APX': 'light',
        }
        return colors.get(self.entity_type, 'secondary')
