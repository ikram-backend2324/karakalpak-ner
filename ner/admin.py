from django.contrib import admin
from .models import Analysis, Result


class ResultInline(admin.TabularInline):
    model       = Result
    extra       = 0
    readonly_fields = ('entity_type', 'raw_value', 'norm_value', 'unit', 'sent_idx')
    can_delete  = False


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display  = ('id', 'original_name', 'status', 'result_count', 'token_count', 'upload_time')
    list_filter   = ('status',)
    search_fields = ('original_name', 'filename')
    readonly_fields = ('upload_time',)
    inlines       = [ResultInline]


@admin.register(Result)
class ResultAdmin(admin.ModelAdmin):
    list_display  = ('id', 'analysis', 'entity_type', 'raw_value', 'norm_value', 'unit')
    list_filter   = ('entity_type',)
    search_fields = ('raw_value', 'norm_value')
