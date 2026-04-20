from django.urls import path
from . import views

urlpatterns = [
    path('',                              views.index,            name='index'),
    path('analyze/',                      views.analyze,          name='analyze'),
    path('result/<int:analysis_id>/',     views.result,           name='result'),
    path('history/',                      views.history,          name='history'),
    path('delete/<int:analysis_id>/',     views.delete_analysis,  name='delete_analysis'),
    path('download/json/<int:analysis_id>/',  views.download_json,  name='download_json'),
    path('download/excel/<int:analysis_id>/', views.download_excel, name='download_excel'),
    path('api/stats/',                    views.api_stats,        name='api_stats'),
    path('api/analyze/',                  views.api_analyze_text, name='api_analyze'),
]
