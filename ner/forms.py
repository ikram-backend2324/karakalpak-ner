from django import forms


class UploadForm(forms.Form):
    ENTITY_CHOICES = [
        ('MON', 'Money (pul ifodalari)'),
        ('PCT', 'Percent (foizli sonlar)'),
        ('DAT', 'Date (sanalar)'),
        ('TIM', 'Time (vaqt)'),
        ('CNT', 'Count (sanoq sonlar)'),
        ('FRC', 'Fraction (kasrli sonlar)'),
        ('ORD', 'Ordinal (tartibli sonlar)'),
        ('APX', 'Approximate (taxminiy sonlar)'),
    ]

    file = forms.FileField(
        label='Select file',
        help_text='Supported formats: .txt, .docx (max 10 MB)',
        widget=forms.FileInput(attrs={
            'accept': '.txt,.docx',
            'class':  'form-control',
            'id':     'fileInput',
        })
    )

    entity_types = forms.MultipleChoiceField(
        choices=ENTITY_CHOICES,
        initial=[c[0] for c in ENTITY_CHOICES],
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label='Entity types to extract',
    )

    def clean_file(self):
        f = self.cleaned_data.get('file')
        if f:
            ext = f.name.rsplit('.', 1)[-1].lower()
            if ext not in ('txt', 'docx'):
                raise forms.ValidationError(
                    'Only .txt and .docx files are allowed.'
                )
            if f.size > 10 * 1024 * 1024:
                raise forms.ValidationError(
                    'File size must be under 10 MB.'
                )
        return f
