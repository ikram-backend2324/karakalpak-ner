"""
Management command: python manage.py create_superuser_auto

Automatically creates a superuser if no superuser exists yet.
Safe to run multiple times — skips creation if a superuser already exists.

Usage:
    python manage.py create_superuser_auto
    python manage.py create_superuser_auto --username admin --password admin123 --email admin@example.com

Deployment usage (e.g. Railway, Render start command):
    python manage.py migrate && python manage.py create_superuser_auto && python manage.py runserver 0.0.0.0:$PORT

Environment variable usage (recommended for production):
    Set these env vars on your server instead of passing arguments:
        DJANGO_SUPERUSER_USERNAME=admin
        DJANGO_SUPERUSER_PASSWORD=yourpassword
        DJANGO_SUPERUSER_EMAIL=admin@example.com
    Then just run: python manage.py create_superuser_auto
"""

import os
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model


class Command(BaseCommand):
    help = 'Auto-create a superuser if none exists. Safe to run on every deployment.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            default=None,
            help='Superuser username (default: DJANGO_SUPERUSER_USERNAME env var or "admin")',
        )
        parser.add_argument(
            '--password',
            type=str,
            default=None,
            help='Superuser password (default: DJANGO_SUPERUSER_PASSWORD env var or "admin123")',
        )
        parser.add_argument(
            '--email',
            type=str,
            default=None,
            help='Superuser email (default: DJANGO_SUPERUSER_EMAIL env var or "admin@ner.local")',
        )

    def handle(self, *args, **options):
        User = get_user_model()

        # ── Check if any superuser already exists ──────────
        if User.objects.filter(is_superuser=True).exists():
            self.stdout.write(
                self.style.WARNING(
                    'Superuser already exists. Skipping creation.'
                )
            )
            return

        # ── Resolve credentials ────────────────────────────
        # Priority: CLI arg > environment variable > default
        username = (
            options.get('username')
            or os.environ.get('DJANGO_SUPERUSER_USERNAME')
            or 'admin'
        )
        password = (
            options.get('password')
            or os.environ.get('DJANGO_SUPERUSER_PASSWORD')
            or 'admin123'
        )
        email = (
            options.get('email')
            or os.environ.get('DJANGO_SUPERUSER_EMAIL')
            or 'admin@ner.local'
        )

        # ── Create superuser ───────────────────────────────
        try:
            User.objects.create_superuser(
                username=username,
                password=password,
                email=email,
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Superuser created successfully.\n"
                    f"  Username : {username}\n"
                    f"  Email    : {email}\n"
                    f"  Password : {'*' * len(password)}\n"
                    f"\n"
                    f"  Admin panel: /admin/"
                )
            )

            # Warn if using default credentials
            if username == 'admin' and password == 'admin123':
                self.stdout.write(
                    self.style.WARNING(
                        "\n  WARNING: You are using default credentials.\n"
                        "  Change them immediately after first login!\n"
                        "  Or set DJANGO_SUPERUSER_USERNAME and\n"
                        "  DJANGO_SUPERUSER_PASSWORD env variables."
                    )
                )

        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f'Failed to create superuser: {e}')
            )