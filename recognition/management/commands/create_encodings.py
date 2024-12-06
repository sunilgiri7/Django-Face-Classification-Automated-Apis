from django.core.management.base import BaseCommand
from recognition.train_v2 import train_and_generate_encodings

class Command(BaseCommand):
    help = 'Generate face encodings from the training dataset'

    def handle(self, *args, **kwargs):
        try:
            train_and_generate_encodings()
            self.stdout.write(self.style.SUCCESS('Encodings successfully generated'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {e}"))
