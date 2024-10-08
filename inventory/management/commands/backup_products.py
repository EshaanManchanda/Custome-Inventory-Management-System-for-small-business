# inventory/management/commands/backup_products.py

from django.core.management.base import BaseCommand
from inventory.models import Product
import json

class Command(BaseCommand):
    help = 'Backup all products to a JSON file'

    def handle(self, *args, **kwargs):
        products = Product.objects.all().values()
        with open('product_backup.json', 'w') as f:
            json.dump(list(products), f, indent=4)
        self.stdout.write(self.style.SUCCESS('Successfully backed up products!'))
