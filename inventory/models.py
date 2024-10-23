# inventory/models.py

from datetime import datetime
from django.db import models

class Vendor(models.Model):
    name = models.CharField(max_length=200)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    payment_upi_id = models.CharField(max_length=50, blank=True, null=True)
    
    def __str__(self):
        return self.name

class Product(models.Model):
    title = models.CharField(max_length=200)
    details = models.TextField(blank=True, null=True)
    anime_name = models.CharField(max_length=200, blank=True, null=True)
    character_name = models.CharField(max_length=200, blank=True, null=True)
    selling_price = models.DecimalField(default=0.0, max_digits=10, decimal_places=2, blank=True, null=True)
    vendor = models.CharField(max_length=200, blank=True, null=True)
    dimensions = models.CharField(default=0.0, max_length=100, blank=True, null=True)
    weight = models.DecimalField(default=0.0, max_digits=5, decimal_places=2, blank=True, null=True)
    size = models.CharField(max_length=50, blank=True, null=True)  # e.g., "10cm"
    additional_charges = models.DecimalField(default=0.0, max_digits=10, decimal_places=2, blank=True, null=True)
    in_stock = models.BooleanField(default=False)
    pre_order = models.BooleanField(default=False)
    video = models.FileField(upload_to='product_videos/', blank=True, null=True)
    # Track when the product was created
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title}"

class ProductImage(models.Model):
    product = models.ForeignKey(Product, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='product_images/', blank=True, null=True)
    
    
class VendorCost(models.Model):
    product = models.ForeignKey(Product, related_name='vendor_costs', on_delete=models.CASCADE)
    vendor = models.ForeignKey(Vendor, related_name='vendor_costs', on_delete=models.CASCADE)
    cost_price = models.DecimalField(default=0.0, max_digits=10, decimal_places=2, blank=True, null=True)

    def __str__(self):
        return f"{self.vendor.name} - {self.product.title}"