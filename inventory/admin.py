from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Product,Vendor,VendorCost

@admin.register(Product)
@admin.register(Vendor)
@admin.register(VendorCost)
class ProductAdmin(ImportExportModelAdmin):
    pass