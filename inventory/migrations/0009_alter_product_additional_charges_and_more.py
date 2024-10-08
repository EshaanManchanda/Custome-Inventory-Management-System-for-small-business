# Generated by Django 4.2.16 on 2024-10-08 02:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventory', '0008_alter_product_dimensions_alter_product_selling_price'),
    ]

    operations = [
        migrations.AlterField(
            model_name='product',
            name='additional_charges',
            field=models.DecimalField(blank=True, decimal_places=2, default=0.0, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='product',
            name='cost_price',
            field=models.DecimalField(blank=True, decimal_places=2, default=0.0, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='product',
            name='weight',
            field=models.DecimalField(blank=True, decimal_places=2, default=0.0, max_digits=5, null=True),
        ),
    ]
