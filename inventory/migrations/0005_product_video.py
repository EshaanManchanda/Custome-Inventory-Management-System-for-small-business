# Generated by Django 4.2.16 on 2024-10-07 15:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventory', '0004_alter_product_vendor_alter_productimage_image_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='video',
            field=models.FileField(blank=True, null=True, upload_to='product_videos/'),
        ),
    ]
