# inventory/urls.py

from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenRefreshView
from .views import bulk_action

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('add/', views.add_product, name='add_product'),
    path('update/<int:pk>/', views.update_product, name='update_product'),
    path('delete/<int:pk>/', views.delete_product, name='delete_product'),
    path('product/<int:pk>/', views.product_detail, name='product_detail'),
    path('bulk-action/', bulk_action, name='bulk_action'),
     path('import-products/', views.import_from_excel, name='import_from_excel'),
   
    path('register/', views.UserRegisterView.as_view(), name='register'),
    path('login/', views.UserLoginView.as_view(), name='login'),

]
