from django.urls import path
from .views import home, train_model_api, predict_api, calculate_rfms_api, categorize_customers_api, calculate_woe_api

urlpatterns = [
    path('', home, name='home'),  # Frontend
    path('train/', train_model_api, name='train_model'),
    path('predict/', predict_api, name='predict'),
    path('rfms/', calculate_rfms_api, name='rfms'),
    path('good_bad/', categorize_customers_api, name='categorize_customers'),
    path('woe/', calculate_woe_api, name='calculate_woe'),
]
