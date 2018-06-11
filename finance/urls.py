from django.urls import path
from django.contrib import admin
from django.conf.urls import url, include
from finance.compute import views
# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('admin/', admin.site.urls),
    #url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^compute/binomial/', views.BinomialView.as_view()),
    url(r'^compute/leastsquare/', views.LeastSquareView.as_view()),
]