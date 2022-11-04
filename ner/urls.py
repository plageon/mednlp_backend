from django.conf.urls import url, include
from django.urls import path

from .views import handle_ner_request,HandleNerRequest,HandlePeriodTypeRequest

urlpatterns = [
    # url(r'handle_ner_request$', handle_ner_request, ),
    # url(r'handle_ner_request$',HandleNerRequest.as_view())
    url(r'handle_period_type_request$',HandlePeriodTypeRequest.as_view())
]