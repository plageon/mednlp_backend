import json
import traceback

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .do_period_type import do_period_type
from django.views import View

# Create your views here.
class HandleNerRequest(View):
    def post(self, request):
        response = {}
        try:
            json_body = json.loads(request.body)
            texts = json_body['texts']
            # res = do_ner(texts)
            # response['result'] = res
            response['msg'] = 'success'
            response['error_num'] = 0
        except Exception as e:
            response['msg'] = str(e)
            print(traceback.format_exc())
            response['error_num'] = 1

        return JsonResponse(response)

class HandlePeriodTypeRequest(View):
    def post(self, request):
        response = {}
        try:
            json_body = json.loads(request.body)
            texts = json_body['texts']
            res = do_period_type(texts)
            response['result'] = res
            response['msg'] = 'success'
            response['error_num'] = 0
        except Exception as e:
            response['msg'] = str(e)
            print(traceback.format_exc())
            response['error_num'] = 1

        return JsonResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def handle_ner_request(request):
    response = {}
    try:
        json_body = json.loads(request.body)
        texts = json_body['texts']
        # res = do_ner(texts)
        # response['result'] = res
        response['msg'] = 'success'
        response['error_num'] = 0
    except Exception as e:
        response['msg'] = str(e)
        print(traceback.format_exc())
        response['error_num'] = 1

    return JsonResponse(response)
