from rest_framework.response import Response
from rest_framework import serializers, views
from django.http import Http404
from .AmericanOption import getAmericanValuesBinomial

def size_restrict(val):
    if len(val)==0:
        raise serializers.ValidationError("You are no eligible for the job")
    return val

class BinomialSerializer(serializers.Serializer):
    stock = serializers.IntegerField()
    strike = serializers.IntegerField()
    depth = serializers.IntegerField()
    rate = serializers.FloatField()
    u = serializers.FloatField()
    d = serializers.FloatField()
    #model_input1 = serializers.IntegerField()
    '''values = serializers.ListField(
        child=serializers.IntegerField(),
        validators=[size_restrict]
    )'''
    #model_input2 = serializers.IntegerField()

class BinomialView(views.APIView):

    def get(self, request):
        # Validate the incoming input (provided through query parameters)
        serializer = BinomialSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        # Get the model input
        data = serializer.validated_data
        # Perform the complex calculations
        S = data['stock']
        de= data['depth']
        K = data['strike']
        u = data['u']
        d = data['d']
        r = data['rate']
        
        try:
            complex_result = getAmericanValuesBinomial(S,de,K,u,d,r)
        except:
            raise Http404
        
        if len(complex_result)==0:
            raise Http404

        # Return it in your custom format
        return Response({
            "complex_result": complex_result,
        })