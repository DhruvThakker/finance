from rest_framework.response import Response
from rest_framework import serializers, views
from django.http import Http404
from .AmericanOption import *

def size_restrict(val):
    if len(val)==0:
        raise serializers.ValidationError("Error!!")
    return val

def method_restrict(val):
    if not (val=="normal" or val=="neural"):
         raise serializers.ValidationError("Only Normal and Neural are Allowed")
    return val

class BinomialSerializer(serializers.Serializer):
    stock = serializers.FloatField()
    strike = serializers.FloatField()
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

class LeastSquareSerializer(serializers.Serializer):
    method = serializers.CharField(validators=[method_restrict])
    paths = serializers.IntegerField()
    length = serializers.IntegerField()
    stock = serializers.FloatField()
    strike = serializers.FloatField()
    rate = serializers.FloatField()
    volatility = serializers.FloatField()

class FiniteDifferenceSerializer(serializers.Serializer):
    stock = serializers.FloatField()
    strike = serializers.FloatField()
    rate = serializers.FloatField()
    volatility = serializers.FloatField()
    time = serializers.FloatField()
    M = serializers.FloatField()
    N = serializers.FloatField()
    delS = serializers.FloatField()

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

class LeastSquareView(views.APIView):

    def get(self, request):
        # Validate the incoming input (provided through query parameters)
        serializer = LeastSquareSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        # Get the model input
        data = serializer.validated_data
        # Perform the complex calculations
        flag = False
        if data['method'] == 'neural':
            flag = True
        
        n_paths = data['paths']
        p_length= data['length']
        S_zero = data['stock']
        K = data['strike']
        r = data['rate']
        volatility = data['volatility']
        
        try:
            if flag:
                complex_result = getAmericanValuesLeastSquareNeural(n_paths, p_length, S_zero, K, r, volatility)
            else:
                complex_result = getAmericanValuesLeastSquareNormal(n_paths, p_length, S_zero, K, r, volatility)
        except:
            raise Http404
        
        if type(complex_result)=='List' and len(complex_result)==0:
            raise Http404

        # Return it in your custom format
        return Response({
            "complex_result": complex_result,
        })

class FiniteDifferenceView(views.APIView):

    def get(self,request):
        # Validate the incoming input (provided through query parameters)
        serializer = FiniteDifferenceSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        # Get the model input
        data = serializer.validated_data
        # Perform the complex calculations
        S0 = data['stock']
        K = data['strike']
        r = data['rate']
        sig = data['volatility']
        t = data['time']
        M = int(data['M'])
        N = int(data['N'])
        dS = data['delS']
        try:
            complex_result = getAmericanValuesFiniteDifference(S0,K,r,sig,t,M,N,dS)
        except:
            print ('Error')
            raise Http404

        # Return it in your custom format
        return Response({
            "complex_result": complex_result,
        })