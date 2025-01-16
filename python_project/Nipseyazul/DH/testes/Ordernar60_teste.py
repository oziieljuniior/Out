import pytest
import numpy as np
from modulos import Ordernar60


def test_ordernar_colunas_valid_input():
    # Teste com um array divisível por 60
    array = list(range(600))  # Tamanho: 300
    obj = Ordernar60(array)
    assert obj.ordernar_colunas() == [60, 120, 180]  # Colunas divisíveis em matrizes


def test_ordernar_colunas_invalid_input():
    # Teste com um array não divisível por 60
    array = list(range(157))  # Tamanho: 157
    obj = Ordernar60(array)
    assert obj.ordernar_colunas() == []


def test_transformar_matriz():
    # Teste para verificar a transformação em matrizes
    array = list(range(300))  # Tamanho: 300
    obj = Ordernar60(array)
    matrizes = obj.tranformar()

    # Checar o número de matrizes criadas
    assert len(matrizes) == 3

    # Verificar as dimensões de cada matriz gerada
    assert matrizes[0].shape == (60, 5)
    assert matrizes[1].shape == (120, 2)
    assert matrizes[2].shape == (180, 1)


def test_transformar_empty_array():
    # Teste com um array vazio
    array = []
    obj = Ordernar60(array)
    matrizes = obj.tranformar()

    # Nenhuma matriz deve ser gerada
    assert len(matrizes) == 0


def test_transformar_invalid_array():
    # Teste com um array de tamanho pequeno (não divisível)
    array = list(range(59))  # Tamanho: 59
    obj = Ordernar60(array)
    matrizes = obj.tranformar()

    # Nenhuma matriz deve ser gerada
    assert len(matrizes) == 0


def test_transformar_correct_output():
    # Teste com uma entrada específica para validar a saída
    array = list(range(120))  # Tamanho: 120
    obj = Ordernar60(array)
    matrizes = obj.tranformar()

    # Checar a primeira matriz gerada
    matriz_esperada = np.array(array).reshape(-1, 60).T
    assert np.array_equal(matrizes[0], matriz_esperada)
