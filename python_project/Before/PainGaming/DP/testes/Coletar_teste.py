from modulos.Coletar import Odd
import pytest


def test_array_float():
    """Verifica se muda o zero do array"""
    odd = Odd([0, 1.5, 0, 2.0])
    assert odd.array_float() == [1.0, 1.5, 1.0, 2.0]


def test_array_int():
    """Verifica se realiza a mudança para 0 e 1"""
    odd = Odd([1.5, 2.0, 3.5], limite=2.0)
    assert odd.array_int() == [0, 1, 1]


def test_array_int_different_limit():
    """Testa um limite diferente"""
    odd = Odd([1.5, 2.0, 3.5], limite=3.0)
    assert odd.array_int() == [0, 0, 1]


def test_empty_array():
    """Testa se retorna um array vazio caso a entrada seja vazia"""
    odd = Odd([])
    assert odd.array_float() == []
    assert odd.array_int() == []


def test_invalid_input():
    """Testa entrada invalidas no array de entrada"""
    with pytest.raises(TypeError):
        Odd(["a", 2.0, 3.5]).array_float()
