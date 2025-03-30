from .Modulos.Acerto60 import calculate_orders

# Teste unitário básico
def test_calculate_orders():
    array4 = [1] * 60  # Lista com 60 elementos, todos iguais a 1
    expected_output = [59]  # A soma dos primeiros 59 elementos é 59
    array5 = calculate_orders(array4)
    assert array5 == expected_output, "Teste falhou!"

# Executar o teste
test_calculate_orders()